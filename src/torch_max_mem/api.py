# -*- coding: utf-8 -*-

"""
This module contains the public API.

Assume you have a function for batched computation of nearest neighbors using brute-force distance calculation.

.. code-block:: python

    import torch

    def knn(x, y, batch_size, k: int = 3):
        return torch.cat(
            [
                torch.cdist(x[start : start + batch_size], y).topk(k=k, dim=1, largest=False).indices
                for start in range(0, x.shape[0], batch_size)
            ],
            dim=0,
        )

Using :func:`maximize_memory_utilization` you can decorate this function to reduce the batch size until no more
out-of-memory error occurs.

.. code-block:: python

    import torch
    from torch_max_mem import maximize_memory_utilization

    @maximize_memory_utilization()
    def knn(x, y, batch_size, k: int = 3):
        return torch.cat(
            [
                torch.cdist(x[start : start + batch_size], y).topk(k=k, dim=0, largest=False).indices
                for start in range(0, x.shape[0], batch_size)
            ],
            dim=0,
        )


In the code, you can now always pass the largest sensible batch size, e.g.,

.. code-block:: python

    x = torch.rand(100, 100, device="cuda")
    y = torch.rand(200, 100, device="cuda")
    knn(x, y, batch_size=x.shape[0])
"""
# cf. https://gist.github.com/mberr/c37a8068b38cabc98228db2cbe358043
from __future__ import annotations

import functools
import inspect
import itertools
import logging
from typing import (
    Any,
    Callable,
    Collection,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "maximize_memory_utilization",
]

R = TypeVar("R")


def upgrade_to_sequence(
    parameter_name: str | Sequence[str], q: int | Sequence[int]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """
    Ensure that both, parameter names and q values, are provided as a sequence.

    Besides upgrading both to a tuple, it will also broadcast q if necessary.

    :param parameter_name:
        the parameter name, or a sequence thereof
    :param q:
        the q value, or a sequence thereof

    :return:
        a tuple of parameter names and a sequence of q values of same length

    :raises ValueError:
        when the (inferred) length of q and parameter_name do not match
    """
    # normalize parameter name
    parameter_names = (
        (parameter_name,) if isinstance(parameter_name, str) else tuple(parameter_name)
    )
    q = (q,) if isinstance(q, int) else tuple(q)
    q = q * len(parameter_names) if len(q) == 1 else q
    if len(q) != len(parameter_names):
        raise ValueError(f"length of {q=} does not match length of {parameter_names=}")
    return parameter_names, q


def determine_default_max_value(
    func: Callable, parameter_name: str, signature: inspect.Signature
) -> int | None:
    """
    Determine the default maximum value based on the signature.

    :param func:
        the function; only used for nice error messages
    :param parameter_name:
        the name of the parameter
    :param signature:
        the signature of the function

    :return:
        the default value as an integer, if any is given.

    :raises ValueError:
        when the function does not have a parameter of the given name
    """
    if parameter_name not in signature.parameters.keys():
        raise ValueError(f"{func} does not have a parameter {parameter_name}.")
    _parameter = signature.parameters[parameter_name]
    if _parameter.annotation != inspect.Parameter.empty and _parameter.annotation != int:
        logger.warning(
            f"Memory utilization maximization is written for integer parameters, but the "
            f"{parameter_name} is annotated as {_parameter.annotation}; casting to int",
        )
    if _parameter.default != inspect.Parameter.empty:
        return int(_parameter.default)
    return None


def determine_max_value(
    bound_arguments: inspect.BoundArguments,
    args: Sequence,
    kwargs: Mapping[str, Any],
    parameter_name: str,
    default_max_value: int | Callable[..., int] | None,
) -> int:
    """
    Either use the provided value, or the default maximum value.

    :param bound_arguments:
        the bound arguments of the function
    :param args:
        the positional parameters of the function: necessary when the default max value is a callable
    :param kwargs:
        the keyword parameters of the function: necessary when the default max value is a callable
    :param parameter_name:
        the parameter name
    :param default_max_value:
        the default max value, or a callable to determine one

    :return:
        the maximum value

    :raises ValueError:
        when the given value to the parameter is None
    """
    max_value = bound_arguments.arguments.pop(parameter_name, default_max_value)
    if max_value is None:
        raise ValueError(
            f"Invalid maximum value for parameter {parameter_name}: {max_value}",
        )
    elif callable(max_value):
        max_value = max_value(*args, **kwargs)
    return max_value


# cf. https://github.com/pykeen/pykeen/pull/279
ADDITIONAL_OOM_ERROR_INFIXES = {
    # An error that occurs because the input in CUDA is too big.
    # cf. https://discuss.pytorch.org/t/cudnn-status-not-supported-this-error-may-appear-if-you-passed-in-a-non-contiguous-input/  # noqa: E501
    "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.",
    # The torch < 2.0 way of OOM errors
    "CUDA out of memory.",
    # cf. https://github.com/pytorch/pytorch/issues/51871
    "nonzero is not supported for tensors with more than INT_MAX elements",
}


def maximize_memory_utilization_decorator(
    parameter_name: str | Sequence[str] = "batch_size",
    q: int | Sequence[int] = 32,
    cpu_warning: bool = True,
) -> Callable[[Callable[..., R]], Callable[..., Tuple[R, tuple[int, ...]]]]:
    """
    Create decorators to create methods for memory utilization maximization.

    :param parameter_name:
        The parameter name.
    :param q:
        Prefer multiples of q as size.
    :param cpu_warning:
        Whether to check the input for CPU tensors and warn about potential CPU OOM problems.

    :return:
        A decorator for functions.
    """
    if cpu_warning:

        def check_for_cpu_tensors(*args, **kwargs):
            """Check whether any tensor argument is on CPU."""
            if any(
                (torch.is_tensor(obj) and obj.device.type == "cpu")
                for obj in itertools.chain(args, kwargs.values())
            ):
                logger.warning(
                    "Using maximize_memory_utilization on non-CUDA tensors. This may lead to "
                    "undocumented crashes due to CPU OOM killer.",
                )

    else:

        def check_for_cpu_tensors(*args, **kwargs):
            """Skip checking whether any tensor argument is on CPU."""

    parameter_names, qs = upgrade_to_sequence(parameter_name, q)

    def decorator_maximize_memory_utilization(
        func: Callable[..., R],
    ) -> Callable[..., Tuple[R, tuple[int, ...]]]:
        """
        Decorate a function to maximize memory utilization.

        :param func:
            The function to decorate.

        :return:
            The decorated function.
        """
        # Input validation
        signature = inspect.signature(func)
        default_max_values = {
            name: determine_default_max_value(func=func, parameter_name=name, signature=signature)
            for name in parameter_names
        }

        @functools.wraps(func)
        def wrapper_maximize_memory_utilization(*args, **kwargs) -> Tuple[R, tuple[int, ...]]:
            """
            Wrap a function to maximize memory utilization by successive halving.

            :param args:
                The positional arguments.
            :param kwargs:
                The key-word based arguments.

            :return:
                A tuple (result, max_value).

            :raises MemoryError:
                if the execution did not even succeed with the smallest parameter value
            :raises RuntimeError:
                if a runtime error which is unrelated to known OOM errors occurred
            """
            check_for_cpu_tensors(*args, **kwargs)
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            # determine max values
            max_values = [
                determine_max_value(
                    bound_arguments=bound_arguments,
                    args=args,
                    kwargs=kwargs,
                    parameter_name=name,
                    default_max_value=default_max_value,
                )
                for name, default_max_value in default_max_values.items()
            ]
            i = 0

            while i < len(max_values):
                while max_values[i] > 0:
                    p_kwargs = {
                        name: max_value for name, max_value in zip(parameter_names, max_values)
                    }
                    try:
                        return (
                            func(*bound_arguments.args, **p_kwargs, **bound_arguments.kwargs),
                            tuple(max_values),
                        )
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                        # check for additional OOM error types
                        if not isinstance(error, torch.cuda.OutOfMemoryError) and (
                            not error.args
                            or not any(
                                infix in error.args[0] for infix in ADDITIONAL_OOM_ERROR_INFIXES
                            )
                        ):
                            raise error

                        # clear cache
                        torch.cuda.empty_cache()
                        logger.info(f"Execution failed with {p_kwargs=}")
                        max_values[i] //= 2
                        if max_values[i] > qs[i]:
                            max_values[i] = max_values[i] // qs[i] * qs[i]
                # we lowered the current parameter to 1, but still see memory issues; continue with the next in line...
                max_values[i] = 1
                i += 1
            raise MemoryError(
                f"Execution did not even succeed with {parameter_names} all equal to 1."
            )

        return wrapper_maximize_memory_utilization

    return decorator_maximize_memory_utilization


class KeyHasher:
    """A hasher based on (a subset of) keys."""

    def __init__(self, keys: Optional[Collection[str]]) -> None:
        """
        Initialize the hasher.

        :param keys:
            the keys whose associated values should be used for hashing
        """
        self.keys = keys or []

    def __call__(self, kwargs: Mapping[str, Any]) -> int:
        """
        Calculate the hash based on the values associated with the selected keys.

        :param kwargs:
            the key-value dictionary

        :return:
            the hash of the tuple of values associated with the stored keys.
        """
        return hash(tuple(kwargs.get(key, None) for key in self.keys))


class MemoryUtilizationMaximizer:
    """Stateful memory utilization maximizer."""

    def __init__(
        self,
        parameter_name: str | Sequence[str] = "batch_size",
        q: int | Sequence[int] = 32,
        cpu_warning: bool = True,
        hasher: Optional[Callable[[Mapping[str, Any]], int]] = None,
        keys: Optional[str] = None,
    ) -> None:
        """
        Initialize the stateful maximizer.

        :param parameter_name:
            The parameter name.
        :param q:
            Prefer multiples of q as size.
        :param cpu_warning:
            Whether to check the input for CPU tensors and warn about potential CPU OOM problems.
        :param hasher:
            a hashing function for separate parameter values depending on hash value; if None, use the same for all
        :param keys:
            the keys to use for creating a hasher. Only used if hasher is None.
        """
        self.parameter_names, self.qs = upgrade_to_sequence(parameter_name=parameter_name, q=q)
        self.cpu_warning = cpu_warning
        self.parameter_value: MutableMapping[int, tuple[int, ...]] = dict()
        # fixme: we do not want to include the parameter names into the hash?
        if hasher is None:
            hasher = KeyHasher(keys=keys)
        self.hasher = hasher

    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Wrap the function."""
        wrapped = maximize_memory_utilization_decorator(
            parameter_name=self.parameter_names,
            q=self.qs,
            cpu_warning=self.cpu_warning,
        )(func)
        signature = inspect.signature(func)

        @functools.wraps(wrapped)
        def inner(*args, **kwargs):
            """Evaluate function with the stored parameter size."""
            h = self.hasher(kwargs)
            if h in self.parameter_value:
                values = self.parameter_value[h]
            else:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                # todo: default logic?
                values = [bound.arguments[name] for name in self.parameter_names]
            kwargs.update(zip(self.parameter_names, values))
            result, self.parameter_value[h] = wrapped(*args, **kwargs)
            return result

        return inner


# alias
maximize_memory_utilization = MemoryUtilizationMaximizer
