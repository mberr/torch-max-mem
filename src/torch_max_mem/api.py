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
                torch.cdist(x[start : start + batch_size], y).topk(k=k, dim=1, largest=False).indices
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
from collections.abc import Collection, Iterable, Mapping, MutableMapping, Sequence
from typing import (
    Any,
    Callable,
    TypeVar,
)

import torch
from typing_extensions import ParamSpec

logger = logging.getLogger(__name__)

__all__ = [
    "maximize_memory_utilization",
]

R = TypeVar("R")
P = ParamSpec("P")


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
    parameter_names = (parameter_name,) if isinstance(parameter_name, str) else tuple(parameter_name)
    q = (q,) if isinstance(q, int) else tuple(q)
    q = q * len(parameter_names) if len(q) == 1 else q
    if len(q) != len(parameter_names):
        raise ValueError(f"length of {q=} does not match length of {parameter_names=}")
    return parameter_names, q


def determine_default_max_value(
    func: Callable[..., Any], parameter_name: str, signature: inspect.Signature
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
    if parameter_name not in signature.parameters:
        raise ValueError(f"{func} does not have a parameter {parameter_name}.")
    _parameter = signature.parameters[parameter_name]
    if _parameter.annotation != inspect.Parameter.empty and _parameter.annotation not in (
        int,
        "int",
    ):
        logger.warning(
            f"Memory utilization maximization is written for integer parameters, but the "
            f"{parameter_name} is annotated as {_parameter.annotation}; casting to int",
        )
    if _parameter.default != inspect.Parameter.empty:
        return int(_parameter.default)
    return None


def determine_max_value(
    bound_arguments: inspect.BoundArguments,
    parameter_name: str,
    default_max_value: int | Callable[P, int] | None,
    *args: P.args,
    **kwargs: P.kwargs,
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
    max_value = bound_arguments.arguments.get(parameter_name)
    if isinstance(max_value, int):
        return max_value
    if max_value is not None:
        raise ValueError(f"{parameter_name}={max_value!r} is neither integer nor None.")
    if default_max_value is None:
        raise ValueError("Neither value nor default value found")
    if isinstance(default_max_value, int):
        return default_max_value
    return default_max_value(*args, **kwargs)


# cf. https://github.com/pykeen/pykeen/pull/279
ADDITIONAL_OOM_ERROR_INFIXES = {
    # An error that occurs because the input in CUDA is too big.
    # cf. https://discuss.pytorch.org/t/cudnn-status-not-supported-this-error-may-appear-if-you-passed-in-a-non-contiguous-input/  # noqa: E501
    "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.",
    # The torch < 2.0 way of OOM errors
    "CUDA out of memory.",
    # cf. https://github.com/pytorch/pytorch/issues/51871
    "nonzero is not supported for tensors with more than INT_MAX elements",
    # cf. https://discuss.pytorch.org/t/runtime-error-invalid-buffer-size-when-calculating-cosine-similarity/152088
    "Invalid buffer size: ",
    # MPS OOM error
    "MPS backend out of memory",
    # CPU OOM error
    "DefaultCPUAllocator: not enough memory:",
}


def iter_tensor_devices(*args: Any, **kwargs: Any) -> Iterable[torch.device]:
    """Iterate over tensors' devices (may contain duplicates)."""
    for obj in itertools.chain(args, kwargs.values()):
        if isinstance(obj, torch.Tensor):
            yield obj.device


def create_tensor_checker(
    safe_devices: Collection[str] | None = None,
) -> Callable[P, None]:
    """
    Create a function that warns when tensors are on any device that is not considered safe.

    :param safe_devices:
        these devices are considered safe, i.e., the program will receive meaningful exceptions to handle out of memory
        (OOM) issues. For example for CPU, OOM errors may trigger the operating system's OOM killer to directly
        terminate the process without any catchable exceptions. Defaults to ``{"cuda"}``.

    :return:
        a function that checks its parameters for tensors and emits a warning if any is on a non-safe device.
    """
    if safe_devices is None:
        safe_devices = {"cuda"}
    safe_devices_set = frozenset(safe_devices)
    logger.debug(
        f"Will warn about running memory utilization maximization on tensors on devices other than {safe_devices_set}",
    )

    def check_tensors(*args: P.args, **kwargs: P.kwargs) -> None:
        """Check whether any tensor argument is on a dangerous device."""
        device_types = {device.type for device in iter_tensor_devices(*args, **kwargs)}

        if not safe_devices_set.issuperset(device_types):
            logger.warning(
                f"Encountered tensors on {device_types=} while only {sorted(safe_devices_set)} are considered safe for "
                f"automatic memory utilization maximization. This may lead to undocumented crashes (but can be safe, "
                f"too).",
            )

    return check_tensors


def floor_to_nearest_multiple_of(x: int, q: int) -> int:
    """
    Try to ensure that x is a multiple of q.

    :param x:
        the input value
    :param q:
        the desired base factor

    :return:
        x if x is smaller than q, otherwise, the largest multiple of q that is smaller than x
    """
    if x <= q:
        return x
    # note: the brackets are for readability only
    return (x // q) * q


def is_oom_error(error: BaseException) -> bool:
    """
    Return whether the given exception is an out-of-memory (like) exception.

    :param error:
        the error

    :return:
        whether it should be handled like an out-of-memory exception
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error)
    return any(infix in message for infix in ADDITIONAL_OOM_ERROR_INFIXES)


def maximize_memory_utilization_decorator(
    parameter_name: str | Sequence[str] = "batch_size",
    q: int | Sequence[int] = 32,
    safe_devices: Collection[str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, tuple[R, tuple[int, ...]]]]:
    """
    Create decorators to create methods for memory utilization maximization.

    :param parameter_name:
        The parameter name.
    :param q:
        Prefer multiples of q as size.
    :param safe_devices:
        These devices are considered safe to run maximization on, cf. :meth:`create_tensor_checker`.

    :return:
        A decorator for functions.
    """
    maybe_warn: Callable[..., None] = create_tensor_checker(safe_devices=safe_devices)
    parameter_names, qs = upgrade_to_sequence(parameter_name, q)

    def decorator_maximize_memory_utilization(
        func: Callable[P, R],
    ) -> Callable[P, tuple[R, tuple[int, ...]]]:
        """
        Decorate a function to maximize memory utilization.

        :param func:
            The function to decorate.

        :return:
            The decorated function.
        """
        # Input validation, and extraction of default maximum values
        signature = inspect.signature(func)
        default_max_values = {
            name: determine_default_max_value(func=func, parameter_name=name, signature=signature)
            for name in parameter_names
        }

        @functools.wraps(func)
        def wrapper_maximize_memory_utilization(*args: P.args, **kwargs: P.kwargs) -> tuple[R, tuple[int, ...]]:
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
            maybe_warn(*args, **kwargs)
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            # determine actual max values
            max_values = [
                determine_max_value(
                    bound_arguments,
                    name,
                    default_max_value,
                    *args,
                    **kwargs,
                )
                for name, default_max_value in default_max_values.items()
            ]
            i = 0

            # store the last error, so we can have a nice traceback for further inspection
            last_error: BaseException | None = None

            while i < len(max_values):
                while max_values[i] > 0:
                    p_kwargs = dict(zip(parameter_names, max_values))
                    # note: changes to arguments apply to both, .args and .kwargs
                    bound_arguments.arguments.update(p_kwargs)
                    try:
                        return func(*bound_arguments.args, **bound_arguments.kwargs), tuple(max_values)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                        # raise errors unrelated to out-of-memory
                        if not is_oom_error(error):
                            raise error

                        # clear cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # https://pytorch.org/docs/stable/notes/mps.html
                        if torch.backends.mps.is_available():
                            # there is no torch.mps.is_available()
                            torch.mps.empty_cache()

                        # reduce parameter
                        logger.info(f"Execution failed with {p_kwargs=}")
                        max_values[i] = floor_to_nearest_multiple_of(x=max_values[i] // 2, q=qs[i])

                        # update last error
                        last_error = error
                # we lowered the current parameter to 1, but still see memory issues; continue with the next in line...
                max_values[i] = 1
                i += 1
            # log memory summary for each CUDA device before raising memory error
            for device in {d for d in iter_tensor_devices(*args, **kwargs) if d.type == "cuda"}:
                logger.debug(f"Memory summary for {device=}:\n{torch.cuda.memory_summary(device=device)}")
            raise MemoryError(f"Execution did not even succeed with {parameter_names} all equal to 1.") from last_error

        return wrapper_maximize_memory_utilization

    return decorator_maximize_memory_utilization


class KeyHasher:
    """A hasher based on (a subset of) keys."""

    @staticmethod
    def normalize_keys(keys: Collection[str] | str | None) -> Collection[str]:
        """
        Normalize keys to be a collection of strings.

        :param keys:
            the keys

        :return:
            - if keys is None, the empty list
            - if keys is a string, a singleton list
            - else the keys
        """
        if keys is None:
            return []
        if isinstance(keys, str):
            return [keys]
        return keys

    def __init__(self, keys: Collection[str] | str | None) -> None:
        """
        Initialize the hasher.

        :param keys:
            the keys whose associated values should be used for hashing
        """
        self.keys = self.normalize_keys(keys)

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
        safe_devices: Collection[str] | None = None,
        hasher: Callable[[Mapping[str, Any]], int] | None = None,
        keys: Collection[str] | str | None = None,
    ) -> None:
        """
        Initialize the stateful maximizer.

        :param parameter_name:
            The parameter name.
        :param q:
            Prefer multiples of q as size.
        :param safe_devices:
            These devices are considered safe to run maximization on, cf. :meth:`create_tensor_checker`.
        :param hasher:
            a hashing function for separate parameter values depending on hash value; if None, use the same for all
        :param keys:
            the keys to use for creating a hasher. Only used if hasher is None.
        """
        self.parameter_names, self.qs = upgrade_to_sequence(parameter_name=parameter_name, q=q)
        self.safe_devices = safe_devices
        self.parameter_value: MutableMapping[int, tuple[int, ...]] = {}
        if hasher is None:
            keys = KeyHasher.normalize_keys(keys)
            intersection = set(keys).intersection(self.parameter_names)
            if intersection:
                logger.warning(
                    f"{intersection=} are contained in the hashing keys *and* the parameter names; "
                    f"likely you want to remove {intersection} from hashing keys.",
                )
            hasher = KeyHasher(keys=keys)
        self.hasher = hasher

    def __call__(self, func: Callable[P, R]) -> Callable[P, R]:
        """Wrap the function."""
        wrapped = maximize_memory_utilization_decorator(
            parameter_name=self.parameter_names,
            q=self.qs,
            safe_devices=self.safe_devices,
        )(func)
        signature = inspect.signature(func)

        @functools.wraps(wrapped)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            """Evaluate function with the stored parameter size."""
            h = self.hasher(kwargs)
            if h in self.parameter_value:
                values = self.parameter_value[h]
            else:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                # todo: default logic?
                values = tuple(bound.arguments[name] for name in self.parameter_names)
            kwargs.update(zip(self.parameter_names, values))
            result, self.parameter_value[h] = wrapped(*args, **kwargs)
            return result

        return inner


# alias
maximize_memory_utilization = MemoryUtilizationMaximizer
