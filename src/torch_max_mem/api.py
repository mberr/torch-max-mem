# -*- coding: utf-8 -*-

"""Main code."""
# cf. https://gist.github.com/mberr/c37a8068b38cabc98228db2cbe358043

import functools
import inspect
import itertools
import logging
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, TypeVar

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "maximize_memory_utilization",
]

R = TypeVar("R")


def _is_oom_error(error: RuntimeError) -> bool:
    """Check whether a runtime error was caused by insufficient memory."""
    message = error.args[0]
    logger.debug(f"Checking error for OOM: {message}")

    # CUDA out of memory
    if "CUDA out of memory." in message:
        return True

    # CUDA error (dimension was larger than int limit)
    if "RuntimeError: CUDA error: invalid configuration argument" in message:
        return True

    # CPU out of memory
    if "DefaultCPUAllocator: can't allocate memory:" in message:
        return True

    return False


def maximize_memory_utilization(
    parameter_name: str = "batch_size",
    q: int = 32,
    cpu_warning: bool = True,
) -> Callable[[Callable[..., R]], Callable[..., Tuple[R, int]]]:
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

    def decorator_maximize_memory_utilization(
        func: Callable[..., R],
    ) -> Callable[..., Tuple[R, int]]:
        """
        Decorate a function to maximize memory utilization.

        :param func:
            The function to decorate.

        :return:
            The decorated function.

        :raises ValueError:
            if the provided function does not contain a suitable parameter
        """
        # Input validation
        signature = inspect.signature(func)
        if parameter_name not in signature.parameters.keys():
            raise ValueError(f"{func} does not have a parameter {parameter_name}.")
        _parameter = signature.parameters[parameter_name]
        if _parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
            # TODO: we could also support positional ones by saving the position
            raise ValueError(f"{parameter_name} must be a keyword based parameter.")
        if _parameter.annotation != inspect.Parameter.empty and _parameter.annotation != int:
            logger.warning(
                f"Memory utilization maximization is written for integer parameters, but the "
                f"{parameter_name} is annotated as {_parameter.annotation}",
            )
        if _parameter.default != inspect.Parameter.empty:
            default_max_value = _parameter.default
        else:
            default_max_value = None

        @functools.wraps(func)
        def wrapper_maximize_memory_utilization(*args, **kwargs) -> Tuple[R, int]:
            """
            Wrap a function to maximize memory utilization by successive halving.

            :param args:
                The positional arguments.
            :param kwargs:
                The key-word based arguments.

            :return:
                A tuple (result, max_value).

            :raises RuntimeError:
                any runtime error which was not caused by (CUDA) OOM.
            :raises MemoryError:
                if the execution did not even succeed with the smallest parameter value
            :raises ValueError:
                if an invalid  (or no) maximum parameter value is found
            """
            check_for_cpu_tensors(*args, **kwargs)
            max_value = kwargs.pop(parameter_name, default_max_value)
            if max_value is None:
                raise ValueError(
                    f"Invalid maximum value for parameter {parameter_name}: {max_value}",
                )
            elif callable(max_value):
                max_value = max_value(*args, **kwargs)
            while max_value > 0:
                p_kwargs = {
                    parameter_name: max_value,
                }
                try:
                    return func(*args, **p_kwargs, **kwargs), max_value
                except RuntimeError as runtime_error:
                    # clear cache
                    torch.cuda.empty_cache()

                    # check whether the error is an out-of-memory error
                    if not _is_oom_error(error=runtime_error):
                        raise runtime_error

                    logger.info(f"Execution failed with {parameter_name}={max_value}")
                    max_value //= 2
                    if max_value > q:
                        max_value = max_value // q * q
            raise MemoryError(f"Execution did not even succeed with {parameter_name}=1.")

        return wrapper_maximize_memory_utilization

    return decorator_maximize_memory_utilization


class MemoryUtilizationMaximizer:
    """Stateful memory utilization maximizer."""

    def __init__(
        self,
        parameter_name: str = "batch_size",
        q: int = 32,
        cpu_warning: bool = True,
        hasher: Optional[Callable[[Mapping[str, Any]], int]] = None,
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
        """
        self.parameter_name = parameter_name
        self.q = q
        self.cpu_warning = cpu_warning
        self.parameter_value: MutableMapping[int, int] = dict()
        self.hasher = hasher

    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Wrap the function."""
        wrapped = maximize_memory_utilization(
            parameter_name=self.parameter_name,
            q=self.q,
            cpu_warning=self.cpu_warning,
        )(func)

        @functools.wraps(wrapped)
        def inner(*args, **kwargs):
            """Evaluate function with the stored parameter size."""
            h = 0 if self.hasher is None else self.hasher(kwargs)
            kwargs[self.parameter_name] = self.parameter_value.get(h) or kwargs[self.parameter_name]
            result, self.parameter_value[h] = wrapped(*args, **kwargs)
            return result

        return inner
