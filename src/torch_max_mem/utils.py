"""Internal utility methods.

These are included in the documentation but should not be expected to have a stable API.
"""

import functools
import inspect
import itertools
import logging
from typing import Any, Callable, Collection, Iterator, Mapping, ParamSpec, Sequence, TypeVar

import torch

logger = logging.getLogger(__name__)

R = TypeVar("R")
P = ParamSpec("P")


def upgrade_to_sequence(
    parameter_name: str | Sequence[str], q: int | Sequence[int]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Ensure that both, parameter names and q values, are provided as a sequence.

    Besides upgrading both to a tuple, it will also broadcast q if necessary.

    Args:
        parameter_name: the parameter name, or a sequence thereof
        q: the q value, or a sequence thereof

    Returns:
        a tuple of parameter names and a sequence of q values of same
        length

    Raises:
        ValueError: when the (inferred) length of q and parameter_name
            do not match
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
    """Determine the default maximum value based on the signature.

    Args:
        func: the function; only used for nice error messages
        parameter_name: the name of the parameter
        signature: the signature of the function

    Returns:
        the default value as an integer, if any is given.

    Raises:
        ValueError: when the function does not have a parameter of the
            given name
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
    args: P.args,
    kwargs: P.kwargs,
    parameter_name: str,
    default_max_value: int | Callable[P, int] | None,
) -> int:
    """Either use the provided value, or the default maximum value.

    Args:
        bound_arguments: the bound arguments of the function
        args: the positional parameters of the function: necessary when
            the default max value is a callable
        kwargs: the keyword parameters of the function: necessary when
            the default max value is a callable
        parameter_name: the parameter name
        default_max_value: the default max value, or a callable to
            determine one

    Returns:
        the maximum value

    Raises:
        ValueError: when the given value to the parameter is None
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


def iter_tensor_devices(*args: Any, **kwargs: Any) -> Iterator[torch.device]:
    """Iterate over tensors' devices (may contain duplicates)."""
    for obj in itertools.chain(args, kwargs.values()):
        if isinstance(obj, torch.Tensor):
            yield obj.device


def create_tensor_checker(
    safe_devices: Collection[str] | None = None,
) -> Callable[..., None]:
    """Create a function that warns when tensors are on any device that is not considered safe.

    Args:
        safe_devices: These (PyTorch) device types are considered safe, i.e., the
            program will receive meaningful exceptions to handle out of memory (OOM) issues.
            For example for CPU, OOM errors may trigger the operating system's
            [OOM killer](https://www.kernel.org/doc/html/latest/admin-guide/mm/concepts.html#oom-killer)
            to directly terminate the process without any catchable exceptions.
            `None` defaults to ``{"cuda"}``.

    Returns:
        a function that checks its parameters for tensors and emits a
            warning if any is on a non-safe device.
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
    """Try to ensure that x is a multiple of q.

    Args:
        x: the input value
        q: the desired base factor

    Returns:
        x if x is smaller than q, otherwise, the largest multiple of q
            that is smaller than x
    """
    if x <= q:
        return x
    # note: the brackets are for readability only
    return (x // q) * q


def is_oom_error(error: BaseException) -> bool:
    """Return whether the given exception is an out-of-memory (like) exception.

    Args:
        error: the error

    Returns:
        whether it should be handled like an out-of-memory exception
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    if not error.args:
        return False
    return any(infix in error.args[0] for infix in ADDITIONAL_OOM_ERROR_INFIXES)


def maximize_memory_utilization_decorator(
    parameter_name: str | Sequence[str] = "batch_size",
    q: int | Sequence[int] = 32,
    safe_devices: Collection[str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, tuple[R, tuple[int, ...]]]]:
    """Create decorators to create methods for memory utilization maximization.

    Args:
        parameter_name: The parameter name.
        q: Prefer multiples of q as size.
        safe_devices: These devices are considered safe to run
            maximization on, cf. `create_tensor_checker`.

    Returns:
        A decorator for functions.
    """
    maybe_warn: Callable[..., None] = create_tensor_checker(safe_devices=safe_devices)
    parameter_names, qs = upgrade_to_sequence(parameter_name, q)

    def decorator_maximize_memory_utilization(
        func: Callable[P, R],
    ) -> Callable[P, tuple[R, tuple[int, ...]]]:
        """
        Decorate a function to maximize memory utilization.

        Args:
            func: The function to decorate.

        Returns:
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

            Args:
                args: The positional arguments.
                kwargs: The key-word based arguments.

            Returns:
                A tuple (result, max_value).

            Raises:
                MemoryError: if the execution did not even succeed with the smallest parameter value
                RuntimeError: if a runtime error which is unrelated to known OOM errors occurred
            """
            maybe_warn(*args, **kwargs)
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            # determine actual max values
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

            # store the last error, so we can have a nice traceback for further inspection
            last_error: BaseException | None = None

            while i < len(max_values):
                while max_values[i] > 0:
                    p_kwargs = {name: max_value for name, max_value in zip(parameter_names, max_values)}
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
    """A hasher based on (a subset of) keys.

    Args:
        keys: the keys whose associated values should be used for
            hashing
    """

    @staticmethod
    def normalize_keys(keys: Collection[str] | str | None) -> Collection[str]:
        """Normalize keys to be a collection of strings.

        Args:
            keys: the keys

        Returns:
            If `keys` is `None`, the empty list. If `keys` is a string, a singleton list with this string.
                Otherwise, `keys`.
        """
        if keys is None:
            return []
        if isinstance(keys, str):
            return [keys]
        return keys

    def __init__(self, keys: Collection[str] | str | None) -> None:
        self.keys = self.normalize_keys(keys)

    def __call__(self, kwargs: Mapping[str, Any]) -> int:
        """Calculate the hash based on the values associated with the selected keys.

        Args:
            kwargs: the key-value dictionary

        Returns:
            the hash of the tuple of values associated with the stored
            keys.
        """
        return hash(tuple(kwargs.get(key, None) for key in self.keys))