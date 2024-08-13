"""The public API."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Collection, Mapping, MutableMapping, ParamSpec, Sequence, TypeVar

from torch_max_mem.utils import KeyHasher, maximize_memory_utilization_decorator, upgrade_to_sequence

logger = logging.getLogger(__name__)

__all__ = [
    "maximize_memory_utilization",
]

R = TypeVar("R")
P = ParamSpec("P")


class MemoryUtilizationMaximizer:
    """A decorator which can be shared across multiple functions.

    The decorator reduces the monitored parameter values when an out of
    memory error occurs. Successful settings are cached to avoid having to
    re-try failing settings on successive calls.

    The decorator object allows configuration of monitored parameter names,
    rounding policies, and hashing of other parameter values to optimize
    independently for each hash digest.

    Args:
        parameter_name: The name or names of the parameters to monitor and reduce
            when out of memory errors occur.
        q: Prefer multiples of `q` when reducing parameter sizes. This can be
            advantageous on certain accelerators, e.g., for
            [Tensor Cores](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/).
            When a single `q` value is provided, it will be shared for all parameters.
        safe_devices: These (PyTorch) device types are considered safe, i.e., the
            program will receive meaningful exceptions to handle out of memory (OOM) issues.
            For example for CPU, OOM errors may trigger the operating system's
            [OOM killer](https://www.kernel.org/doc/html/latest/admin-guide/mm/concepts.html#oom-killer)
            to directly terminate the process without any catchable exceptions.
            `None` defaults to ``{"cuda"}``.
        hasher: A hashing function which determines a group index from a given key-word
            parameter dictionary. All inputs with the same group index share the safe
            parameter values. Defaults to using a single group for all inputs.
        keys: An alternative to provide a hash function, based on the tuple of parameter
            values for the given keys. Only used if `hasher` is `None`.
    """

    def __init__(
        self,
        parameter_name: str | Sequence[str] = "batch_size",
        q: int | Sequence[int] = 32,
        safe_devices: Collection[str] | None = None,
        hasher: Callable[[Mapping[str, Any]], int] | None = None,
        keys: Collection[str] | str | None = None,
    ) -> None:
        self.parameter_names, self.qs = upgrade_to_sequence(parameter_name=parameter_name, q=q)
        self.safe_devices = safe_devices
        self.parameter_value: MutableMapping[int, tuple[int, ...]] = dict()
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
        """Wrap the function with automatic parameter reduction on out of memory errors.

        Args:
            func: The function to wrap, which needs to have parameters with
                the configured names.

        Returns:
            A function with the same signature but transparent reduction of the
                monitored parameter values whenever out of memory errors occur.
        """
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
