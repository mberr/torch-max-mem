"""Tests."""

import unittest
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from torch_max_mem import infer_maximum_batch_size, maximize_memory_utilization
from torch_max_mem.api import floor_to_nearest_multiple_of, is_oom_error, maximize_memory_utilization_decorator


def knn(x: torch.Tensor, y: torch.Tensor, batch_size: int, k: int = 3) -> torch.Tensor:
    """Compute k-nearst neigbors via batched brute-force distance calculation."""
    return torch.cat(
        [
            torch.cdist(x[start : start + batch_size], y).topk(k=k, dim=1, largest=False).indices
            for start in range(0, x.shape[0], batch_size)
        ],
        dim=0,
    )


wrapped_knn = maximize_memory_utilization_decorator(parameter_name="batch_size")(knn)
wrapped_knn_stateful = maximize_memory_utilization()(knn)


class TestDecorator(unittest.TestCase):
    """Test the decorator."""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @property
    def rng(self) -> torch.Generator:
        """Return the random number generator."""
        return torch.Generator(device=self.device).manual_seed(42)

    def test_knn(self) -> None:
        """Test consistent results between original and wrapped method."""
        x = torch.rand(100, 100, device=self.device, generator=self.rng)
        y = torch.rand(200, 100, device=self.device, generator=self.rng)
        for batch_size in [1, 10, x.shape[0]]:
            reference = knn(x, y, batch_size)
            optimized = wrapped_knn(x, y, batch_size=x.shape[0])[0]
            assert reference.shape == optimized.shape
            assert torch.allclose(reference, optimized)

    def test_knn_stateful(self) -> None:
        """Test consistent results between original and wrapped method for stateful wrapper."""
        x = torch.rand(100, 100, device=self.device, generator=self.rng)
        y = torch.rand(200, 100, device=self.device, generator=self.rng)
        for batch_size in [1, 10, x.shape[0]]:
            reference = knn(x, y, batch_size)
            optimized = wrapped_knn_stateful(x, y, batch_size=x.shape[0])
            assert reference.shape == optimized.shape
            assert torch.allclose(reference, optimized)


def test_parameter_types() -> None:
    """Test decoration for various parameter types."""

    @maximize_memory_utilization()
    def positional_or_keyword_only_func(a: Any, batch_size: int) -> None:
        """Evaluate a function where batch_size is a positional or keyword parameter."""

    @maximize_memory_utilization()
    def keyword_only_func(*a: Any, batch_size: int) -> None:
        """Evaluate a function where batch_size is a keyword-only parameter."""


def test_stateful_positional_parameter() -> None:
    """Test that the tuned parameter can be passed positionally."""

    @maximize_memory_utilization()
    def func(x: Any, batch_size: int = 8) -> int:
        """Return the batch size."""
        return batch_size

    assert func(None, 4) == 4
    assert func(None, batch_size=4) == 4


def test_stateful_hasher_sees_positional_keys() -> None:
    """Test that hashing keys are picked up when passed positionally."""
    maximizer = maximize_memory_utilization(keys="n")

    @maximizer
    def func(n: int, batch_size: int = 1024) -> int:
        """Fail whenever the batch size exceeds n."""
        if batch_size > n:
            raise torch.cuda.OutOfMemoryError
        return batch_size

    # tune for a small n, passed positionally
    assert func(64) == 64
    # a larger n must not silently reuse the value tuned for n=64
    assert func(2048) == 1024
    assert len(maximizer.parameter_value) == 2


@pytest.mark.parametrize("keys", [None, ("a",), ("a", "b", "c")])
def test_key_hasher(keys: tuple[str, ...] | None) -> None:
    """Test ad-hoc hasher."""

    def func(a: Any, b: Any, c: Any, batch_size: int) -> None:
        """Test function."""

    wrapped = maximize_memory_utilization(keys=keys)(func)
    wrapped(a=1, b=3, c=7, batch_size=2)


def test_default_no_arg() -> None:
    """Test decoration's interaction with default parameters."""

    @maximize_memory_utilization()
    def func(batch_size: int = 7) -> None:
        """Test function."""

    # call with no arg
    func()


def test_infer_maximum_batch_size() -> None:
    """Test batch size inference from another parameter's length."""

    @infer_maximum_batch_size()
    def func(x: Sequence[Any], batch_size: int | None = None) -> int:
        """Return the batch size."""
        assert batch_size is not None
        return batch_size

    assert func(list(range(5))) == 5
    # an explicitly given batch size is not overridden
    assert func(list(range(5)), batch_size=2) == 2


def test_infer_maximum_batch_size_max_value() -> None:
    """Test that the inferred batch size is capped at max_value."""

    @infer_maximum_batch_size(max_value=3)
    def func(x: Sequence[Any], batch_size: int | None = None) -> int:
        """Return the batch size."""
        assert batch_size is not None
        return batch_size

    # inferred value is capped
    assert func(list(range(5))) == 3
    # smaller inferred value is untouched
    assert func(list(range(2))) == 2
    # an explicitly given batch size is not capped
    assert func(list(range(5)), batch_size=4) == 4


def test_infer_maximum_batch_size_custom_names() -> None:
    """Test batch size inference with non-default parameter names."""

    @infer_maximum_batch_size(parameter_name="chunk_size", x_parameter_name="y")
    def func(y: Sequence[Any], chunk_size: int | None = None) -> int:
        """Return the chunk size."""
        assert chunk_size is not None
        return chunk_size

    assert func(list(range(7))) == 7


def test_infer_maximum_batch_size_missing_parameter() -> None:
    """Test that decoration fails if the length parameter does not exist."""
    with pytest.raises(ValueError, match="does not have a parameter"):

        @infer_maximum_batch_size(x_parameter_name="does_not_exist")
        def func(x: Sequence[Any], batch_size: int | None = None) -> None:
            """Test function."""


def test_infer_maximum_batch_size_stacked() -> None:
    """Test combining batch size inference with memory utilization maximization."""
    x = torch.rand(100, 100)
    y = torch.rand(200, 100)

    @infer_maximum_batch_size()
    @maximize_memory_utilization()
    def wrapped_knn(x: torch.Tensor, y: torch.Tensor, batch_size: int, k: int = 3) -> torch.Tensor:
        """Compute k-nearest neighbors via batched brute-force distance calculation."""
        return knn(x, y, batch_size, k=k)

    reference = knn(x, y, batch_size=x.shape[0])
    # infer_maximum_batch_size preserves the wrapped function's ParamSpec, so mypy still considers batch_size
    # required here, even though it is optional at runtime
    optimized = wrapped_knn(x, y)  # type: ignore[call-arg]
    assert reference.shape == optimized.shape
    assert torch.allclose(reference, optimized)


def test_optimization() -> None:
    """Test optimization."""

    @maximize_memory_utilization()
    def func(batch_size: int = 8) -> int:
        """Test function."""
        if batch_size > 2:
            raise torch.cuda.OutOfMemoryError
        return batch_size

    assert func() == 2


def test_optimization_multi_level() -> None:
    """Test optimization with multiple levels."""

    @maximize_memory_utilization(parameter_name=("batch_size", "slice_size"))
    def func(batch_size: int = 8, slice_size: int = 16) -> tuple[int, int]:
        """Test function."""
        if batch_size > 1 or slice_size > 8:
            raise torch.cuda.OutOfMemoryError
        return batch_size, slice_size

    assert func() == (1, 8)


@pytest.mark.parametrize(("x", "q"), [(15, 4), (3, 4)])
def test_floor_to_nearest_multiple_of(x: int, q: int) -> None:
    """Test floor_to_nearest_multiple_of."""
    r = floor_to_nearest_multiple_of(x=x, q=q)
    # check type
    assert isinstance(r, int)
    # check flooring
    assert r <= x
    # check multiple of q if possible
    assert r < q or (r % q == 0)
    # check maximality
    assert r + q > x


@pytest.mark.parametrize(
    ("error", "exp"),
    [
        # base cases
        (NameError(), False),
        # CUDA
        (torch.cuda.OutOfMemoryError(), True),
        # MPS
        # cf. https://github.com/mberr/torch-max-mem/issues/14
        (RuntimeError("Invalid buffer size: 74.51 GB"), True),
        (
            RuntimeError(
                "MPS backend out of memory (MPS allocated: 119.30 MB, other allocations: 43.18 GB, max allowed: "
                "36.27 GB). Tried to allocate 4.76 MB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 "
                "to disable upper limit for memory allocations (may cause system failure).",
            ),
            True,
        ),
        # cf. https://github.com/mberr/torch-max-mem/pull/15
        (RuntimeError("selected index k out of range"), False),
        # CUDA allocator failures under memory pressure
        # cf. https://github.com/mberr/torch-max-mem/issues/45
        (
            RuntimeError(
                '!handles_.at(i) INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":430, '
                "please report a bug to PyTorch.",
            ),
            True,
        ),
        (RuntimeError("CUDA driver error: device not ready"), True),
    ],
)
def test_oom_error_detection(error: BaseException, exp: bool) -> None:
    """Test OOM error detection."""
    assert is_oom_error(error) is exp


@pytest.mark.slow
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS support.")
def test_large_on_mps() -> None:
    """Test memory optimization on a large input."""
    # note: torch.cdist calculates the pairwise distances, so its output has shape x.shape[0] * y.shape[0]
    # On MPS, it will run into a SEGFAULT when this exceeds int32, so we use a small enough input here
    x = torch.rand(21_474, 100, device="mps")
    y = torch.rand(200_000, 100, device="mps")
    _result, (batch_size,) = wrapped_knn(x, y, batch_size=x.shape[0])
    assert batch_size > 0
    assert batch_size < x.shape[0], "test example was too small"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support.")
def test_large_on_cuda() -> None:
    """Test memory optimization on a large input."""
    x = torch.rand(32_000, 100, device="cuda")
    y = torch.rand(200_000, 100, device="cuda")
    _result, (batch_size,) = wrapped_knn(x, y, batch_size=x.shape[0])
    assert batch_size < x.shape[0], "test example was too small"
    assert batch_size > 0
