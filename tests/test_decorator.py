"""Tests."""

import unittest
from typing import Optional

import pytest
import torch

from torch_max_mem import maximize_memory_utilization
from torch_max_mem.api import floor_to_nearest_multiple_of, is_oom_error, maximize_memory_utilization_decorator


def knn(x, y, batch_size, k: int = 3):
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

    def test_knn(self):
        """Test consistent results between original and wrapped method."""
        x = torch.rand(100, 100, device=self.device, generator=self.rng)
        y = torch.rand(200, 100, device=self.device, generator=self.rng)
        for batch_size in [1, 10, x.shape[0]]:
            reference = knn(x, y, batch_size)
            optimized = wrapped_knn(x, y, batch_size=x.shape[0])[0]
            assert reference.shape == optimized.shape
            assert torch.allclose(reference, optimized)

    def test_knn_stateful(self):
        """Test consistent results between original and wrapped method for stateful wrapper."""
        x = torch.rand(100, 100, device=self.device, generator=self.rng)
        y = torch.rand(200, 100, device=self.device, generator=self.rng)
        for batch_size in [1, 10, x.shape[0]]:
            reference = knn(x, y, batch_size)
            optimized = wrapped_knn_stateful(x, y, batch_size=x.shape[0])
            assert reference.shape == optimized.shape
            assert torch.allclose(reference, optimized)


def test_parameter_types():
    """Test decoration for various parameter types."""

    @maximize_memory_utilization()
    def positional_or_keyword_only_func(a, batch_size: int):
        """Evaluate a function where batch_size is a positional or keyword parameter."""

    @maximize_memory_utilization()
    def keyword_only_func(*a, batch_size: int):
        """Evaluate a function where batch_size is a keyword-only parameter."""


@pytest.mark.parametrize("keys", [None, ("a",), ("a", "b", "c")])
def test_key_hasher(keys: Optional[tuple[str, ...]]):
    """Test ad-hoc hasher."""

    def func(a, b, c, batch_size: int):
        """Test function."""
        pass

    wrapped = maximize_memory_utilization(keys=keys)(func)
    wrapped(a=1, b=3, c=7, batch_size=2)


def test_default_no_arg():
    """Test decoration's interaction with default parameters."""

    @maximize_memory_utilization()
    def func(batch_size: int = 7):
        """Test function."""

    # call with no arg
    func()


def test_optimization():
    """Test optimization."""

    @maximize_memory_utilization()
    def func(batch_size: int = 8):
        """Test function."""
        if batch_size > 2:
            raise torch.cuda.OutOfMemoryError
        return batch_size

    assert func() == 2


def test_optimization_multi_level():
    """Test optimization with multiple levels."""

    @maximize_memory_utilization(parameter_name=("batch_size", "slice_size"))
    def func(batch_size: int = 8, slice_size: int = 16):
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
    ],
)
def test_oom_error_detection(error: BaseException, exp: bool) -> None:
    """Test OOM error detection."""
    assert is_oom_error(error) is exp


@pytest.mark.slow
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS support.")
def test_large_on_mps():
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
def test_large_on_cuda():
    """Test memory optimization on a large input."""
    x = torch.rand(32_000, 100, device="cuda")
    y = torch.rand(200_000, 100, device="cuda")
    _result, (batch_size,) = wrapped_knn(x, y, batch_size=x.shape[0])
    assert batch_size < x.shape[0], "test example was too small"
    assert batch_size > 0
