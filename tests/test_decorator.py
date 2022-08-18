# -*- coding: utf-8 -*-

"""Tests."""

import unittest
from typing import Optional, Tuple
from unittest import mock

import numpy.testing
import pytest
import torch

from torch_max_mem import maximize_memory_utilization
from torch_max_mem.api import maximize_memory_utilization_decorator


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

    def test_knn(self):
        """Test consistent results between original and wrapped method."""
        x = torch.rand(100, 100, device=self.device)
        y = torch.rand(200, 100, device=self.device)
        for batch_size in [1, 10, x.shape[0]]:
            numpy.testing.assert_array_equal(
                knn(x, y, batch_size).numpy(),
                wrapped_knn(x, y, batch_size=x.shape[0])[0].numpy(),
            )

    def test_knn_stateful(self):
        """Test consistent results between original and wrapped method for stateful wrapper."""
        x = torch.rand(100, 100, device=self.device)
        y = torch.rand(200, 100, device=self.device)
        for batch_size in [1, 10, x.shape[0]]:
            numpy.testing.assert_array_equal(
                knn(x, y, batch_size).numpy(),
                wrapped_knn_stateful(x, y, batch_size=x.shape[0]).numpy(),
            )


def test_parameter_types():
    """Test decoration for various parameter types."""

    @maximize_memory_utilization()
    def positional_or_keyword_only_func(a, batch_size: int):
        """Evaluate a function where batch_size is a positional or keyword parameter."""

    @maximize_memory_utilization()
    def keyword_only_func(*a, batch_size: int):
        """Evaluate a function where batch_size is a keyword-only parameter."""


@pytest.mark.parametrize("keys", [None, ("a",), ("a", "b", "c")])
def test_key_hasher(keys: Optional[Tuple[str]]):
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

    @mock.patch("torch_max_mem.api.is_oom_error", lambda error: True)
    @maximize_memory_utilization()
    def func(batch_size: int = 8):
        """Test function."""
        if batch_size > 2:
            raise RuntimeError()
        return batch_size

    assert func() == 2
