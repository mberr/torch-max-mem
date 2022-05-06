# -*- coding: utf-8 -*-

"""Tests."""

import unittest

import numpy.testing
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
