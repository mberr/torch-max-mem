<!--
This part of the project documentation focuses on a
**learning-oriented** approach. You'll learn how to
get started with the code in this project.

> **Note:** Expand this section by considering the
> following points:

- Help newcomers with getting started
- Teach readers about your library by making them
    write code
- Inspire confidence through examples that work for
    everyone, repeatably
- Give readers an immediate sense of achievement
- Show concrete examples, no abstractions
- Provide the minimum necessary explanation
- Avoid any distractions
-->
This tutorial is here to help you learn how to use `torch_max_mem`, step by step.

## Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/torch_max_mem/) with:

``` shell 
pip install torch_max_mem
```

The most recent code and data can be installed directly from GitHub with:

``` shell
pip install git+https://github.com/mberr/torch-max-mem.git
```

To install in development mode, use the following:

``` shell
git clone git+https://github.com/mberr/torch-max-mem.git
cd torch-max-mem
pip install -e .
```


## First Steps
Assume you have a function for batched computation of nearest neighbors using brute-force distance calculation, e.g.,
```python
import torch

def knn(x, y, batch_size, k: int = 3):
    return torch.cat(
        [
            torch.cdist(
                x[start : start + batch_size], 
                y,
            ).topk(k=k, dim=1, largest=False).indices
            for start in range(0, x.shape[0], batch_size)
        ],
        dim=0,
    )
```

Using a larger batch size usually improves execution speed,
However, if it gets too large, you may run into out-of-memory problems.

To illustrate this problem consider running the knn example on large inputs
```python
x = torch.rand(2**16, 128, device="cuda")
y = torch.rand(2**17, 128, device="cuda")
```
When using different batch sizes on an RTX 2070 accelerator, the results are as follows

| knn           |       |       |       |       |       |
| ------------- | ----: | ----: | ----: | ----: | ----: |
| batch size    |    16 |    64 |   256 |  4096 | 65536 |
| duration [ms] |  3936 |  1503 |   946 |   784 |   OOM |

<!--
| batch size | duration [ms] |
| -- |-- |
| 16     |   3936 |
| 64     |   1503 |
| 256    |    946 |
| 1024   |    811 |
| 4096   |    784 |
| 16384  |    784 |
| 65536  |    OOM |
-->

Note how the total time generally decreases as the batch size increases, but at some point we run into out-of-memory issues.

When considering a method in isolation, it is often possible to develop some heuristics for appropriate batch sizes.
However, we often encounter similar problems in more complex systems, e.g. when running evaluation methods while training a model.
In these cases, it is often difficult to estimate a priori what the optimal batch size would be.
In these cases, we must either use a conservative batch size, which can lead to slowdowns, or implement custom fallback logic to handle out-of-memory issues.

`torch_max_mem` aims to provide reusable components to gracefully handle out-of-memory issues and reduce the batch size (or similar parameters) as needed.
In this small example, you can do this by just adding a single decorator

```python
import torch
from torch_max_mem import maximize_memory_utilization

@maximize_memory_utilization()
def knn(x, y, batch_size, k: int = 3):
    # note: the same method as before
    return torch.cat(
        [
            torch.cdist(
                x[start : start + batch_size],
                y,
            ).topk(k=k, dim=1, largest=False).indices
            for start in range(0, x.shape[0], batch_size)
        ],
        dim=0,
    )
```

In the code, you can now always pass the largest sensible batch size, e.g.,
```python
knn(x, y, batch_size=x.shape[0])
```

It will also remember the reduced batch size on the next call, so that the overhead of unsuccessful attempts at large batch sizes is paid only once.

## Multiple Parameters
!!! note
    *TODO*

## Groups
!!! note
    *TODO*