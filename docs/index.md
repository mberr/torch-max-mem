This site contains the project documentation for the
`torch_max_mem` project.

## Table Of Contents

The documentation follows the best practice for
project documentation as described by Daniele Procida
in the [Diátaxis documentation framework](https://diataxis.fr/)
and consists of four separate parts:

1. [Tutorials](tutorials.md)
2. [How-To Guides](how-to-guides.md)
3. [Reference](reference.md)
4. [Explanation](explanation.md)

Quickly find what you're looking for depending on
your use case by looking at the different pages.


## Project Overview
Assume you have a function for batched computation of nearest neighbors using brute-force distance calculation, e.g.,
```python
import torch

def knn(x, y, batch_size, k: int = 3):
    return torch.cat(
        [
            torch.cdist(x[start : start + batch_size], y).topk(k=k, dim=1, largest=False).indices
            for start in range(0, x.shape[0], batch_size)
        ],
        dim=0,
    )
```


Using `maximize_memory_utilization` you can decorate this function to reduce the batch size until no more
out-of-memory error occurs.

```python

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
```

In the code, you can now always pass the largest sensible batch size, e.g.,

```python
x = torch.rand(100, 100, device="cuda")
y = torch.rand(200, 100, device="cuda")
knn(x, y, batch_size=x.shape[0])
```

## Acknowledgements

The documentation was build using the
Real Python tutorial
[Build Your Python Project Documentation With MkDocs](
    https://realpython.com/python-project-documentation-with-mkdocs/).

Parts of the logic have been developed together with
[Laurent Vermue](https://github.com/lvermue) for [PyKEEN](https://github.com/pykeen/pykeen).