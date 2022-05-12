<!--
<p align="center">
  <img src="https://github.com/mberr/torch-max-mem/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  torch-max-mem
</h1>

<p align="center">
    <a href="https://github.com/mberr/torch-max-mem/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/mberr/torch-max-mem/workflows/Tests/badge.svg" />
    </a>
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href="https://pypi.org/project/torch_max_mem">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/torch_max_mem" />
    </a>
    <a href="https://pypi.org/project/torch_max_mem">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/torch_max_mem" />
    </a>
    <a href="https://github.com/mberr/torch-max-mem/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/torch_max_mem" />
    </a>
    <a href='https://torch_max_mem.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/torch_max_mem/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
</p>

This package provides decorators for memory utilization maximization with PyTorch and CUDA by starting with a maximum parameter size and applying successive halving until no more out-of-memory exception occurs.

## 💪 Getting Started

Assume you have a function for batched computation of nearest neighbors using brute-force distance calculation.

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

With `torch_max_mem` you can decorate this function to reduce the batch size until no more out-of-memory error occurs.

```python
import torch
from torch_max_mem import maximize_memory_utilization


@maximize_memory_utilization()
def knn(x, y, batch_size, k: int = 3):
    return torch.cat(
        [
            torch.cdist(x[start : start + batch_size], y).topk(k=k, dim=0, largest=False).indices
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

## 🚀 Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/torch_max_mem/) with:

```bash
$ pip install torch_max_mem
```

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/mberr/torch-max-mem.git
```

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/mberr/torch-max-mem.git
$ cd torch-max-mem
$ pip install -e .
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/mberr/torch-max-mem/blob/master/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

Parts of the logic have been developed with [Laurent Vermue](https://github.com/lvermue) for [PyKEEN](https://github.com/pykeen/pykeen).


### ⚖️ License

The code in this package is licensed under the MIT License.

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instrutions</summary>

  
The final section of the README is for if you want to get involved by making a code contribution.

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/mberr/torch-max-mem/actions?query=workflow%3ATests).

### 📖 Building the Documentation

```shell
$ tox -e docs
``` 

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg` and
   `src/torch_max_mem/version.py` to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel
3. Uploads to PyPI using `twine`. Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.
</details>
