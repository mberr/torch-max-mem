<!--
<p align="center">
  <img src="https://github.com/mberr/torch-max-mem/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  torch-max-mem
</h1>

<p align="center">
    <a href="https://github.com/mberr/torch-max-mem/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/mberr/torch-max-mem/actions/workflows/tests.yml/badge.svg" /></a>
    <a href="https://pypi.org/project/torch_max_mem">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/torch_max_mem" /></a>
    <a href="https://pypi.org/project/torch_max_mem">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/torch_max_mem" /></a>
    <a href="https://github.com/mberr/torch-max-mem/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/torch_max_mem" /></a>
    <a href='https://torch_max_mem.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/torch_max_mem/badge/?version=latest' alt='Documentation Status' /></a>
    <a href="https://codecov.io/gh/mberr/torch-max-mem/branch/main">
        <img src="https://codecov.io/gh/mberr/torch-max-mem/branch/main/graph/badge.svg" alt="Codecov status" /></a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /></a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;"></a>
    <a href="https://github.com/mberr/torch-max-mem/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/></a>
    <!-- uncomment if you archive on zenodo
    <a href="https://zenodo.org/badge/latestdoi/XXXXXX">
        <img src="https://zenodo.org/badge/XXXXXX.svg" alt="DOI"></a>
    -->
</p>

This package provides decorators for memory utilization maximization with
PyTorch and CUDA by starting with a maximum parameter size and applying
successive halving until no more out-of-memory exception occurs.

## 💪 Getting Started

Assume you have a function for batched computation of nearest neighbors using
brute-force distance calculation.

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

With `torch_max_mem` you can decorate this function to reduce the batch size
until no more out-of-memory error occurs.

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

## 🚀 Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/torch_max_mem/) with uv:

```console
uv pip install torch_max_mem
```

or with pip:

```console
python3 -m pip install torch_max_mem
```

The most recent code and data can be installed directly from GitHub with uv:

```console
uv pip install git+https://github.com/mberr/torch-max-mem.git
```

or with pip:

```console
python3 -m pip install git+https://github.com/mberr/torch-max-mem.git
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are
appreciated. See
[CONTRIBUTING.md](https://github.com/mberr/torch-max-mem/blob/master/.github/CONTRIBUTING.md)
for more information on getting involved.

## 👋 Attribution

Parts of the logic have been developed with
[Laurent Vermue](https://github.com/lvermue) for
[PyKEEN](https://github.com/pykeen/pykeen).

### ⚖️ License

The code in this package is licensed under the MIT License.

### 🍪 Cookiecutter

This package was created with
[@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using
[@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack)
template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>

The final section of the README is for if you want to get involved by making a
code contribution.

### Development Installation

To install in development mode, use the following:

```console
git clone git+https://github.com/mberr/torch-max-mem.git
cd snekpack-demo
uv pip install -e .
```

Alternatively, install using pip:

```console
python3 -m pip install -e .
```

### Updating Package Boilerplate

This project uses `cruft` to keep boilerplate (i.e., configuration, contribution
guidelines, documentation configuration) up-to-date with the upstream
cookiecutter package. Install cruft with either `uv tool install cruft` or
`python3 -m pip install cruft` then run:

```console
cruft update
```

More info on Cruft's update command is available
[here](https://github.com/cruft/cruft?tab=readme-ov-file#updating-a-project).

### 🥼 Testing

After cloning the repository and installing `tox` with
`uv tool install tox --with tox-uv` or `python3 -m pip install tox tox-uv`, the
unit tests in the `tests/` folder can be run reproducibly with:

```console
tox -e py
```

Additionally, these tests are automatically re-run with each commit in a
[GitHub Action](https://github.com/mberr/torch-max-mem/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```console
git clone git+https://github.com/mberr/torch-max-mem.git
cd snekpack-demo
tox -e docs
open docs/build/html/index.html
```

The documentation automatically installs the package as well as the `docs` extra
specified in the [`pyproject.toml`](pyproject.toml). `sphinx` plugins like
`texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

The documentation can be deployed to [ReadTheDocs](https://readthedocs.io) using
[this guide](https://docs.readthedocs.io/en/stable/intro/import-guide.html). The
[`.readthedocs.yml`](.readthedocs.yml) YAML file contains all the configuration
you'll need. You can also set up continuous integration on GitHub to check not
only that Sphinx can build the documentation in an isolated environment (i.e.,
with `tox -e docs-test`) but also that
[ReadTheDocs can build it too](https://docs.readthedocs.io/en/stable/pull-requests.html).

</details>

## 🧑‍💻 For Maintainers

<details>
  <summary>See maintainer instructions</summary>

### Initial Configuration

#### Configuring ReadTheDocs

[ReadTheDocs](https://readthedocs.org) is an external documentation hosting
service that integrates with GitHub's CI/CD. Do the following for each
repository:

1. Log in to ReadTheDocs with your GitHub account to install the integration at
   https://readthedocs.org/accounts/login/?next=/dashboard/
2. Import your project by navigating to https://readthedocs.org/dashboard/import
   then clicking the plus icon next to your repository
3. You can rename the repository on the next screen using a more stylized name
   (i.e., with spaces and capital letters)
4. Click next, and you're good to go!

#### Configuring Archival on Zenodo

[Zenodo](https://zenodo.org) is a long-term archival system that assigns a DOI
to each release of your package. Do the following for each repository:

1. Log in to Zenodo via GitHub with this link:
   https://zenodo.org/oauth/login/github/?next=%2F. This brings you to a page
   that lists all of your organizations and asks you to approve installing the
   Zenodo app on GitHub. Click "grant" next to any organizations you want to
   enable the integration for, then click the big green "approve" button. This
   step only needs to be done once.
2. Navigate to https://zenodo.org/account/settings/github/, which lists all of
   your GitHub repositories (both in your username and any organizations you
   enabled). Click the on/off toggle for any relevant repositories. When you
   make a new repository, you'll have to come back to this

After these steps, you're ready to go! After you make "release" on GitHub (steps
for this are below), you can navigate to
https://zenodo.org/account/settings/github/repository/mberr/torch-max-mem to see
the DOI for the release and link to the Zenodo record for it.

#### Registering with the Python Package Index (PyPI)

The [Python Package Index (PyPI)](https://pypi.org) hosts packages so they can
be easily installed with `pip`, `uv`, and equivalent tools.

1. Register for an account [here](https://pypi.org/account/register)
2. Navigate to https://pypi.org/manage/account and make sure you have verified
   your email address. A verification email might not have been sent by default,
   so you might have to click the "options" dropdown next to your address to get
   to the "re-send verification email" button
3. 2-Factor authentication is required for PyPI since the end of 2023 (see this
   [blog post from PyPI](https://blog.pypi.org/posts/2023-05-25-securing-pypi-with-2fa/)).
   This means you have to first issue account recovery codes, then set up
   2-factor authentication
4. Issue an API token from https://pypi.org/manage/account/token

This only needs to be done once per developer.

#### Configuring your machine's connection to PyPI

This needs to be done once per machine.

```console
uv tool install keyring
keyring set https://upload.pypi.org/legacy/ __token__
keyring set https://test.pypi.org/legacy/ __token__
```

Note that this deprecates previous workflows using `.pypirc`.

### 📦 Making a Release

Publishing to PyPI happens automatically via the
[`release.yml`](.github/workflows/release.yml) GitHub Actions workflow, using
PyPI's [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).
No API token needs to be stored anywhere - the workflow only needs to be
triggered by tagging and publishing a GitHub release.

#### Tagging the release

After installing the package in development mode and installing `tox` with
`uv tool install tox --with tox-uv` or `python3 -m pip install tox tox-uv`, run
the following from the console:

```console
tox -e tag-release
```

This does the following:

1. Uses [bump-my-version](https://github.com/callowayproject/bump-my-version) to
   switch the version number in the `pyproject.toml`, `CITATION.cff`,
   `src/torch_max_mem/version.py`, and
   [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Pushes the resulting commit and tag to GitHub
3. Bumps the version to the next `-dev` patch and pushes that too. If you made
   big changes and want to bump the version by minor instead, use
   `tox -e bumpversion -- minor` after.

#### Releasing on GitHub

1. Navigate to https://github.com/mberr/torch-max-mem/releases/new to draft a
   new release
2. Click the "Choose a Tag" dropdown and select the tag corresponding to the
   release you just made
3. Click the "Generate Release Notes" button to get a quick outline of recent
   changes. Modify the title and description as you see fit
4. Click the big green "Publish Release" button

Publishing the release triggers the `release.yml` workflow, which builds the
package and uploads it to PyPI. It also triggers Zenodo to assign a DOI to your
release.

#### One-time setup: registering the trusted publisher

Before the first automated release, a repository maintainer needs to register
this workflow as a trusted publisher on PyPI:

1. On the
   [PyPI project page](https://pypi.org/manage/project/torch_max_mem/settings/publishing/)
   (or, for a brand new project, https://pypi.org/manage/account/publishing/),
   add a new GitHub publisher with:
   - Owner: `mberr`
   - Repository name: `torch-max-mem`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
2. Optionally, create a `pypi` environment under the repository's Settings ->
   Environments with required reviewers, so publishing needs manual approval.

The manual, credential-based release commands (`tox -e finish`,
`tox -e release`, `tox -e release-via-env`) are still available below as a
fallback if needed.

#### Uploading to PyPI manually

If you need to publish without going through GitHub Actions, run:

```console
tox -e finish
```

which performs the same version bump as `tox -e tag-release`, but also builds
and uploads the package directly to PyPI using
[`uv publish`](https://docs.astral.sh/uv/guides/publish/#publishing-your-package)
with credentials from `keyring`, before pushing the tag and bumping to the next
`-dev` version.

### Updating Package Boilerplate

This project uses `cruft` to keep boilerplate (i.e., configuration, contribution
guidelines, documentation configuration) up-to-date with the upstream
cookiecutter package. Install cruft with either `uv tool install cruft` or
`python3 -m pip install cruft` then run:

```console
$ cruft update
```

More info on Cruft's update command is available
[here](https://github.com/cruft/cruft?tab=readme-ov-file#updating-a-project).

</details>
