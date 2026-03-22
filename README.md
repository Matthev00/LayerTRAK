# LayerTRAK

[![Release](https://img.shields.io/github/v/release/SIWY-2026/LayerTRAK)](https://img.shields.io/github/v/release/SIWY-2026/LayerTRAK)
[![Build status](https://img.shields.io/github/actions/workflow/status/SIWY-2026/LayerTRAK/main.yml?branch=main)](https://github.com/SIWY-2026/LayerTRAK/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/SIWY-2026/LayerTRAK/branch/main/graph/badge.svg)](https://codecov.io/gh/SIWY-2026/LayerTRAK)
[![Commit activity](https://img.shields.io/github/commit-activity/m/SIWY-2026/LayerTRAK)](https://img.shields.io/github/commit-activity/m/SIWY-2026/LayerTRAK)
[![License](https://img.shields.io/github/license/SIWY-2026/LayerTRAK)](https://img.shields.io/github/license/SIWY-2026/LayerTRAK)

Evaluating Data Attribution with TRAK on Targeted Layers of Residual Networks

- **Github repository**: <https://github.com/SIWY-2026/LayerTRAK/>
- **Documentation** <https://SIWY-2026.github.io/LayerTRAK/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:SIWY-2026/LayerTRAK.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version



---

Repository initiated with [osprey-oss/cookiecutter-uv](https://github.com/osprey-oss/cookiecutter-uv).
