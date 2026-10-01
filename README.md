# Dynamics of Specialization in Neural Modules under Ressource Constraints

## Introduction

A self-contained implementation of the code needed to create and train modular neural-networks, with varying levels of structural modularity, to reproduce results and findings from https://arxiv.org/abs/2106.02626.

## Installation
You will need to create a new environement and install the dynspec package locally, which will take care of most requirements

```
conda create -n dynspec python=3.10
conda activate dynspec
pip install -e .
```

You will also need to install the correct pytorch version for your system separately, head over to https://pytorch.org/get-started/locally/

You're all set !

## Example
You will find a main example in the notebook modular_networks.ipynb. This notebook shows how to create and train architecture with varying parameters, and their resulting specialization levels and dynamics. It goes over all the main findings of the paper, in a straightforward fashion. 


## Held-out digit compositions

The fixed-budget uniform-versus-distance wiring experiment, preregistered decision
rule, results, and reproduction commands are in [experiment.md](experiment.md).
It extends the upstream task with explicitly held-out digit pairs using symbolic
inputs; it is not a replication of the paper's full MNIST/EMNIST study.

## uv and PyCharm

Run `uv sync --locked` from this checkout to install the project and development
tools into `.venv` (Python 3.11.9 is pinned). In PyCharm, use
`.venv/bin/python` as the project interpreter. Local run configurations are
**Composition tests** and **Composition experiment**; the experiment uses
`--output reproduced`. Choose a fresh output prefix for each additional run.

```sh
uv run --locked python -m unittest -v test_composition
uv run --locked python -m dynspec.composition --output reproduced
uv run --locked ruff check --isolated dynspec/composition.py test_composition.py
```
