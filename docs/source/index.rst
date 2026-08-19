squarenet documentation
=======================

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Space-filling-net/SquareNet/blob/main/00_getting_started.ipynb
   :alt: Open In Colab

.. image:: https://img.shields.io/pypi/v/squarenet.svg
   :target: https://pypi.org/project/squarenet/
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/squarenet/badge/?version=latest
   :target: https://squarenet.readthedocs.io/en/latest/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/GitHub-Source-6f42c1?logo=github
   :target: https://github.com/Space-filling-net/SquareNet
   :alt: GitHub


Introduction
============

.. image:: https://raw.githubusercontent.com/Space-filling-net/SquareNet/main/plots/plot_6.png
   :alt: SquareNet example plot
   :align: center


SquareNet maps unstructured **point clouds** to structured grids through a
**bijective transformation**: one point, one cell, no overlap, fully invertible.

The practical payoff: you replace expensive spatial queries
(k-NN, radius search, neighborhood graphs) with plain tensor indexing.

Think of it as an alternative to kd-trees, voxelization, rasterization,
or graph-based approaches, but with a regular tensor structure allowing
for massive parallelisation.


Features
--------

* ✔ Works in any dimension
* ✔ Handles non-convex geometries and irregular distributions
* ✔ Scales to millions of points (seconds, not minutes)
* ✔ Compatible with PyTorch and JAX
* ✔ Native padding for mismatch between number of grid slots and number of points
  (since version 1.2)

.. image:: https://raw.githubusercontent.com/Space-filling-net/SquareNet/main/plots/raw_grided2.png
   :width: 400px
   :alt: Raw gridification example

.. toctree::
   :maxdepth: 2
   :caption: API

   api/squarenet
   api/public
   api/all

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/00_getting_started
   examples/01_some_examples
   examples/02_jax_and_pytorch
