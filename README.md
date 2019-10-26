# NonlinearOptics

Split step method library for simulation of nonlinear optical processes in Python.
Some analysis routines are also provided, which are useful for gaussian quantum optics.

Currently the code supports a few chi(2) and chi(3) processes, 1 dimensional propagation, in the no pump depletion approximation.
This includes dispersion and nonlinear effects for arbitrary pump/signal shapes.
Higher dimensions and decoherence processes may be added as needed.

Two implementations are provided. One is a pure Python implementation with NumPy.
The other is a C++ library that can be imported in Python, that must be compiled but runs twice as fast.
The interface to the C++ version is almost identical to the Python implementation. It requires [Pybind11](https://pybind11.readthedocs.io/en/master/) for Python binding and [Eigen](http://eigen.tuxfamily.org/) for vectorized operations.

The `NonlinearMedium` (Python) or `nonlinearmedium` (C++) modules contain the classes for simulating propagation based on given parameters.
The simulations solve the dimensionless propagation equations.
Perhaps the most useful feature provided is calculating an equation's Green's function.

Also in this repository are a collection of Jupyter notebooks that test for correct behavior and reproduce the results of some relevant theory/computation papers.
These are found in the `tests/` directory, and may also serve as useful examples.

The `NonlinearHelper` module provides functions for analysis routines, for example:
- converting to dimensionless quantities,
- calculating covarience matrices and related quantities,
- homodyne detection and other squeezing related calculations,
- converting to/from the quadrature and creation bases.

The `decompositions` module is borrowed from [Strawberry Fields](https://strawberryfields.readthedocs.io/) and provides good implementations of matrix decompositions such as Bloch Messiah and Williamson.

To compile, create a build directory, from inside `cmake .. -DCMAKE_BUILD_TYPE=Release` and `make` (preferablly with some `-j`).
The binary will be created in the main directory.
