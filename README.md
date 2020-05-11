# NonlinearOptics

Split step method library for simulation of nonlinear optical processes in Python.
Some analysis routines are also provided, which are useful for gaussian quantum optics.

Currently the code supports a few chi(2) and chi(3) processes, 1 dimensional propagation, in the no pump depletion approximation.
This includes dispersion and nonlinear effects for arbitrary pump/signal shapes.
Higher dimensions and decoherence processes may be added as needed.

Two implementations are provided. One is a Python implementation with NumPy.
The other is a C++ library that can be imported in Python, that must be compiled but runs significantly faster (x3 for PDC, x20 for SFG).
The interface to the C++ version is almost identical to the Python implementation.
To compile, it requires [Pybind11](https://pybind11.readthedocs.io/en/master/) for Python binding and [Eigen](http://eigen.tuxfamily.org/) for vectorized operations.
It will also compile with [fftw](http://www.fftw.org/) if it is found.

The `NonlinearMedium` (Python) or `nonlinearmedium` (C++) modules contain the classes for simulating propagation based on given parameters.
The simulations solve the dimensionless propagation equations.
Perhaps the most useful feature provided is calculating an equation's Green's function.

Also in this repository are a collection of Jupyter notebooks that test for correct behavior and reproduce the results of some relevant theory/computation papers.
These are found in the `tests/` directory, and may also serve as useful examples.
They are saved with [Jupytext]{https://jupytext.readthedocs.io/en/latest/}.

The `NonlinearHelper` module provides functions for analysis routines, for example:
- calculating covariance matrices and related quantities,
- homodyne detection and other squeezing-related calculations,
- converting to dimensionless quantities,
- converting to/from the quadrature and creation operator bases.

The `decompositions` module is borrowed from [Strawberry Fields](https://strawberryfields.readthedocs.io/) with minor modification and provides good implementations of matrix decompositions such as Takagi, Bloch Messiah and Williamson.

To compile, create a build directory, from inside `cmake .. -DCMAKE_BUILD_TYPE=Release` and `make` (optionally with `-j`X for speed).
The binary will be created in the main directory.
The Python implementation can be used as is.
