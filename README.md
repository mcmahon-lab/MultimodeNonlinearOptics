# NonlinearOptics

Split step method simulation library for nonlinear optical processes in Python and C++.
Analysis routines are also provided, which are useful for gaussian quantum optics.

Currently the code supports a few &#43859;&#8317;&#178;&#8318; and &#43859;&#8317;&#179;&#8318; processes, with 1-dimensional propagation, in the undepleted-pump approximation.
This includes dispersion and nonlinear effects for arbitrary pump/signal shapes.
It is straightforward to add differential equations with arbitrary number of signal modes.

The `nonlinearmedium` module contains the classes for simulating propagation based on given material parameters.
The simulations solve the dimensionless propagation equations.
Perhaps the most useful feature provided is calculating an equation's Green's function.

`nonlinearmedium` is a compiled C++ library that can be imported in Python.
To compile, it requires [Pybind11](https://pybind11.readthedocs.io/en/master/) for Python binding and [Eigen](http://eigen.tuxfamily.org/) for vectorized operations.
It will also compile with [fftw](http://www.fftw.org/) if it is found.
There is also a deprecated Python implementation using NumPy, with a similar interface to the C++ version.

This repository also contains a collection of Jupyter notebooks that test for correct behavior or reproduce the results of some relevant theory/computation papers.
These are found in the `tests/` directory, and may also serve as useful examples.
They are saved with [Jupytext](https://jupytext.readthedocs.io/en/latest/).

The `NonlinearHelper` module provides functions for configuring simulations and analysis routines, for example:
- Calculating covariance matrices and related quantities,
- Converting to dimensionless quantities,
- Generating poling patterns,
- Basis transformations.

The `decompositions` module is borrowed from [Strawberry Fields](https://strawberryfields.readthedocs.io/) with minor improvements and provides good implementations of matrix decompositions such as Takagi, Bloch-Messiah and Williamson.

The `materials` library is built using Symbolic Python, and can evaluate Sellmeier equations as a function of wavelength and temperature.
This allows easy calculation of the index, group index, group velocity dispersion, or wavenumber *&#946;* coefficients up to 3rd order
(*ie k = &#946;&#8320; + &#946;&#8321; &#916;&#969; + &#946;&#8322; &#916;&#969;&#178; + &#946;&#8323; &#916;&#969;&#179; &#8230;*).

To compile `nonlinearmedium`, create a build directory and, from inside, run `cmake .. -DCMAKE_BUILD_TYPE=Release` and `make -jX` (replacing `X` with some number of compilation cores, *eg* `-j4`).
The binary will be created in the main directory, and can be imported like a regular Python module.
