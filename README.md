# NonlinearOptics

Python and C++ split step method simulation library for nonlinear optical processes and Gaussian quantum optics.

The code currently supports &#43859;&#8317;&#178;&#8318; and &#43859;&#8317;&#179;&#8318; processes, with 1-dimensional propagation.
This includes dispersion/diffraction and nonlinear effects, for arbitrary pump/signal shapes.
It is straightforward to add differential equations with an arbitrary number of signal modes.

The `nonlinearmedium` module contains the classes for simulating optical propagation based on given material and optical-field parameters.
The simulations solve the dimensionless propagation equations.
Green's functions can be calculated with linear equations.
`nonlinearmedium` is a compiled C++ library meant to be imported and used in Python.
To compile, it requires [Pybind11](https://pybind11.readthedocs.io/) for Python binding and [Eigen](http://eigen.tuxfamily.org/) for vectorized operations.
It will also compile with [fftw](http://www.fftw.org/) if it is found.
The program is written using the curiously recurring template pattern (CRTP) to efficiently implement as many differential equation solvers as one can dream of.
The solvers are implemented and described in the `solver/` directory, and registered in `src/nlmModulePy.cpp`.

This repository also contains a collection of Jupyter notebooks that test for correct behavior or reproduce some published results.
These are found in the `tests/` directory, and may serve as useful examples.
They are saved with [Jupytext](https://jupytext.readthedocs.io/).

The `classical`, `poling` and `multimode` Python modules provide functions for configuring simulations and analysis routines, for example:
- Converting to dimensionless quantities,
- Generating poling patterns,
- Calculating covariance matrices and related quantities.

The `decompositions` module is borrowed from [Strawberry Fields](https://strawberryfields.readthedocs.io/) with minor improvements and provides good implementations of matrix decompositions.
For example, Bloch-Messiah, which is useful in Gaussian quantum optics.

The `materials` library is built using Symbolic Python ([Symengine](https://symengine.org/) or [SymPy](https://www.sympy.org/)), and can evaluate Sellmeier equations as a function of wavelength and temperature.
This allows easy calculation of the index, group index, group velocity dispersion, or wavenumber *&#946;* coefficients up to 3rd order
(*ie k = &#946;&#8320; + &#946;&#8321; &#916;&#969; + &#946;&#8322; &#916;&#969;&#178; + &#946;&#8323; &#916;&#969;&#179; &#8230;*).
It is straightforward to add materials: only a Sellmeier equation is required, Python decorators take care of the rest.

To compile `nonlinearmedium`, create a build directory and, from inside, run `cmake .. -DCMAKE_BUILD_TYPE=Release` and `make -jX` (replacing `X` with some number of compilation cores, *eg* `-j4`).
The binary will be created in the main directory, and can be imported like a regular Python module.
