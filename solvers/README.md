# NonlinearOptics

Python and C++ split step method simulation library for nonlinear optical processes and Gaussian quantum optics.

The code currently supports &#43859;&#8317;&#178;&#8318; and &#43859;&#8317;&#179;&#8318; processes, with 1-dimensional propagation.
This includes dispersion/diffraction and nonlinear effects, for arbitrary pump/signal shapes.
It is straightforward to add differential equations with an arbitrary number of signal modes.

The `nonlinearmedium` module contains the classes for simulating optical propagation based on given material and optical-field parameters.
The simulations solve the dimensionless propagation equations.
Green's functions can be calculated with linear equations.
`nonlinearmedium` is a compiled C++ library meant to be imported and used in Python.
To compile, it requires [Pybind11](https://pybind11.readthedocs.io/en/master/) for Python binding and [Eigen](http://eigen.tuxfamily.org/) for vectorized operations.
It will also compile with [fftw](http://www.fftw.org/) if it is found.
The program is written using the curiously recurring template pattern to efficiently and optimally implement as many differential equation solvers as one can dream of.
The solver implementations are found in the `solver/` directory and registered in `src/nlmModulePy.cpp`.

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


---
## Solvers

`nonlinearmedium` currently contains the equation solvers described below, defined in the `solver/` directory.
They are split into two categories.
1) Linear equations, intended for simulation of quantum signals or classical light where the pump may be approximated as undepleted.
   For these equations, a Green's functions may be computed.
2) Fully nonlinear equations, where the pump is depleted.
   A Green's function may not be computed.

In these equations the sign of the nonlinear interaction L<sub>NL</sub> depends on the poling, if applicable.
D&#770; represents the differential dispersion operator for a mode
(*i [&beta;&#8321; &part;<sub>t</sub> + &beta;&#8322; &part;<sub>t</sub>&#178; + &beta;&#8323; &part;<sub>t</sub>&#179;]*).

### Linear equations

###### *Pump equation*
Unless specified otherwise, the pump propagates influenced only by dispersion, and the effective intensity scales according Rayleigh length.

<span>
A<sub>p</sub>(z, &#916;&#969;) = A<sub>p</sub>(0, &#916;&#969;) exp(i k(&#916;&#969;) z) /
&radic;<span style="text-decoration:overline;">1 + ((z-L/2) / z&#7523)&#178;</span>
</span>


###### Chi2PDC
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL</sub><sup>-1</sup> A<sub>p</sub> A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
</span>

Degenerate optical parametric amplification.

###### Chi2PDCII
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p</sub> A&#8321;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
<i>i L</i><sub>NL1</sub><sup>-1</sup> A<sub>p</sub> A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
</span>

Non-degenerate (or type II) optical parametric amplification.

###### Chi2SFG
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p</sub> A&#8321;
e<sup><i>i</i> &#916;<i>k z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
<i>i L</i><sub>NL1</sub><sup>-1</sup> A<sub>p</sub><sup>&#8224;</sup> A&#8320;
e<sup>-<i>i</i> &#916;<i>k z</i></sup>
</span>

Sum (or difference) frequency generation.

###### Chi2AFC

Adiabatic frequency conversion (*aka* adiabatic sum/difference frequency generation),
in a rotating frame with a linearly varying poling frequency built-in to the solver.
Same equation as above, but poling is disabled; intended as a faster, approximate version of Chi2SFG applied to AFC.

###### Chi2SFGII
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p</sub> A&#8323;
e<sup><i>i</i> &#916;<i>k&#8320; z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
<i>i L</i><sub>NL1</sub><sup>-1</sup> A<sub>p</sub> A&#8322;
e<sup><i>i</i> &#916;<i>k&#8321; z</i></sup>
<br>
A&#8322;'(z, t) = D&#770; A&#8322; + <i>i L</i><sub>NL2</sub><sup>-1</sup>
(A<sub>p</sub><sup>&#8224;</sup> A&#8321; e<sup>-<i>i</i> &#916;<i>k&#8321; z</i></sup> +
 A<sub>p</sub> A&#8323;<sup>&#8224;</sup> e<sup><i>i</i> &#916;<i>k&#8322; z</i></sup>)
<br>
A&#8323;'(z, t) = D&#770; A&#8323; + <i>i L</i><sub>NL3</sub><sup>-1</sup>
(A<sub>p</sub><sup>&#8224;</sup> A&#8320; e<sup>-<i>i</i> &#916;<i>k&#8320; z</i></sup> +
 A<sub>p</sub> A&#8322;<sup>&#8224;</sup> e<sup><i>i</i> &#916;<i>k&#8322; z</i></sup>)
</span>


Simultaneous sum frequency generation and non-degenerate parametric amplification.

###### Chi2SFGPDC
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p</sub> A&#8321;
e<sup><i>i</i> &#916;<i>k&#8320; z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; + 
<i>i L</i><sub>NL1</sub><sup>-1</sup>
(A<sub>p</sub><sup>&#8224;</sup> A&#8320; e<sup>-<i>i</i> &#916;<i>k&#8320; z</i></sup> +
 A<sub>p</sub> A&#8320;<sup>&#8224;</sup> e<sup><i>i</i> &#916;<i>k&#8321; z</i></sup>)
</span>

Simultaneous sum frequency generation and parametric amplification.

###### Chi3
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL</sub><sup>-1</sup>
(2|A<sub>p</sub>|&#178; A&#8320; +
 A<sub>p</sub>&#178; A&#8320;<sup>&#8224;</sup>)
<br>
A<sub>p</sub>'(z, t) = D&#770; A<sub>p</sub> +
<i>i L</i><sub>NL</sub><sup>-1</sup> |A<sub>p</sub>|&#178; A<sub>p</sub> /
&radic;<span style="text-decoration:overline;">1 + ((z-L/2) / zr)&#178;</span>
</span>

Noise reduction by self phase modulation.

###### Chi2SFGXPM
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p</sub> A&#8321;
e<sup><i>i</i> &#916;<i>k z</i></sup>
+ 2 <i>i L</i><sub>NL2</sub><sup>-1</sup> |A<sub>p</sub>|&#178; A&#8320;
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
<i>i L</i><sub>NL1</sub><sup>-1</sup> A<sub>p</sub><sup>&#8224;</sup> A&#8320;
e<sup>-<i>i</i> &#916;<i>k z</i></sup>
+ 2 <i>i L</i><sub>NL3</sub><sup>-1</sup> |A<sub>p</sub>|&#178; A&#8321;
</span>

Sum (or difference) frequency generation with cross phase modulation.

###### Chi2SFGOPA
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p0</sub> A&#8321;
e<sup><i>i</i> &#916;<i>k&#8320; z</i></sup> +
<i>i L</i><sub>NL2</sub><sup>-1</sup> A<sub>p1</sub> A&#8321;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k&#8321; z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; + 
<i>i L</i><sub>NL1</sub><sup>-1</sup>
(A<sub>p0</sub><sup>&#8224;</sup> A&#8320; e<sup>-<i>i</i> &#916;<i>k&#8320; z</i></sup> +
 A<sub>p0</sub> A&#8321;<sup>&#8224;</sup> e<sup><i>i</i> &#916;<i>k&#8322; z</i></sup>) +
<i>i L</i><sub>NL3</sub><sup>-1</sup> A<sub>p1</sub> A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k&#8321; z</i></sup>
</span>

Simultaneous sum frequency generation and non-degenerate optical parametric amplification with two pumps.


### Fully nonlinear equations

In these equations the strength of the interaction L<sub>NL</sub> scales according the Rayleigh length
(1 / &radic;<span style="text-decoration:overline;">1 + ((z-L/2) / zr)&#178;</span>).

###### Chi2DSFG
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A&#8321; A&#8322;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
<i>i L</i><sub>NL1</sub><sup>-1</sup> A&#8320; A&#8322;
e<sup>-<i>i</i> &#916;<i>k z</i></sup>
<br>
A&#8322;'(z, t) = D&#770; A&#8322; +
<i>i L</i><sub>NL2</sub><sup>-1</sup> A&#8321; A&#8320<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
</span>

Sum or difference frequency generation, or optical parametric amplification.

###### Chi2SHG
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A&#8321; A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
&frac12 <i>i L</i><sub>NL1</sub><sup>-1</sup> A&#8320;&#178;
e<sup>-<i>i</i> &#916;<i>k z</i></sup>
</span>

Second harmonic generation.

###### Chi2SHGOPA
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A&#8321; A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k&#8320; z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; + <i>i L</i><sub>NL1</sub><sup>-1</sup>
(&frac12 A&#8320;&#178; e<sup>-<i>i</i> &#916;<i>k&#8320; z</i></sup> +
 A&#8322; A&#8323; e<sup><i>i</i> &#916;<i>k&#8321; z</i></sup>)
<br>
A&#8322;'(z, t) = D&#770; A&#8322; +
<i>i L</i><sub>NL2</sub><sup>-1</sup> A&#8321; A&#8323<sup>&#8224;</sup>
e<sup>-<i>i</i> &#916;<i>k&#8321; z</i></sup>
<br>
A&#8323;'(z, t) = D&#770; A&#8323; +
<i>i L</i><sub>NL3</sub><sup>-1</sup> A&#8321; A&#8322<sup>&#8224;</sup>
e<sup>-<i>i</i> &#916;<i>k&#8321; z</i></sup>
</span>

Non-degenerate optical parametric amplification driven by second harmonic generation.

###### Chi2SHGXPM
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A&#8321; A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
+ <i>i L</i><sub>NL2</sub><sup>-1</sup> (|A&#8320;|&#178; + 2 |A&#8321;|&#178;) A&#8320;
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
&frac12 <i>i L</i><sub>NL1</sub><sup>-1</sup> A&#8320;&#178;
e<sup>-<i>i</i> &#916;<i>k z</i></sup>
+ 2 <i>i L</i><sub>NL2</sub><sup>-1</sup> (2 |A&#8320;|&#178; + |A&#8321;|&#178;) A&#8321;
</span>

Second harmonic generation with self and cross phase modulation.

###### Chi2ASHG
Adiabatic second harmonic generation, in a rotating frame with a linearly varying poling frequency built-in to the solver.
Same equation as above, but poling is disabled; intended as a faster, approximate version of Chi2SHG applied to the adiabatic case.
