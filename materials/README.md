# Materials

The purpose of the classes in this directory is to encode the parameters of nonlinear-optical materials.
Most importantly we use these to evaluate Sellmeier equations as a function of wavelength, temperature and angle.
These values may then be used in simulations, or for temperature or angle tuning.
This module is built using Symbolic Python ([Symengine](https://symengine.org/) or [SymPy](https://www.sympy.org/)).

A material must include `ind` as a class variable, which must be a Sellmeier equation written in symbolic Python, as a function of:
- `l0`, the free space wavelength (&lambda;&#8320;),
- `T`, the temperature (if applicable),
- `th`, the angle (&theta;; if applicable),

imported from `nlMatieral.py`.

The class must have the `nlMaterial` decorator.
The dependence on temperature and angle may be turned on or off by setting `angleTuning` and `temperatureTuning` to `True` or `False` as decorator arguments.

The material class will be populated with the following class functions, as a function of the above variables:
- `n`, the index,
- `ng`, the group index,
- `beta0`, the 0th order dispersion (wavenumber),
- `beta1`, the 1st order dispersion (inverse group velocity),
- `beta2` and `gvd`, the 2nd order dispersion, or group velocity dispersion,
- `beta3`, the 3rd order dispersion.
- `walkoff` (if `angleTuning` is set), the spatial walk-off angle.
Additional information may be stored as class variables so long as they do not use these names.

For convenience, `nlMaterial` also provide the function `angledRefractiveIndex`
[cos(&theta;)&#178; / n<sub>*i*</sub>&#178; + sin(&theta;)&#178; / n<sub>*j*</sub>&#178;]<sup>-1/2</sup>
for computing the index between two crystal axes *i*, *j*.
