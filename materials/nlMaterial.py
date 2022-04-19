try:
  from symengine import symbols, diff, pi, Lambdify
  def lambdify(*funcArgs):
    f = Lambdify(*funcArgs)
    return lambda *callArgs : float(f(*callArgs))
except:
  from sympy import symbols, diff, pi, lambdify

l0 = symbols("l0") # Lambda_0, free space wavelength
T = symbols("T")   # Temperature
th = symbols("th") # Theta, tuning angle

# Decorator does all the math for the material classes
def nlMaterial(deleteInd=True, angleTuning=False, temperatureTuning=True):
  """
  Decorator that creates classes for storing and calculating the properties of different materials simulations.
  Note: for decorator to work, class must have properties:
  - ind: a sympy symbolic expression as a function of "l0" and "T" indicating the wavelength in um and temperature in C.
  Optional Parameters:
    deleteInd: Remove the Symbolic Sellmeier expressions from the class
    angleTuning: Include an angle argument to the class functions (critical phase matching angle)
    temperatureTuning: Include a temperature argument to the class functions
  """
  def nlmDecorator(cls):
    methodArguments = ([th] if angleTuning else []) + [l0] + ([T] if temperatureTuning else [])

    if not hasattr(cls, "ind"):
      raise AttributeError("Class does not have an index expression!")

    c = 299792458 # m / s

    groupInd = cls.ind - l0 * diff(cls.ind, l0)
    GVD = -l0**2 / (1e6 * 2 * pi * c**2) * diff(groupInd, l0)

    cls.n = lambdify(methodArguments, cls.ind)
    cls.n.__doc__ = "Calculate refractive index n from the Sellmeier equation"
    cls.n = staticmethod(cls.n)

    cls.ng = lambdify(methodArguments, groupInd)
    cls.ng.__doc__ = "Calculate the group index from the Sellmeier equation"
    cls.ng = staticmethod(cls.ng)

    cls.gvd = lambdify(methodArguments, GVD)
    cls.gvd.__doc__ = "Calculate 2nd order dispersion (group dispersion) from the Sellmeier equation"
    cls.gvd = staticmethod(cls.gvd)
    cls.beta2 = cls.gvd

    cls.beta1 = lambdify(methodArguments, groupInd / c)
    cls.beta1.__doc__ = "Calculate 1st order dispersion (inverse group velocity) from the Sellmeier equation"
    cls.beta1 = staticmethod(cls.beta1)

    cls.beta0 = lambdify(methodArguments, 2 * pi * 1e6 / l0 * cls.ind)
    cls.beta0.__doc__ = "Calculate wavenumber (0th order dispersion) from the Sellmeier equation"
    cls.beta0 = staticmethod(cls.beta0)

    cls.beta3 = lambdify(methodArguments,
                         diff(groupInd.subs(l0, (2 * pi * c * 1e6) / symbols("w")), symbols("w"), 3).
                         subs(symbols("w"), (2 * pi * c * 1e6) / l0))
    cls.beta3.__doc__ = "Calculate 3rd order dispersion from the Sellmeier equation"
    cls.beta3 = staticmethod(cls.beta3)

    if deleteInd:
      del cls.ind

    return cls
  return nlmDecorator
