from .nlMaterial import *
try:
  from symengine import sqrt as ssqrt
except:
  from sympy import sqrt as ssqrt

info = """Fused Silica
Malitson et al 1965
n2 = 2.2e-16 cm^2/W"""

@nlMaterial(temperatureTuning=False)
class Silica:
  """
  Fused Silica
  """
  info = info

  a1 = 0.6961663
  a2 = 0.4079426
  a3 = 0.8974794
  b1 = 0.0684043
  b2 = 0.1162414
  b3 = 9.896161

  ind = ssqrt(1 + (a1 * l0**2) / (l0**2 - b1**2)
                + (a2 * l0**2) / (l0**2 - b2**2)
                + (a3 * l0**2) / (l0**2 - b3**2)
                )

  del a1, a2, a3, b1, b2, b3
