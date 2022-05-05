from .nlMaterial import *
try:
  from symengine import sqrt as ssqrt
except:
  from sympy import sqrt as ssqrt

info = """MgO Doped LN
Gayer et al 2008
d33 = 25.3
d31 = 4.85
d22 = 2.59
(Px)   (  0   0   0   0  d31 -d22) (Ex^2)
(Py) ~ (-d22 d22  0  d31  0    0 ) (Ey^2)
(Pz)   (-d31 d31 d33  0   0    0 ) (Ez^2)
                                  (2Ey Ez)
                                  (2Ez Ex)
                                  (2Ex Ey)"""

@nlMaterial
class MgOLNe:
  """
  MgO-doped Lithium Niobate extraordinary axis.
  """
  info = info

  a1 = 5.756
  a2 = 0.0983
  a3 = 0.2020
  a4 = 189.32
  a5 = 12.52
  a6 = 1.32E-02
  b1 = 2.860E-06
  b2 = 4.700E-08
  b3 = 6.113E-08
  b4 = 1.516E-04

  f = (T - 24.5) * (T + 570.82)
  ind = ssqrt(a1 + b1 * f + (a2 + b2 * f) / (l0**2 - (a3 + b3 * f)**2)
              + (a4 + b4 * f) / (l0**2 - a5**2) - a6 * l0**2)

  del a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, f


@nlMaterial
class MgOLNo:
  """
  MgO-doped Lithium Niobate ordinary axis.
  """
  info = info

  a1 = 5.653
  a2 = 0.1185
  a3 = 0.2091
  a4 = 89.61
  a5 = 10.85
  a6 = 1.97E-02
  b1 = 7.941E-07
  b2 = 3.134E-08
  b3 = -4.641E-09
  b4 = -2.188E-06

  f = (T - 24.5) * (T + 570.82)
  ind = ssqrt(a1 + b1*f + (a2 + b2 * f) / (l0**2 - (a3 + b3 * f)**2)
              + (a4 + b4 * f) / (l0**2 - a5**2) - a6 * l0**2)

  del a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, f

