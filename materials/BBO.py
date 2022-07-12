from .nlMaterial import *
try:
  from symengine import sqrt as ssqrt, sin as ssin, cos as scos
except:
  from sympy import sqrt as ssqrt, sin as ssin, cos as scos

info = """BBO
Tamosauskas et al 2018
Eimeri et al 1987
d22 = ± 2.2
d15 = d31 = ± 0.08
dooe = d31 sinθ – d22 cosθ sin3φ
deoe = doee = d22 cos2θ cos3φ
"""

@nlMaterial(deleteInd=False)
class BBOe:
  """
  Barium Borate extraordinary axis.
  """
  info = info

  a1 = 1.151075
  a2 = 0.21803
  a3 = 0.656
  b1 = 0.007142
  b2 = 0.02259
  b3 = 263

  f = -9.3e-6 * (T - 22.5)

  ind = ssqrt(1 + (a1 * l0**2) / (l0**2 - b1)
                + (a2 * l0**2) / (l0**2 - b2)
                + (a3 * l0**2) / (l0**2 - b3)) + f

  del a1, a2, a3, b1, b2, b3, f


@nlMaterial(deleteInd=False)
class BBOo:
  """
  Barium Borate ordinary axis.
  """
  info = info

  a1 = 0.90291
  a2 = 0.83155
  a3 = 0.76536
  b1 = 0.003926
  b2 = 0.018786
  b3 = 60.01

  f = -16.6e-6 * (T - 22.5)

  ind = ssqrt(1 + (a1 * l0**2) / (l0**2 - b1)
                + (a2 * l0**2) / (l0**2 - b2)
                + (a3 * l0**2) / (l0**2 - b3)) + f

  del a1, a2, a3, b1, b2, b3, f


@nlMaterial(angleTuning=True)
class BBO:
  """
  Angle-tuned Barium Borate
  """
  ind = 1 / ssqrt(scos(th)**2 / BBOo.ind**2 + ssin(th)**2 / BBOe.ind**2)


del BBOo.ind, BBOe.ind

