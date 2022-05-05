from .nlMaterial import *
try:
  from symengine import sqrt as ssqrt, sin as ssin, cos as scos
except:
  from sympy import sqrt as ssqrt, sin as ssin, cos as scos

info = """LBO
Kato 1994
d31 =  1.05 pm/V
d32 = -0.98 pm/V
d33 =  0.05 pm/V
XY plane: dooe = d32 cosφ
YZ plane: doeo = deoo = d31 cosθ
"""

@nlMaterial(deleteInd=False)
class LBOz:
  """
  Lithium triborate z axis.
  """
  info = info

  a  = 2.5865
  b1 = 0.01310
  b2 = 0.01223
  c1 = -0.01862
  c2 = 4.5778e-5
  c3 = -3.2526e-5
  
  d1 = 1.5
  d2 = -9.7
  d3 = -74.49e-4

  f = (d1 * l0 + d2) * 1e-6 * ((T - 20.) + d3 * (T - 20.)**2)

  ind = ssqrt(a + b1 / (l0**2 - b2)
              + c1 * l0**2
              + c2 * l0**4
              + c3 * l0**6) + f

  del a, b1, b2, c1, c2, c3, d1, d2, d3, f


@nlMaterial(deleteInd=False)
class LBOy:
  """
  Lithium triborate y axis.
  """
  info = info

  a  = 2.5390
  b1 = 0.01277
  b2 = 0.01189
  c1 = -0.01849
  c2 = 4.3025e-5
  c3 = -2.9131e-5

  d1 = 6.01
  d2 = -19.4
  d3 = -32.89e-4

  f = (d1 * l0 + d2) * 1e-6 * ((T - 20.) + d3 * (T - 20.)**2)

  ind = ssqrt(a + b1 / (l0**2 - b2)
              + c1 * l0**2
              + c2 * l0**4
              + c3 * l0**6) + f

  del a, b1, b2, c1, c2, c3, d1, d2, d3, f


@nlMaterial(deleteInd=False)
class LBOx:
  """
  Lithium triborate x axis.
  """
  info = info

  a  = 2.4542
  b1 = 0.01125
  b2 = 0.01135
  c1 = -0.01388

  d1 = -3.76
  d2 = 2.30
  d3 = 29.13e-3

  f = (d1 * l0 + d2) * 1e-6 * ((T - 20.) + d3 * (T - 20.)**2)

  ind = ssqrt(a + b1 / (l0**2 - b2)
              + c1 * l0**2) + f

  del a, b1, b2, c1, d1, d2, d3, f


@nlMaterial(angleTuning=True)
class LBOxy:
  """
  Angle-tuned extraordinary axis for lithium triborate on the xy-plane
  Generally Type I processes. φ=0 -> y, φ=90 -> x
  """
  ind = 1 / ssqrt(scos(th)**2 / LBOy.ind**2 + ssin(th)**2 / LBOx.ind**2)


@nlMaterial(angleTuning=True)
class LBOyz:
  """
  Angle-tuned extraordinary axis for lithium triborate on the yz-plane
  Generally Type II processes. θ=0 -> y, θ=90 -> z
  """
  ind = 1 / ssqrt(scos(th)**2 / LBOy.ind**2 + ssin(th)**2 / LBOz.ind**2)


del LBOz.ind, LBOy.ind, LBOx.ind

