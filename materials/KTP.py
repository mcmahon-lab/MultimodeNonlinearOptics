from .nlMaterial import *
try:
  from symengine import sqrt as ssqrt, Piecewise
except:
  from sympy import sqrt as ssqrt, Piecewise

info = """KTP
Konig and Wong 2004
Fradkin et al 1999
Fan et al 1987
Emanueli and Arie 2003
d33 = 16.9
d32 = 4.4
d31 = 2.5
(Px)   ( 0   0   0   0  d31 0) (Ex^2)
(Py) ~ ( 0   0   0  d32  0  0) (Ey^2)
(Pz)   (d31 d32 d33  0   0  0) (Ez^2)
                               (2Ey Ez)
                               (2Ez Ex)
                               (2Ex Ey)"""

@nlMaterial()
class KTPz:
  """
  KTP z-axis
  """
  info = info

  a = 2.12725
  b = 1.18431
  c = 5.14852e-2
  d = 0.6603
  e = 100.00507
  f = 9.68956e-3
  nz = ssqrt(a + b / (1 - c / l0**2) + d / (1 - e / l0**2) - f * l0**2)

  a0 = 9.9587e-6
  a1 = 9.9228e-6
  a2 = -8.9603e-6
  a3 = 4.1010e-6

  b0 = 1.1882e-8
  b1 = 10.459e-8
  b2 = -9.8136e-8
  b3 = 3.1481e-8

  Dn = (a0 + a1 / l0 + a2 / l0**2 + a3 / l0**3) * (T - 25.) + (b0 + b1 / l0 + b2 / l0**2 + b3 / l0**3) * (T - 25.)**2

  ind = nz + Dn

  del a, b, c, d, e, f, nz, a0, a1, a2, a3, b0, b1, b2, b3, Dn


@nlMaterial()
class KTPy:
  """
  KTP y-axis
  """
  info = info

  # for ~1600 nm
  a = 2.09930
  b = 0.922683
  c = 0.0467695
  d = 0.0138408
  nyHI = ssqrt(a + b / (1 - c / l0**2) - d * l0**2)

  # for ~800
  a = 2.19229
  b = 0.83547
  c = 0.04970
  d = 0.01621
  nyLO = ssqrt(a + b / (1 - c / l0**2) - d * l0**2)

  a0 = 6.2897e-6
  a1 = 6.3061e-6
  a2 = -6.0629e-6
  a3 = 2.6486e-6

  b0 = -0.14445e-8
  b1 = 2.2244e-8
  b2 = -3.5770e-8
  b3 = 1.3470e-8

  Dn = (a0 + a1 / l0 + a2 / l0**2 + a3 / l0**3) * (T - 25.) + (b0 + b1 / l0 + b2 / l0**2 + b3 / l0**3) * (T - 25.)**2

  ind = Piecewise((nyHI, l0 > 1.2), (nyLO, l0 < 1.2)) + Dn

  del a, b, c, d, nyHI, nyLO, a0, a1, a2, a3, b0, b1, b2, b3, Dn

