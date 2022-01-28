"""
Module for generating poling patterns for chi(2) media.
"""

import numpy as np


def periodicPoling(deltaBeta0, L):
  polPeriod = 2 * np.pi / abs(deltaBeta0)
  nDomains  = 2 * L / polPeriod
  poling = np.ones(int(nDomains) + int(np.ceil(nDomains % 1)))
  poling[-1] = nDomains % 1
  poling *= L / np.sum(poling)
  return poling


def linearPoling(kMin, kMax, L, dL):
  """
  Create a poling design that has linearly varying phase matching, up to a given resolution.
  This is done by defining an instantaneous (spatial) frequency that varies linearly in z.
  The variables define the curve such that k(z) = a z + b, k(0) = kMin and k(L) = kMax.
  """
  z = np.linspace(0, L, int(round(L / dL)))
  polingDirection = np.sign(np.sin(0.5 * (kMax - kMin) * z**2 / L + kMin * z))
  polingDirection[polingDirection == 0.] = 1. # slight hack for the very unlikely case besides z=0

  p = np.concatenate([[0.], polingDirection, [0.]])
  polingProfile = np.diff(np.where(p[:-1] != p[1:]))
  return polingProfile.flatten() * dL


def detunePoling(kMin, kMax, k0, ka, L, dL):
  """
  Create a poling design that nonlinearly varies the phase matching, up to a given resolution, following an arctanh curve.
  This is done by defining an instantaneous (spatial) frequency that varies as the arctanh of z.
  The variables define the curve such that k(z) = k0 + ka arctanh(a (z - z0)), such that k(0) = kMin and k(L) = kMax.
  Useful for rapidly tuning in and/or detuning out of a phase matching frequency, or for apodization.
  """
  z = np.linspace(0, L, int(round(L / dL)))
  a = (np.tanh((kMax - k0) / ka) + np.tanh((k0 - kMin) / ka)) / L
  z0 = np.tanh((k0 - kMin) / ka) / a
  phase = ka / (2 * a) * np.log(1 - a**2 * (z - z0)**2) + ka * (z - z0) * np.arctanh(a * (z - z0)) + k0 * z

  polingDirection = np.sign(np.sin(phase - phase[0]))
  polingDirection[polingDirection == 0.] = 1.

  p = np.concatenate([[0.], polingDirection, [0.]])
  polingProfile = np.diff(np.where(p[:-1] != p[1:]))
  return polingProfile.flatten() * dL


def threeWaveMismatchRange(omega, domega, dbeta0, sign1, sign2,
                           beta1a=0, beta2a=0, beta3a=0,
                           beta1b=0, beta2b=0, beta3b=0,
                           beta1c=0, beta2c=0, beta3c=0):
  """
  Estimate the range of wavenumber mismatch of a three-wave mixing process over some bandwidth.
  """
  assert abs(sign1) == 1 and abs(sign2) == 1, "Sign must be +/-1"
  disp = lambda b1, b2, b3, w: b1 * w + 0.5 * b2 * w**2 + 1/6 * b3 * w**3
  mismatch = dbeta0 + disp(beta1a, beta2a, beta3a, omega) \
            + sign1 * disp(beta1b, beta2b, beta3b, omega) \
            + sign2 * disp(beta1c, beta2c, beta3c, omega)

  if isinstance(domega, (float, int)):
    maxdk = np.max(np.abs(mismatch[np.abs(omega) < domega]))
    mindk = np.min(np.abs(mismatch[np.abs(omega) < domega]))
  elif isinstance(domega, (tuple, list, np.ndarray)):
    maxdk = np.max(np.abs(mismatch[np.logical_and(-domega[0] < omega, omega < domega[1])]))
    mindk = np.min(np.abs(mismatch[np.logical_and(-domega[0] < omega, omega < domega[1])]))
  else:
    raise ValueError("unrecognized type for domega")

  return mindk, maxdk


def combinePoling(polingA, polingB, tol):
  """
  Combine poling structures by flipping the sign each time either structure flips (multiplying).
  polingA and polingB must contain the lengths of each domain.
  Note: to combine two structures you need to start with the sum and difference of spatial frequencies.
  """
  if abs(np.sum(polingA) - np.sum(polingB)) > tol:
   raise ValueError("Patterns A and B different lengths")

  combinedDomains = []
  indexA = -1
  indexB = -1
  remainingA = 0
  remainingB = 0

  while indexA < polingA.size and indexB < polingB.size:
    if abs(remainingA - remainingB) < tol:
      cumulative = 0
      while abs(remainingA - remainingB) < tol and (indexA < polingA.size-1 and indexB < polingB.size-1):
        cumulative += remainingA
        indexA += 1
        indexB += 1
        if polingA[indexA] > polingB[indexB]:
          remainingA = polingA[indexA] - polingB[indexB]
          cumulative += polingB[indexB]
          indexB += 1
          remainingB = polingB[indexB % polingB.size]
        else:
          remainingB = polingB[indexB] - polingA[indexA]
          cumulative += polingA[indexA]
          indexA += 1
          remainingA = polingA[indexA % polingA.size]
      if cumulative > 0: combinedDomains.append(cumulative)

    if remainingA > remainingB:
      if remainingB > 0: combinedDomains.append(remainingB)
      remainingA -= remainingB
      indexB += 1
      remainingB = polingB[indexB % polingB.size]
    else:
      if remainingA > 0: combinedDomains.append(remainingA)
      remainingB -= remainingA
      indexA += 1
      remainingA = polingA[indexA % polingA.size]

  if min(remainingA, remainingB) > 0: combinedDomains.append(min(remainingA, remainingB))

  if indexA < polingA.size-1:
    for i in range(indexA, polingA.size):
      combinedDomains.append(polingA[i])
  elif indexB < polingB.size-1:
    for i in range(indexB, polingB.size):
      combinedDomains.append(polingB[i])

  return np.array(combinedDomains)

