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


def linearPoling(k0, kf, L, dL):
  """
  Create a poling design that has linearly varying phase matching, up to a given resolution dL.
  This is done by defining an instantaneous (spatial) frequency that varies linearly in z.
  The variables define the curve such that k(z) = a z + b, k(0) = k0 and k(L) = kf.
  """
  z = np.linspace(0, L, int(round(L / dL)))
  polingDirection = np.sign(np.sin(0.5 * (kf - k0) * z**2 / L + k0 * z))
  polingDirection[polingDirection == 0.] = 1. # slight hack for the very unlikely case besides z=0

  p = np.concatenate([[0.], polingDirection, [0.]])
  polingProfile = np.diff(np.where(p[:-1] != p[1:]))
  return polingProfile.ravel() * dL


def linearPolingContinuous(k0, kf, L):
  """
  Create a poling design that has linearly varying phase matching, with unlimited resolution.
  This is done by defining an instantaneous (spatial) frequency that varies linearly in z.
  The variables define the curve such that k(z) = a z + b, k(0) = k0 and k(L) = kf.
  """
  # Need to solve the equation: (kf-ki) / (2 L) z^2 + ki z = n pi for various values of n that we must find
  nFinal = int((kf + k0) * L / (2 * np.pi)) # n at the endpoint
  switches = False # whether n changes monotonically or reverts
  zSwitch = -k0 * L / (kf - k0) # point at which n changes direction
  if 0 < zSwitch < L:
    switches = True
    nSwitch = int(-k0**2 * L / (2 * np.pi * (kf - k0))) # value of n at switching (expression evaluated at zSwitch)
    degenerateSol = (nSwitch == -k0**2 * L / (2 * np.pi * (kf - k0))) # if spatial frequency->0 as expr->n pi TODO tolerance
    print(degenerateSol)
  else:
    nSwitch = None

  if switches:
    # find the point where direction of n switches, ie where n passes through zero again
    direction1 = np.sign(nSwitch) if np.sign(nSwitch) != 0 else 1
    direction2 = 1 if nFinal > nSwitch else -1
    additionalDoms = (2 if direction2 == np.sign(nFinal) else 1)
    nTimes2Pi = (2 * np.pi) * np.concatenate([np.arange(0, nSwitch, direction1),
                                              degenerateSol * [nSwitch], [0] * (nSwitch == 0),
                                              np.arange(nSwitch + direction2 * (nSwitch != 0),
                                                        nFinal + additionalDoms * direction2, direction2)])
    nSwitch = abs(nSwitch)
    if nSwitch == 0: nSwitch = 1
  else:
    direction = np.sign(nFinal) if np.sign(nFinal) != 0 else 1
    additionalDoms = (2 if direction == np.sign(nFinal) else 1)
    nTimes2Pi = (2 * np.pi) * np.arange(0, nFinal + additionalDoms * np.sign(nFinal), direction)

  # discriminant:
  pmFactor = np.sqrt((L * k0)**2 + nTimes2Pi * (L * (kf - k0)))
  # +/- depending on whether it's a positive or negative spatial frequency
  if k0 < 0:
    pmFactor[:nSwitch] *= -1
  elif switches:
    pmFactor[nSwitch:] *= -1
  if switches and degenerateSol: pmFactor[nSwitch] = 0
  # quadratic equation:
  z = (pmFactor - L * k0) / (kf - k0)
  z[-1] = L

  return np.diff(z)


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
  return polingProfile.ravel() * dL


def dutyCyclePmf(nlf, deltaBeta0, L, minSize, normalize=True):
  """
  Generate a custom phase matching function based on varying the duty-cycle of the periodic poling.
  The real-space nonlinearity function must be provided, the inverse Fourier transform of the PMF.
  The function is normalized to 1 unless otherwise normalize is set to False.
  Only supports real positive functions
  (ie function may not have a complex phase: this requires a phase shift in the periods).
  If the duty cycle generates domains smaller than minSize these are set to zero
  In this case those regions should be manually filled with the domain-deletion strategy deletedDomainPmf().
  """
  poling = periodicPoling(deltaBeta0, L)
  nDomainPairs = poling.size // 2
  halfPeriod = poling[0]

  if minSize > poling[0]: raise ValueError("minSize larger than half period.")
  minDuty = 0.5 * minSize / poling[0]
  hasSingleDomain = poling.size % 2

  relativeNL = nlf(np.linspace(halfPeriod, L - poling[-1] - halfPeriod * hasSingleDomain, nDomainPairs))
  if normalize:
    relativeNL *= 1 / np.max(np.abs(relativeNL))
  elif np.any(np.abs(relativeNL) > 1):
    raise ValueError("Function has value larger than 1")

  dutyCycle = np.arcsin(relativeNL) / np.pi

  dutyCycle[np.abs(dutyCycle) < minDuty] = 0
  dutyCycle[np.abs(1 - dutyCycle) < minDuty] = 1
  dutyCycle[dutyCycle < 0] += 1 # Note negative values affects duty cycle but not effective nonlinearity

  # Split poling into pairs of domains and vary their width according to the PMF
  dcPoling = np.empty_like(poling)
  dcPoling[0:2*nDomainPairs:2] = poling[0:2*nDomainPairs:2] * (2 * dutyCycle)
  dcPoling[1:2*nDomainPairs:2] = poling[1:2*nDomainPairs:2] * (2 * (1 - dutyCycle))

  # Fix the last domain
  if hasSingleDomain:
    dutyCycleEnd = np.arcsin(nlf(L)) / np.pi
    if dutyCycleEnd >= minDuty and poling[-1] > dutyCycleEnd * poling[0]:
      dcPoling[-1] = dutyCycleEnd * poling[0]
      dcPoling = np.concatenate([dcPoling, [poling[-1] - dcPoling[-1]]])
      # Variables not used later, but note nDomainPairs += 1, hasSingleDomain = False
    dcPoling[-1] = poling[-1]
  else:
    dcPoling[-1] = halfPeriod + poling[-1] - dcPoling[-2]
    if dcPoling[-1] < 0: # in case using a duty cycle >0.5
      dcPoling[-2] += dcPoling[-1]
      dcPoling = dcPoling[:-1]
      # Variables not used later, but note nDomainPairs -= 1, hasSingleDomain = True

  # Fix the phase for the varying duty cycle so that the (odd numbered) domains are always centered
  nonZeroInds = np.argwhere(dutyCycle)
  initialOffset = halfPeriod * (0.5 - dutyCycle[nonZeroInds[0]])
  for [i] in nonZeroInds[1:]:
    offsetDiff = halfPeriod * (0.5 - dutyCycle[i]) - initialOffset
    dcPoling[2*i-1] += offsetDiff
    dcPoling[2*i+1] -= offsetDiff
  # check last domain again
  while dcPoling[-1] < 0:
    dcPoling[-2] += dcPoling[-1]
    dcPoling = dcPoling[:-1]

  # remove empty domains and combine adjacent ones
  iAccum = 0
  accum = False
  for i in range(dcPoling.size-1):
    if dcPoling[i] == 0:
      accum = True
    elif accum:
      dcPoling[iAccum] += dcPoling[i]
      dcPoling[i] = 0
      accum = False
    else:
      iAccum = i

  # Don't accidentally flip the last domain when duty cycle is 0
  if dutyCycle[-1] == 0:
    dcPoling[iAccum] += dcPoling[-1]
    dcPoling[-1] = 0

  dcPoling = dcPoling[dcPoling > 0].copy()

  return dcPoling


def deletedDomainPmf(nlf, deltaBeta0, L, dutyCycle=0.5, normalize=True, override=False):
  """
  Generate a custom phase matching function based on domain-deletion in the periodic poling.
  The real-space nonlinearity function must be provided, the inverse Fourier transform of the PMF.
  The function is normalized to 1 unless normalize is set to False.
  Only supports real positive functions
  (ie function may not have a complex phase: this requires a phase shift in the periods).
  The duty cycle may be specified to provide a smaller nonlinearity unit, allowing for a higher density,
  or smoother discretization of the PMF function.
  However, both are not possible simultaneously, since the largest value of the function cannot exceed
  the effective nonlinearity due to a change in duty cycle.
  (This can be overridden with override for specific cases, but the function is not guaranteed to be optimal.)
  """
  if 0 >= dutyCycle or dutyCycle >= 1:
    raise ValueError("Duty cycle not in the open interval (0, 1)")

  if dutyCycle != 0.5 and normalize:
    raise ValueError("Changing the duty cycle and normalizing are incompatible")

  poling = periodicPoling(deltaBeta0, L)
  halfPeriod = poling[0]
  hasSingleDomain = bool(poling.size % 2)
  nDomainPairs = poling.size // 2

  if dutyCycle != 0.5:
    if not hasSingleDomain: lastDomainLength = poling[-1]
    poling[0:2*nDomainPairs:2] *= 2 * dutyCycle
    poling[1:2*nDomainPairs:2] *= 2 * (1 - dutyCycle)
    if not hasSingleDomain: # calculate the remaining length for the last domain
      poling[-1] = lastDomainLength + halfPeriod - poling[-2]
      if poling[-1] < 0: # in case using a duty cycle >0.5
        poling[-2] += poling[-1]
        poling = poling[:-1]
        hasSingleDomain = True
        nDomainPairs -= 1
    # check if the duty cycle reduces the last domain enough to fit another domain
    elif 2 * dutyCycle * halfPeriod < poling[-1]: # and hasSingleDomain
      # Note poling[0] = 2 * dutyCycle * halfPeriod
      poling = np.concatenate([poling[:-1], [poling[0], poling[-1] - poling[0]]])
      hasSingleDomain = False
      nDomainPairs += 1

  relativeNL = nlf(np.linspace(halfPeriod, L - poling[-1] - halfPeriod * (poling.size % 2),
                               nDomainPairs + hasSingleDomain))
  if normalize:
    relativeNL *= 1 / np.max(np.abs(relativeNL))
  elif np.any(np.abs(relativeNL) > 1):
    raise ValueError("Function has value larger than 1")

  # account for the effect of duty cycle on NL if applicable
  nlUnit = np.sin(np.pi * dutyCycle)

  # if the effective nonlinearity is reduced via duty cycle must
  # make sure the function does not change faster than the nonlinearity
  if dutyCycle != 0.5:
    if np.any(nlUnit < relativeNL) and not override:
      raise ValueError("Function takes on values larger than the effective nonlinearity by duty cycle "
                       f"({nlUnit} vs {relativeNL.max()}). "
                       "Combining domain deletion and duty cycle modulation is only allowed when the "
                       "function takes values less than the duty cycle allows. It is suggested that "
                       "the crystal be split up into regions with different design strategies.")

  # We need to (approximately) match the integral of the function with discrete impulses
  integral = np.cumsum(relativeNL)
  integral *= integral[-1] / (nlUnit * np.round(integral[-1] / nlUnit)) # precompute units needed and redistribute weight
  discreteCumSum = 0
  nlLocations = np.zeros(relativeNL.size, dtype=np.bool)
  for i in range(integral.size - hasSingleDomain):
    if abs(discreteCumSum + nlUnit - integral[i]) < abs(discreteCumSum - integral[i]):
      discreteCumSum += nlUnit
      nlLocations[i] = True
  if hasSingleDomain:
    nlUnit = np.sin(0.5 * np.pi * poling[-1] / halfPeriod)
    if abs(discreteCumSum + nlUnit - integral[i]) < abs(discreteCumSum - integral[i]):
      discreteCumSum += nlUnit
      nlLocations[-1] = True

  # Make domains based on where we need to flip the nonlinearity
  locationIndices = np.flatnonzero(nlLocations)
  domainLength = halfPeriod * (2 * dutyCycle)
  startsOn = (locationIndices[0] == 0)
  endsOn = (locationIndices[-1] == nDomainPairs+hasSingleDomain-1)

  newPoling = np.zeros(2 * locationIndices.size
                       + (not startsOn)
                       - (endsOn and hasSingleDomain)
                       )

  if startsOn:
    newPoling[0::2] = domainLength
    for i in range(1, locationIndices.size):
      newPoling[2*i-1] = np.sum(poling[2*locationIndices[i-1]+1 : 2*locationIndices[i]])
    if not endsOn: # Extend the last domain to the end of the crystal
      newPoling[2*i+1] = np.sum(poling[2*locationIndices[-1]+1 :])
    else: # Make the last domain the remaining length of the crystal
      newPoling[-1] = poling[-1]

  else:
    newPoling[1::2] = domainLength
    newPoling[0] = np.sum(poling[0 : 2*locationIndices[0]])
    for i in range(1, locationIndices.size):
      newPoling[2*i] = np.sum(poling[2*locationIndices[i-1]+1 : 2*locationIndices[i]])
    if not endsOn: # Extend the last domain to the end of the crystal
      newPoling[2*i+2] = np.sum(poling[2*locationIndices[-1]+1 :])
    else: # Make the last domain the remaining length of the crystal
      newPoling[-1] = poling[-1]

  return newPoling


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


def combinePoling(polingA, polingB, minLength, tol):
  """
  Combine poling structures by flipping the sign each time either structure flips (multiplying).
  polingA and polingB must contain the lengths of each domain.
  Note: to combine two structures you need to start with the sum and difference of spatial frequencies.
  minLength is the minimum domain length. tol is the allowable error tolerance in the domain lengths.
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

    if remainingA > remainingB and remainingB > minLength:
      combinedDomains.append(remainingB)
      remainingA -= remainingB
      indexB += 1
      remainingB = polingB[indexB % polingB.size]
    elif remainingB > remainingA and remainingA > minLength:
      combinedDomains.append(remainingA)
      remainingB -= remainingA
      indexA += 1
      remainingA = polingA[indexA % polingA.size]
    else: # remainingA, remainingB < minLength
      indexA += 1
      indexB += 1
      combinedDomains.append(remainingA)
      remainingA += polingA[indexA % polingA.size]
      remainingB += polingB[indexB % polingB.size]

  if min(remainingA, remainingB) > 0: combinedDomains.append(min(remainingA, remainingB))

  if indexA < polingA.size-1:
    for i in range(indexA, polingA.size):
      combinedDomains.append(polingA[i])
  elif indexB < polingB.size-1:
    for i in range(indexB, polingB.size):
      combinedDomains.append(polingB[i])

  return np.array(combinedDomains)


def domainsToSpace(poling, nZSteps):
  """
  Convert an array of domain lengths to an array of orientations (+/-1) in discretized space.
  For visualizing or generating spatial Fourier transforms.
  """
  poling = np.array(poling)
  if np.any(poling <= 0):
    raise ValueError("Poling contains non-positive length domains")

  poleDomains = np.cumsum(poling, dtype=np.float_)
  poleDomains *= nZSteps / poleDomains[-1]

  _poling = np.empty(nZSteps)
  prevInd = 0
  direction = 1

  for i in range(poleDomains.size):
    currInd = poleDomains[i]
    currIndRound = int(currInd)

    if currInd < prevInd:
      raise ValueError("Poling period too small for given resolution")

    _poling[prevInd:currIndRound] = direction

    if (currIndRound < nZSteps): # interpolate indices corresponding to steps on the boundary of two domains
      _poling[currIndRound] = direction * (2 * abs(np.fmod(currInd, 1)) - 1)

    direction *= -1
    prevInd = currIndRound + 1

  return _poling

