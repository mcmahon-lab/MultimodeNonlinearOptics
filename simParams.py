"""
Module contains a function for generating the input conditions to a `nonlinearmedium`
simulation (given a material, optical wavelengths, pump properties, etc), potentially
simplifying the process.
"""

import numpy as np
from classical import findFrameOfReference, calculateDispLength, calculateChi2NlLength, \
                      calculateChi3NlLength, calcRayleighLength, normalizeDispersion


def generateParameters(wavelengths, timeScale, materials, nonlinearities,
                       length, timeBins, tMax, peakPower, temperature,
                       modeRadius, mismatchFuncs):
  """
  wavelengths:     All the central wavelengths given in nanometers. The pump wavelength
                   must be first
  timeScale:       The characteristic time of the simulation in picoseconds
  materials:       One or multiple material objects which inherits from `nLMaterial`,
                   to provide dispersion information
  nonlinearities:  Indicate what nonlinearites to calculate and their strength
                   Indicate chi(2) or chi(3), the value of d or n_2, and the wavelength involved
                   The data structure should be (bool isChi2, float d_or_n2, int wavelengthIndex)
  length:          The length of propagation in meters
  timeBins:        Number of bins/points to include on the time axis in the simulation
  tMax:            Half width of the simulation time axis in units of `timeScale`s
  peakPower:       The peak power of the pump pulse in Watts
  temperature:     The temperature of the material in celsius (passed to material object)
  modeRadius:      Radius of the pump mode, in meters, to calculate the intensity and
                   Rayleigh length if applicable
  mismatchFuncs:   Function(s) indicating how to add wavenumbers
                   Eg for SFG: mismatchFunc = lambda k1, k2, k3: k1 + k2 - k3
                   Order must match that of `wavelengths`

  Returns: tuple of normalized simulation input parameters and dimensionful
  time, frequency and wavelength axes. The data structure is as follows:
   (
    (dispersion length, nonlinear lengths [list], propagation length, Rayleigh length), # normalized
    (delta beta_0, beta_1, beta_2, beta_3),     # all lists, normalized dispersion parameters
    (time, angFreq, wavelengthAxes, angFreqMax) # dimensionful, in ps, 2 pi THz, nm
   )
  Note: by default everything is normalized to the dispersion length (ie to beta2 of the pump),
  dispersion length = 1, unless it is infinite.
  To normalize with respect to another length, divide all lengths and multiply all dispersion
  parameters by one of the other normalized lengths.
  """
  c = 299792458 # m / s

  try:
    nMaterials = len(materials)
  except TypeError:
    nMaterials = 1

  try:
    nWavelengths = len(wavelengths)
  except TypeError:
    nWavelengths = 1
    wavelengths = (wavelengths,)

  if nMaterials != 1 and nWavelengths != nMaterials:
    raise ValueError("Must have as many materials as wavelenghths, if more than one material.")

  try:
    len(nonlinearities[0])
    nNonlinearities = len(nonlinearities)
  except TypeError:
    nNonlinearities = 1
    nonlinearities = (nonlinearities,)

  try:
    nMismatchFuncs = len(mismatchFuncs)
  except TypeError:
    if mismatchFuncs is not None:
      nMismatchFuncs = 1
      mismatchFuncs = (mismatchFuncs,)
    else:
      nMismatchFuncs = 0
      mismatchFuncs = ()

  freqs = [2 * np.pi * c / wav for wav in wavelengths] # 2pi GHz

  # Material index and dispersion values
  index = [0] * nWavelengths
  group = [0] * nWavelengths
  beta2 = [0] * nWavelengths
  beta3 = [0] * nWavelengths
  for i, wavelength in enumerate(wavelengths):
    mat = (materials[i] if nMaterials > 1 else materials)
    index[i] = mat.n(wavelength*1e-3, temperature)
    group[i] = mat.ng(wavelength*1e-3, temperature)
    beta2[i] = 1e27 * mat.gvd(wavelength*1e-3, temperature)
    beta3[i] = 1e39 * mat.beta3(wavelength*1e-3, temperature)

  # Walk-off
  beta1 = findFrameOfReference(*group) # ps / km

  # Phase velocity mismatch
  diffBeta0 = [0] * nMismatchFuncs
  for i, mismatchFunc in enumerate(mismatchFuncs):
    diffBeta0[i] = 2 * np.pi * 1e12 * mismatchFunc(*[ind / wavelength for ind, wavelength in zip(index, wavelengths)]) # km^-1

  # Characteristic lengths, m
  dispLength = calculateDispLength(beta2[0], timeScale, pulseTypeFWHM=None)

  NL = [0] * nNonlinearities
  for i, (isChi2, nlValue, wavInd) in enumerate(nonlinearities):
    if isChi2:
      # d = nlValue
      NL[i] = calculateChi2NlLength(nlValue, peakPower, modeRadius, index[0], index[wavInd], freqs[wavInd])
    else:
      # n2 = nlValue
      NL[i] = calculateChi3NlLength(nlValue, wavelengths[wavInd], peakPower, modeRadius)

  rayleighLength = calcRayleighLength(modeRadius, wavelengths[0] * 1e-9, index[0])

  # Normalized quantities
  normalizerLength = dispLength if dispLength != float("inf") else \
                     _findNormalization(timeScale, NL, group, beta2, beta3, length)

  relDispLen  = dispLength / normalizerLength
  relLength   = length / normalizerLength
  relNlLength = [nl / normalizerLength for nl in NL]
  rayleighLengthN = rayleighLength / normalizerLength

  diffBeta0N, beta1N, beta2N, beta3N = \
    normalizeDispersion(timeScale, normalizerLength / 1000, diffBeta0, beta1, beta2, beta3)

  # Normalized axes
  tau   = (2 * tMax / nt) * ifftshift(np.arange(-nt / 2, nt / 2))
  omega = (-np.pi / tMax) *  fftshift(np.arange(-nt / 2, nt / 2))
  wMax = np.max(omega)
  # Dimensionful axes
  time    = tau * timeScale   # ps
  angFreq = omega / timeScale # 2 pi THz
  wavelengthAxes = [2 * np.pi * c / (1000 * angFreq + freq) for freq in freqs] # nm
  angFreqMax = wMax / timeScale


  return (relDispLen, relNlLength, relLength, rayleighLengthN), \
         (diffBeta0N, beta1N, beta2N, beta3N), \
         (time, angFreq, wavelengthAxes, angFreqMax)


def _findNormalization(timeScale, NL, group, beta2, beta3, length):
  inf = normalizerLength = float("inf")

  i = 0
  while normalizerLength == inf:
    normalizerLength = NL[i]
    i += 1
    if i == len(NL): break

  i = 0
  while normalizerLength == inf:
    normalizerLength = 1000 * timeScale / abs(group[i])
    i += 1
    if i == len(group): break

  i = 0
  while normalizerLength == inf:
    normalizerLength = 1000 * timeScale**2 / abs(beta2[i])
    i += 1
    if i == len(beta2): break

  i = 0
  while normalizerLength == inf:
    normalizerLength = 1000 * timeScale**3 / abs(beta3[i])
    i += 1
    if i == len(beta3): break

  if normalizerLength == inf:
    normalizerLength = length

  return normalizerLength
