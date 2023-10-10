"""
Module for helper functions related to classical linear and nonlinear optics.
"""

import numpy as np


def calculateDispLength(beta2, timeScale, pulseTypeFWHM=None):
  """
  Return dispersion length (meters).
  beta2: group velocity dispersion (ps^2 / km or fs^2/mm)
  timeScale:  width of pulse (ps)
  pulseTypeFWHM: calculate time scale from FHWM for "sech" or "gauss" (Note: in power/field^2)
  """
  DS = 1000 * timeScale**2 / abs(beta2)
  if pulseTypeFWHM == "sech":
    DS /= 4 * np.log(1 + np.sqrt(2))**2
    # DS /= 4 * np.log(2 + np.sqrt(3))**2
  elif pulseTypeFWHM == "gauss":
    DS /= 4 * np.log(2)
    # DS /= 8 * np.log(2)
  return DS


def calculateChi2NlLength(d, peakPower, beamRadius, indexP, indexS, freqS):
  """
  Return nonlinear length (meters).
  d: nonlinear coefficient (pm / V)
  peakPower: pulse peak power (W)
  beamRadius: effective radius of the beam, to calculate intensity (m)
  indexP: refractive index at the pump frequency
  indexS: refractive index at the signal frequency
  freqS:  frequency of the signal (2 pi GHz)
  """
  c = 299792458 # m / s
  e0 = 1 / (4e-7 * np.pi * c**2) # F / m
  peakField = np.sqrt(2 * peakPower / (np.pi * beamRadius**2) / (indexP * e0 * c)) # V / m
  NL = 1 / ((2 * d * 1e-12 * freqS * 1e9 * peakField) / (indexS * c))
  return NL


def calculateChi3NlLength(n2, wavelength, peakPower, beamRadius):
  """
  Return nonlinear length (meters).
  n2: nonlinear index in (cm^2 W^-1)
  wavelength: in nm
  peakPower: pulse peak power (W)
  beamRadius: effective radius of the beam, to calculate intensity (m)
  """
  gamma = 2 * np.pi * n2 / (wavelength * np.pi * beamRadius**2)
  NL = 1e-5 / (peakPower * gamma)
  return NL


def calculateChi3NlLengthGamma(gamma, peakPower):
  """
  Return nonlinear length (meters).
  gamma: nonlinear coefficient (W^-1 km^-1); gamma = 2 pi n_2 / (lambda A_eff)
  peakPower: pulse peak power (W)
  """
  NL = 1000 / (peakPower * gamma)
  return NL


def findFrameOfReference(*ngs):
  """
  Calculate all the beta 1 (group wave number) values in a centered frame of reference, from the group indices n_g.
  Returns values in ps / km
  """
  c = 299792458 # m / s
  fom = 0.5 * (max(ngs) + min(ngs))
  return [(ng - fom) / c * 1e15 for ng in ngs]


def normalizeDispersion(timeScale, dispLength, beta0=[], beta1=[], beta2=[], beta3=[]):
  """
  Return normalized dispersion values, ie in units of dispersion length. Any dispersion length is valid.
  Assumes compatibility between units provided.
  """
  beta0N = [dispLength * b0 for b0 in beta0]                if isinstance(beta0, (list, tuple)) else dispLength * beta0
  beta1N = [dispLength * b1 / timeScale for b1 in beta1]    if isinstance(beta1, (list, tuple)) else dispLength * beta1 / timeScale
  beta2N = [dispLength * b2 / timeScale**2 for b2 in beta2] if isinstance(beta2, (list, tuple)) else dispLength * beta2 / timeScale**2
  beta3N = [dispLength * b3 / timeScale**3 for b3 in beta3] if isinstance(beta3, (list, tuple)) else dispLength * beta3 / timeScale**3
  return tuple(b for b in (beta0N, beta1N, beta2N, beta3N) if b)


def calcChirp(z):
  """
  Compute chirp coefficient C in exp(-0.5 C T^2). Variable z is in units of dispersion length.
  """
  return z / (1 + z**2)


def calcRayleighWidth(length, wavelength, index):
  """
  Calculate the Rayleigh width (ie radius, for a Gaussian beam) in some medium for
  a given length and wavelength.
  """
  return np.sqrt(length * wavelength / (np.pi * index))


def calcRayleighLength(width, wavelength, index):
  """
  Calculate the Rayleigh length (for a Gaussian beam) in some medium for a given
  width and wavelength.
  """
  return width**2 * np.pi * index / wavelength

