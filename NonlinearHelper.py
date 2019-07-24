import numpy as np

def calculateLengthScales(gamma, peakPower, beta2, timeScale, pulseTypeFWHM=None, refractiveInd=1):
  """
  Return dispersion length and nonlinear lengths (meters).
  gamma: nonlinear coefficient (W^-1 km^-1)
  peakPower: pulse peak power (W)
  beta2: group velocity dispersion (ps^2 / km)
  fwhm:  width of pulse (ps)
  pulseTypeFWHM: calculate time scale from FHWM for "sech" or "gauss" (Note: in power/field^2)
  refractiveInd: increase optical path length based on index of refraction
  """
  NL = 1000 / (peakPower * gamma)
  DS = 1000 * timeScale**2 / abs(beta2)
  if pulseTypeFWHM == "sech":
    DS /= 4 * np.log(1 + np.sqrt(2))**2
    # DS /= 4 * np.log(2 + np.sqrt(3))**2
  elif pulseTypeFWHM == "gauss":
    DS /= 4 * np.log(2)
    # DS /= 8 * np.log(2)
  return NL * refractiveInd, DS * refractiveInd


if __name__ == "__main__":
  print(calculateLengthScales(2, 2000, -20, 0.125, "sech", 1.55))