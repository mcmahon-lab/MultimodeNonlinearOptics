import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift, ifft2
from scipy.linalg import det, sqrtm, inv, eig, norm, dft


def calculateLengthScales(gamma, peakPower, beta2, timeScale, pulseTypeFWHM=None):
  """
  Return dispersion length and nonlinear lengths (meters).
  gamma: nonlinear coefficient (W^-1 km^-1)
  peakPower: pulse peak power (W)
  beta2: group velocity dispersion (ps^2 / km)
  fwhm:  width of pulse (ps)
  pulseTypeFWHM: calculate time scale from FHWM for "sech" or "gauss" (Note: in power/field^2)
  """
  NL = 1000 / (peakPower * gamma)
  DS = 1000 * timeScale**2 / abs(beta2)
  if pulseTypeFWHM == "sech":
    DS /= 4 * np.log(1 + np.sqrt(2))**2
    # DS /= 4 * np.log(2 + np.sqrt(3))**2
  elif pulseTypeFWHM == "gauss":
    DS /= 4 * np.log(2)
    # DS /= 8 * np.log(2)
  return NL, DS


def calcQuadratureGreens(greenC, greenS):
  Z = np.block([[np.real(greenC + greenS), -np.imag(greenC - greenS)],
                [np.imag(greenC + greenS),  np.real(greenC - greenS)]]).astype(dtype=np.float_)

  return Z


def calcCovarianceMtx(Z, tol=1e-4):
  cov = Z @ Z.T
  determinant = det(cov)
  assert abs(determinant - 1) < tol, "det(C) = %f" % determinant
  return cov


def calcLOSqueezing(C, pumpTimeProf, tol=1e-4):
  freqDomain = fftshift(fft(pumpTimeProf))

  localOscillX = np.hstack([freqDomain.real,  freqDomain.imag]) / np.linalg.norm(freqDomain)
  localOscillP = np.hstack([freqDomain.imag, -freqDomain.real]) / np.linalg.norm(freqDomain)

  covMatrix = np.zeros((2, 2))
  covMatrix[0,0] = localOscillX.T @ C @ localOscillX
  covMatrix[1,1] = localOscillP.T @ C @ localOscillP
  covMatrix[0,1] = covMatrix[1,0] = (localOscillX.T @ C @ localOscillP + localOscillP.T @ C @ localOscillX) / 2

  variances = np.linalg.eigvals(covMatrix)

  assert abs(covMatrix[0,0] - 1) < tol, "C_xx = %f" % covMatrix[0,0]

  variances[0], variances[1] = np.min(variances), np.max(variances)
  return variances


# simpler version than Xanadu
def blochMessiahEigs(Z):
  sigma = sqrtm(Z @ Z.T)
  eigenvalues, eigenvectors = eig(sigma)

  sortedEig = np.sort(eigenvalues).real
  return sortedEig

def blochMessiahVecs(Z):
  sigma = sqrtm(Z @ Z.T)
  eigenvalues, eigenvectors = eig(sigma)

  indices = np.argsort(eigenvalues, )

  sortedEig = eigenvalues[indices].real
  sortedVec = eigenvectors[indices]

  u = inv(sigma) @ Z
  eigenvectors_ = eigenvectors.T @ u

  sortedVec_ = eigenvectors_[indices]

  return sortedEig, sortedVec, sortedVec_


def convertGreenFreqToTime(greenC, greenS):
  # TODO might need some transposition steps doesn't seem 100% correct
  nFreqs = greenC.shape[0]
  dftMat = np.conj(dft(nFreqs))
  gCtime = ifftshift(ifft2(fftshift(greenC)) * nFreqs)
  gStime = ifftshift(dftMat.T @ fftshift(greenS) @ dftMat / nFreqs)
  return gCtime, gStime


def calcChirp(z):
  """
  Compute chirp coefficient C in exp(-0.5 C T^2). Variable z is in units of dispersion length.
  """
  return (0.5 * z) / (1 + 0.25 * z**2)

if __name__ == "__main__":
  print(calculateLengthScales(2, 2000, -20, 0.125, "sech"))