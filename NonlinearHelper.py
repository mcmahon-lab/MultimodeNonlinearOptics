import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift, ifft2
from scipy.linalg import det, sqrtm, inv, eig, eigvals, norm, dft


def calculateLengthScales(gamma, peakPower, beta2, timeScale, pulseTypeFWHM=None):
  """
  Return dispersion length and nonlinear lengths (meters).
  gamma: nonlinear coefficient (W^-1 km^-1)
  peakPower: pulse peak power (W)
  beta2: group velocity dispersion (ps^2 / km)
  timeScale:  width of pulse (ps)
  pulseTypeFWHM: calculate time scale from FHWM for "sech" or "gauss" (Note: in power/field^2)
  """
  # TODO DEPRECATED
  NL = 1000 / (peakPower * gamma)
  DS = 1000 * timeScale**2 / abs(beta2)
  if pulseTypeFWHM == "sech":
    DS /= 4 * np.log(1 + np.sqrt(2))**2
    # DS /= 4 * np.log(2 + np.sqrt(3))**2
  elif pulseTypeFWHM == "gauss":
    DS /= 4 * np.log(2)
    # DS /= 8 * np.log(2)
  return NL, DS


def calculateDispLength(beta2, timeScale, pulseTypeFWHM=None):
  """
  Return dispersion length (meters).
  beta2: group velocity dispersion (ps^2 / km)
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


def calculateChi3NlLength(gamma, peakPower):
  """
  Return nonlinear length (meters).
  gamma: nonlinear coefficient (W^-1 km^-1)
  peakPower: pulse peak power (W)
  """
  NL = 1000 / (peakPower * gamma)
  return NL


def calcQuadratureGreens(greenC, greenS):
  """
  Convert the Green's (transmission) matrix to the x and p quadrature basis from the a basis
  greenC and greenS are such that a_out = C a + S a^†:
  """
  Z = np.block([[np.real(greenC + greenS), -np.imag(greenC - greenS)],
                [np.imag(greenC + greenS),  np.real(greenC - greenS)]]).astype(dtype=np.float_)

  return Z


def calcCovarianceMtx(Z, tol=1e-4):
  """
  Calculate the covariance matrix in x/p quadrature basis from the transmission Green's matrix.
  Checks that the determinant of the covariance matrix is approximately unity.
  """
  cov = Z @ Z.T
  determinant = det(cov)
  assert abs(determinant - 1) < tol, "det(C) = %f" % determinant
  return cov


def normalizedCov(Cov):
  """
  Return a "normalized" covariance matrix to highlight differences from the identity.
  """
  diagC = np.diag(Cov)
  normC = np.tile(diagC, (diagC.shape[0], 1))
  return (Cov - np.eye(Cov.shape[0])) / np.sqrt(normC * normC.T)


def calcLOSqueezing(C, pumpProf, tol=1e-4, inTimeDomain=True):
  """
  Compute the squeezing observed combining the covariance matrix and a local oscillator with a given profile.
  """
  if inTimeDomain:
    freqDomain = fftshift(fft(pumpProf))
  else:
    freqDomain = fftshift(pumpProf)

  localOscillX = np.hstack([freqDomain.real,  freqDomain.imag]) / np.linalg.norm(freqDomain)
  localOscillP = np.hstack([freqDomain.imag, -freqDomain.real]) / np.linalg.norm(freqDomain)

  covMatrix = np.zeros((2, 2))
  covMatrix[0,0] = localOscillX.T @ C @ localOscillX
  covMatrix[1,1] = localOscillP.T @ C @ localOscillP
  covMatrix[0,1] = covMatrix[1,0] = ((localOscillX + localOscillP).T @ C @ (localOscillX + localOscillP)
                                     - covMatrix[0,0] - covMatrix[1,1]) / 2
  # more efficient version of (localOscillX.T @ C @ localOscillP + localOscillP.T @ C @ localOscillX) / 2

  variances = np.linalg.eigvals(covMatrix)

  assert abs(covMatrix[0,0] - 1) < tol, "C_xx = %f" % covMatrix[0,0]

  variances[0], variances[1] = np.min(variances), np.max(variances)
  return variances


def downSampledCov(Z, perBin):
  """
  Downsample a covariance matrix by grouping modes together.
  """
  N = Z.shape[0]
  assert N // 2 % perBin == 0

  newZ = np.zeros((N // perBin, N))

  for i in range(N // perBin):
    newZ[:, i] = np.sum(Z[:, i].flatten().reshape(-1, perBin), axis=1) / np.sqrt(perBin)

  return newZ @ newZ.T


def obtainFrequencySqueezing(C, bandSize=1):
  """
  Calculate squeezing as a function of frequency (single frequency LO).
  """

  nFreqs = C.shape[0] // 2
  covMatrix = np.zeros((2, 2))

  squeezing = np.zeros(nFreqs)
  antisqueezing = np.zeros(nFreqs)

  for i in range(nFreqs):
    covMatrix[0,0] = C[i, i]
    covMatrix[0,1] = C[i, nFreqs + i]
    covMatrix[1,0] = C[nFreqs + i, i]
    covMatrix[1,1] = C[nFreqs + i, nFreqs + i]

    variances = np.linalg.eigvals(covMatrix)
    squeezing[i], antisqueezing[i] = np.min(variances), np.max(variances)

  return squeezing, antisqueezing


def blochMessiahEigs(Z):
  """
  Obtain the Bloch-Messiah principal values (supermode uncertainties).
  """
  sigma = sqrtm(Z @ Z.T)
  eigenvalues = eigvals(sigma)

  sortedEig = np.sort(eigenvalues).real
  return sortedEig

def blochMessiahVecs(Z):
  """
  Obtain the full Bloch-Messiah decomposition.
  Less robust than version in decompositions.py; does not work well with degeneracies.
  """
  sigma = sqrtm(Z @ Z.T)
  eigenvalues, eigenvectors = eig(sigma)

  indices = np.argsort(eigenvalues)
  N = Z.shape[0] // 2
  indices = np.concatenate([indices[-1:-N-1:-1], indices[:N]])

  sortedEig = eigenvalues[indices].real
  sortedVec = eigenvectors[:, indices]

  u = inv(sigma) @ Z
  eigenvectors_ = eigenvectors.T @ u

  sortedVec_ = eigenvectors_[indices]

  return sortedEig, sortedVec, sortedVec_


def convertGreenFreqToTime(greenC, greenS):
  """
  Convert Green's matrix from frequency to time domain
  """
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


def calcRayleighWidth(length, wavelength, index):
  """
  Calculate the Rayleigh width (ie radius, for a Gaussian beam) for a given length and wavelength in the medium.
  """
  return np.sqrt(length * wavelength / (np.pi * index))


def basisTransforms(n):
  """
  Return the two matrices to transform from the a basis to the xp basis, and from the xp basis to the quadrature basis
  """
  toXPTrans = np.block([[np.eye(n),      np.eye(n)], [-1j * np.eye(n),  1j * np.eye(n)]]) / np.sqrt(2)
  frXPTrans = np.block([[np.eye(n), 1j * np.eye(n)], [      np.eye(n), -1j * np.eye(n)]]) / np.sqrt(2)
  return toXPTrans, frXPTrans


def combineGreens(Cfirst, Sfirst, Csecond, Ssecond):
  """
  Combine successive a basis C and S Green's matrices.
  """
  Ctotal = Csecond @ Cfirst + Ssecond * np.conjugate(Sfirst)
  Stotal = Csecond @ Sfirst + Ssecond * np.conjugate(Cfirst)
  return Ctotal, Stotal


def linearPoling(kMin, kMax, L, dL):
  """
  Create a poling design that has linearly increasing phase matching, up to a given resolution
  This is done by defining an instantaneous (spatial frequency) that varies linearly in z
  """
  z = np.linspace(dL / 2, L + dL / 2, round(L / dL))
  polingDirection = np.sign(np.sin(0.5 * (kMax - kMin) * z**2 / L + kMin * z))
  polingDirection[polingDirection == 0.] = 1. # TODO improve how we correct for 0

  p = np.concatenate([[0.], polingDirection, [0.]])
  polingProfile = np.diff(np.where(p[:-1] != p[1:]))
  return polingProfile.flatten()


def incoherentPowerGreens(Z):
  """
  Convert a Green's matrix in the quadrature basis into one for incoherent light.
  Note: output is a linear transformation for power.
  """
  N = Z.shape[0] // 2
  return 0.5 * (Z[:N, :N]**2 + Z[N:, N:]**2 + Z[:N, N:]**2 + Z[N:, :N]**2)
