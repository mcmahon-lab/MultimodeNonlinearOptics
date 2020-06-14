import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift, fft2, ifft2
from scipy.linalg import det, sqrtm, inv, eig, eigvals, norm


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
  greenC and greenS are such that a_out = C a + S a^†.
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


def calcCovarianceMtxABasis(greenC, greenS):
  """
  Calculate the covariance matrix in a basis from the transmission Green's matrices.
  """
  Z = np.block([[greenC, greenS],
                [greenS.conj(),  greenC.conj()]]) * np.sqrt(1 / 2)
  cov = Z @ Z.T.conj()
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

  localOscillX = np.hstack([freqDomain.real,  freqDomain.imag]) / norm(freqDomain)
  localOscillP = np.hstack([freqDomain.imag, -freqDomain.real]) / norm(freqDomain)

  covMatrix = np.zeros((2, 2))
  covMatrix[0,0] = localOscillX.T @ C @ localOscillX
  covMatrix[1,1] = localOscillP.T @ C @ localOscillP
  covMatrix[0,1] = covMatrix[1,0] = ((localOscillX + localOscillP).T @ C @ (localOscillX + localOscillP)
                                     - covMatrix[0,0] - covMatrix[1,1]) / 2
  # more efficient version of (localOscillX.T @ C @ localOscillP + localOscillP.T @ C @ localOscillX) / 2

  variances = eigvals(covMatrix)

  assert abs(covMatrix[0,0] - 1) < tol, "C_xx = %f" % covMatrix[0,0]

  variances[0], variances[1] = np.min(variances), np.max(variances)
  return variances


def downSampledCov(Z, perBin):
  """
  Downsample a covariance matrix by grouping modes together.
  assert N // 2 % perBin == 0
  """
  N = Z.shape[0]
  assert N // 2 % perBin == 0

  newZ = np.zeros((N // perBin, N))

  for i in range(N // perBin):
    newZ[:, i] = np.sum(Z[:, i].flatten().reshape(-1, perBin), axis=1) / np.sqrt(perBin)

  return newZ @ newZ.T


def downSampledCov(Z, measurements, omega, freqCutoff, nModeClasses=1):
  """
  Downsample a covariance matrix.
  For reducing the covariance matrix to the number of detectors, if these are less than the simulation window size.
  If the covariance matrix is made up of unrelated modes (ie different frequency bands),
  all with the same number of internal modes, these can be treated separately by setting nModeClasses.
  """
  freqCut = np.abs(fftshift(omega)) < freqCutoff
  nBinsCut = np.sum(freqCut)
  start = np.argmax(freqCut)

  sampling = np.round(np.linspace(start, start+nBinsCut-1, measurements)).astype(np.int)

  nt = Z.shape[0] // 2 // nModeClasses
  Zcut = np.empty((2 * nModeClasses * measurements, 2 * nModeClasses * nt), dtype=Z.dtype)

  subset = np.concatenate([sampling + nt * i for i in range(nModeClasses)])
  Zcut[:measurements*nModeClasses] = Z[subset]
  Zcut[measurements*nModeClasses:] = Z[nt * nModeClasses + subset]

  covCut = calcCovarianceMtx(Z, np.inf)

  return covCut


def downSampledCovABasis(C, S, measurements, omega, freqCutoff, nModeClasses=1):
  """
  Downsample a covariance matrix.
  For reducing the covariance matrix to the number of detectors, if these are less than the simulation window size.
  If the covariance matrix is made up of unrelated modes (ie different frequency bands),
  all with the same number of internal modes, these can be treated separately by setting nModeClasses.
  """
  freqCut = np.abs(fftshift(omega)) < freqCutoff
  nBinsCut = np.sum(freqCut)
  start = np.argmax(freqCut)

  sampling = np.round(np.linspace(start, start+nBinsCut-1, measurements)).astype(np.int)

  nt = C.shape[0] // nModeClasses
  Ccut = np.empty((nModeClasses * measurements, nModeClasses * nt), dtype=C.dtype)
  Scut = np.empty((nModeClasses * measurements, nModeClasses * nt), dtype=S.dtype)

  subset = np.concatenate([sampling + nt * i for i in range(nModeClasses)])
  Ccut[:] = C[subset]
  Scut[:] = S[subset]

  aCovCut = calcCovarianceMtxABasis(Ccut, Scut)

  return aCovCut


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
  polar = sqrtm(Z @ Z.T)
  diagonal, leftVectors = eig(polar)

  indices = np.argsort(diagonal)
  N = Z.shape[0] // 2
  indices = np.concatenate([indices[-1:-N-1:-1], indices[:N]])

  sortedDiagonal = diagonal[indices].real
  sortedLeftVecs = leftVectors[:, indices]

  polarUnitary = inv(polar) @ Z
  rightVectors = leftVectors.T @ polarUnitary

  sortedRightVecs = rightVectors[indices]

  return sortedDiagonal, sortedLeftVecs, sortedRightVecs


def ftGreens(greenC, greenS, toTime=True):
  """
  Convert Green's matrix from frequency to time domain
  """
  nt = greenS.shape[0]
  if toTime:
    convertedC = fftshift(fft(ifft(ifftshift(greenC), axis=0), axis=1))
    convertedS = fftshift(ifft2(ifftshift(greenS))) * nt
  else:
    convertedC = fftshift(ifft(fft(ifftshift(greenC), axis=0), axis=1))
    convertedS = fftshift(fft2(ifftshift(greenS))) / nt

  return convertedC, convertedS


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


def calcRayleighLength(width, wavelength, index):
  """
  Calculate the Rayleigh length (for a Gaussian beam) for a given width and wavelength in the medium.
  """
  return width**2 * np.pi * index / wavelength


def basisTransforms(n):
  """
  Return the two matrices to transform from the a basis to the xp basis, and from the xp basis to the a basis
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


def incoherentPowerGreens(Z):
  """
  Convert a Green's matrix in the quadrature basis into one for incoherent light.
  Note: output is a linear transformation for power.
  """
  N = Z.shape[0] // 2
  return 0.5 * (Z[:N, :N]**2 + Z[N:, N:]**2 + Z[:N, N:]**2 + Z[N:, :N]**2)


def parametricFluorescence(modes, diag):
  """
  Given decomposition of a transmission matrix in the complex frequency domain, predict observed parametric fluorescence.
  """
  incoherent = np.sum(np.abs(modes)**2 * np.sinh(np.log(diag[:, np.newaxis]))**2, axis=0)
  coherent   = np.abs(np.sum(modes * np.sinh(np.log(diag[:, np.newaxis])), axis=0))**2
  return incoherent, coherent


def effectiveAdjacencyMatrix(modes, diag):
  """
  Given decomposition of a transmission matrix in the complex frequency domain, find the effective GBS adjacency matrix.
  The matrix is B = U tanh(r_j) U^T, and the sampling probability is proportional to |Haf(B_n)|^2 for submatrix B_n.
  Expects a vector of exp(r_j).
  """
  return modes.T @ np.diag(np.tanh(np.abs(np.log(diag)))) @ modes


def fullEffectiveAdjacencyMatrix(cov):
  """
  Given a covariance matrix, find the full effective GBS adjacency matrix.
  Note: returns a 2Mx2M matrix, if the covariance matrix is purely a result of squeezing,
  the 1st and 4th quadrants are populated (quadrants are complex conjugates).
  The 2nd and 3rd quadrants are due to thermal modes, or losses (quadrants are transpositions).
  With only squeezing you can reduce to a MxM, Haf(*) -> |Haf(*)|^2 and you recover the form U tanh(r_j) U^T (as above).
  """
  n = cov.shape[0]
  idnt = np.identity(n)
  zero = np.zeros((n // 2, n // 2))
  return np.block([[zero, idnt[:n//2,:n//2]],
                   [idnt[:n//2,:n//2], zero]]) @ (np.eye(n) - inv(cov + np.eye(n) / 2))


def covLumpLoss(cov, loss):
  """
  Add a lump loss to a covariance matrix.
  This is equivalent to doubling the modes, applying a beam-splitter, then tracing out the new modes.
  In Green function formalism:
  Z' = [[t  ir] [[Z  0]
        [ir  t]] [0 I/2]]
  or = [[t  r]  [[Z  0]
        [r -t]]  [0 I/2]]
  Z' Z'^† = t^2 Z Z^† + r^2 I/2 = t^2 C + r^2 I/2
  """
  if not np.shape(loss):
    return (1 - loss) * cov + (loss * 0.5) * np.identity(cov.shape[0])

  else:
    reflection = np.sqrt(1 - loss)
    return np.outer(reflection, reflection) * cov + (loss * 0.5) * np.identity(cov.shape[0])
