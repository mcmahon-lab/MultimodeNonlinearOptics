"""
Module for helper functions related to multimode and gaussian quantum optics 
Note: the functions provided here use a convention of hbar = 2, such that:
 x = a^† + a
 p = i (a^† - a)
 [a_i^†, a_j] = δ_ij
 [x_i, p_j] = 2 i δ_ij
"""

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift, fft2, ifft2
from scipy.linalg import det, sqrtm, inv, eig, eigvals, norm


def calcQuadratureGreens(greenC, greenS):
  """
  Convert the Green's matrix to the x and p quadrature basis from the a basis (bosonic)
  greenC and greenS are such that a_out = C a + S a^†.
  Derived using x = a^† + a, p = i(a^† - a).
  """
  Z = np.block([[np.real(greenC + greenS), -np.imag(greenC - greenS)],
                [np.imag(greenC + greenS),  np.real(greenC - greenS)]]).astype(dtype=np.float_)

  return Z


def calcCovarianceMtx(Z, tol=1e-4):
  """
  Calculate the covariance matrix in x/p quadrature basis from the transmission Green's matrix.
  Checks that the determinant of the covariance matrix is approximately unity.
  Derived using x_i x_j = 1/2 {Z_ik x_k, Z_jk x_k}.
  """
  cov = Z @ Z.T
  determinant = det(cov)
  assert abs(determinant - 1) < tol, "det(C) = %f" % determinant
  return cov


def calcCovarianceMtxABasis(greenC, greenS):
  """
  Calculate the covariance matrix in a basis (bosonic) from the transmission Green's matrices.
  Derived using a_i a_j = 1/2 {Z_ik a_k, Z_jk a_k}.
  """
  Z = np.block([[greenC, greenS],
                [greenS.conj(), greenC.conj()]]) * np.sqrt(1 / 2)
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
  Compute the squeezing observed combining the covariance matrix and a local oscillator
  with a given profile.
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


def downSampledCov(Z, measurements, omega, freqCutoff, nModeClasses=1):
  """
  Downsample a covariance matrix.
  For reducing the covariance matrix to the number of detectors, if these are fewer
  than the simulation window size.
  If the covariance matrix is made up of unrelated modes (ie different frequency bands),
  with the same number of internal modes, these can be treated separately by setting nModeClasses.
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

  covCut = calcCovarianceMtx(Zcut, np.inf)

  return covCut


def downSampledCovABasis(C, S, measurements, omega, freqCutoff, nModeClasses=1):
  """
  Downsample a covariance matrix.
  For reducing the covariance matrix to the number of detectors, if these are fewer
  than the simulation window size.
  If the covariance matrix is made up of unrelated modes (ie different frequency bands),
  with the same number of internal modes, these can be treated separately by setting nModeClasses.
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
  Convert bosonic Green's matrices to and from the frequency and time domain.
  Derivation:
  Fa = F(Ca + Sa*) = (F C F^-1)(F a) + (F S F)(F^-1 a*) = (F C F^-1)(F a) + (F S F)(F a)*, since F^-1 = F*
  """
  if toTime:
    convertedC = fftshift(fft(ifft(ifftshift(greenC), axis=0), axis=1))
    convertedS = fftshift(ifft2(ifftshift(greenS))) * greenS.shape[0]
  else:
    convertedC = fftshift(ifft(fft(ifftshift(greenC), axis=0), axis=1))
    convertedS = fftshift(fft2(ifftshift(greenS))) / greenS.shape[0]

  return convertedC, convertedS


def basisTransforms(n):
  """
  Return the two matrices to transform from the a basis (bosonic) to the xp basis (quadrature),
  and back from the xp basis to the a basis.
  """
  toXPTrans = np.block([[np.eye(n),      np.eye(n)], [-1j * np.eye(n),  1j * np.eye(n)]])
  frXPTrans = np.block([[np.eye(n), 1j * np.eye(n)], [      np.eye(n), -1j * np.eye(n)]]) * 0.5
  return toXPTrans, frXPTrans


def combineGreens(Cfirst, Sfirst, Csecond, Ssecond):
  """
  Combine successive a basis (bosonic) C and S Green's matrices.
  """
  Ctotal = Csecond @ Cfirst + Ssecond @ np.conjugate(Sfirst)
  Stotal = Csecond @ Sfirst + Ssecond @ np.conjugate(Cfirst)
  return Ctotal, Stotal


def incoherentPowerGreens(Z):
  """
  Calculate a transformatoin matrix for incoherent light from a Green's matrix in the quadrature basis.
  Note: output is a linear transformation for power.
  Derived by parametrizing x = cos θ, p = sin θ and averaging over all
  x_j^2 + p_j^2 = (Z_ji^xx cos θ + Z_ji^xp sin θ)^2 + (Z_ji^px cos θ + Z_ji^pp sin θ)^2
  """
  N = Z.shape[0] // 2
  return 0.5 * (Z[:N, :N]**2 + Z[N:, N:]**2 + Z[:N, N:]**2 + Z[N:, :N]**2)


def photonMean(V):
  """
  Calculate the mean photon number per mode from a Gaussian quadrature covariance matrix.
  n = a^† a = 1/4 (x + i p) (x - i p) = 1/4 (x^2 + p^2 - i [x, p]) = 1/4 (x^2 + p^2 - 2).
  Note: this does not include displacements.
  """
  N = V.shape[0] // 2
  return 0.25 * (V[:N, :N].diagonal() + V[N:, N:].diagonal() - 2)

def photonMeanA(sigma):
  """
  Calculate the mean photon number per mode from a Gaussian covariance matrix in the a basis.
  n = a^† a = a a^† - 1/2.
  Note: this does not include displacements.
  """
  N = sigma.shape[0] // 2
  return sigma[:N, :N].diagonal().real - 0.5


def photonCov(V):
  """
  Calculate the photon number covariance from a Gaussian quadrature covariance matrix.
  Note: only for zero mean displacement states, non-zero displacement requires additional terms.
  See "Mode structure and photon number correlations in squeezed quantum pulses".
  Derived using different convention, n_k = 1/4 (x_k^2 + p_k^2 - 2).
  """
  N = V.shape[0] // 2
  return 0.125 * (V[:N, :N]**2 + V[:N, N:]**2 + V[N:, :N]**2 + V[N:, N:]**2) - 0.25 * np.eye(N)

def photonCovA(sigma):
  """
  Calculate the photon number covariance from a covariance matrix in the a basis.
  """
  N = V.shape[0] // 2
  return np.abs(sigma[:nt, :nt]).real**2 + np.abs(sigma[:nt, nt:]).real**2 - 0.25 * np.eye(N)


def parametricFluorescence(modes, diag):
  """
  Given decomposition of a transmission matrix in the complex frequency domain,
  predict observed parametric fluorescence.
  """
  return np.sum(np.abs(modes)**2 * np.sinh(np.log(diag[:, np.newaxis]))**2, axis=0)


def effectiveAdjacencyMatrix(modes, diag):
  """
  Given decomposition of a transmission matrix in the complex frequency domain,
  find the effective GBS adjacency matrix.
  The matrix is B = U tanh(r_j) U^T, and the sampling probability is proportional
  to |Haf(B_n)|^2 for submatrix B_n.
  Expects a vector of exp(r_j).
  """
  return modes.T @ np.diag(np.tanh(np.abs(np.log(diag)))) @ modes


def fullEffectiveAdjacencyMatrix(cov):
  """
  Given a covariance matrix, find the full effective GBS adjacency matrix.
  Note: returns a 2Mx2M matrix, if the covariance matrix is purely a result of squeezing,
  the 1st and 4th quadrants are populated (quadrants are complex conjugates).
  The 2nd and 3rd quadrants are due to thermal modes, or losses (quadrants are transpositions).
  With pure squeezing, one can reduce to a MxM matrix, Haf(*) -> |Haf(*)|^2,
  and recover the form U tanh(r_j) U^T (as above).
  """
  n = cov.shape[0]
  idnt = np.identity(n)
  zero = np.zeros((n // 2, n // 2))
  return np.block([[zero, idnt[:n//2,:n//2]],
                   [idnt[:n//2,:n//2], zero]]) @ (idnt - inv(cov + 0.5 * idnt))


def covLumpLoss(cov, transmission, abasis=False):
  """
  Add a lump loss to a covariance matrix.
  This is equivalent to doubling the modes, applying a beam-splitter, and tracing out the new modes.
  In Green function formalism:
  Z' = [[t  ir] [[Z 0]
        [ir  t]] [0 I/2]] (bosonic operators)
  or = [[t  r]  [[Z 0]
        [r -t]]  [0 I]]   (quadrature operators)
  Z' Z'^† = t^2 Z Z^† + r^2 I [1/2] = t^2 C + r^2 I [1/2]
  """
  vacuumVar = (0.5 if abasis else 1)
  if not np.shape(transmission):
    return transmission * cov + ((1 - transmission) * vacuumVar) * np.identity(cov.shape[0])

  else:
    transmission = np.tile(transmission, 2)
    sqrtTrans = np.sqrt(transmission)
    return np.outer(sqrtTrans, sqrtTrans) * cov + ((1 - transmission) * vacuumVar) * np.identity(cov.shape[0])

