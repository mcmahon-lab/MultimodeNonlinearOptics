if __name__ == "__main__":
  import numpy as np
  from numpy.fft import fft, ifft, fftshift, ifftshift, ifft2
  import matplotlib.pyplot as plt
  from decompositions import bloch_messiah, sympmat
  from scipy.linalg import dft

  # Equation defined in terms of dispersion and nonlinear lengh ratio N^2 = Lds / Lnl
  # The length z is given in units of dispersion length (of pump)
  # The time is given in units of initial width (of pump)

  # Total length (in units of pump dispersion length or, if infinite, nonlinear length)
  z = 1

  # nonlinear and dispersion length scales of the pump
  NL = 1 # if no dispersion keep fixed at 1 to not mess up z, otherwise change relative to DS
  # NL = np.inf
  DS = 1 # should keep fixed at 1 to not mess up z
  # DS = np.inf

  noDispersion = True if DS == np.inf else False
  noNonlinear  = True if NL == np.inf else False

  if noDispersion: assert NL == 1, "Non unit NL"
  else:            assert DS == 1, "Non unit DS"

  # soliton order
  Nsquared = DS / NL
  if noDispersion: Nsquared = 1
  if noNonlinear:  Nsquared = 0

  # positive or negative dispersion for pump, relative dispersion for signal
  beta2  = -1 # should be +/- 1
  beta2s = -1

  # group velocity difference (relative to beta2 and pulse width)
  beta1  = 0
  beta1s = 0

  # initial chirp (pump)
  chirp = 0

  # time windowing and resolution
  nFreqs = Nt = 512
  tMax = 10
  dt = 2 * tMax / Nt

  # space resolution
  nZSteps = int(100 * z / np.min([1, DS, NL]))
  dz = z / nZSteps

  print("DS", DS, "NL", NL, "Nt", Nt, "Nz", nZSteps)

  # time and frequency axes
  tau = np.arange(-Nt / 2, Nt / 2) * dt
  omega = np.pi / tMax * fftshift(np.arange(-nFreqs / 2, nFreqs / 2))

  # dispersion
  if noDispersion:
    dispersionPump = dispersionSign = 0
  else:
    dispersionPump = 0.5 * beta2  * omega**2 - beta1  * omega
    dispersionSign = 0.5 * beta2s * omega**2 - beta1s * omega

  # initial time domain envelopes (pick Gaussian or Soliton Hyperbolic Secant)
  env = 1 / np.cosh(tau) * np.exp(-0.5j * tau**2 * chirp)
  # env = np.exp(-0.5 * tau**2 * (1 + 1j * chirp))

  # helpers
  nlStep = 1j * Nsquared * dz
  dispStepPump = np.exp(1j * dispersionPump * dz)
  dispStepSign = np.exp(1j * dispersionSign * dz)

  # Grids for PDE propagation
  computationPumpF = np.zeros((nZSteps, nFreqs), dtype=np.complex64)
  computationPumpT = np.zeros((nZSteps, nFreqs), dtype=np.complex64)
  computationSignF = np.zeros((nZSteps, nFreqs), dtype=np.complex64)
  computationSignT = np.zeros((nZSteps, nFreqs), dtype=np.complex64)


  # Start by computing pump propagation
  def runPumpSimulation():
    computationPumpF[0, :] = fft(env) * np.exp(0.5j * dispersionPump * dz)
    computationPumpT[0, :] = ifft(computationPumpF[0, :])

    for i in range(1, nZSteps):
      temp = computationPumpT[i-1, :] * np.exp(nlStep * np.abs(computationPumpT[i-1, :])**2)
      computationPumpF[i, :] = fft(temp) * dispStepPump
      computationPumpT[i, :] = ifft(computationPumpF[i, :])

    computationPumpF[-1, :] *= np.exp(-0.5j * dispersionPump * dz)
    computationPumpT[-1, :] = ifft(computationPumpF[-1, :])


  # Next, signal propagation
  def runSignalSimulation(fSig):
    computationSignF[0, :] = fSig * np.exp(0.5j * dispersionSign * dz)
    computationSignT[0, :] = ifft(computationSignF[0, :])

    for i in range(1, nZSteps):
      # do a Runge-Kutta step for the non-linear propagation
      pumpTimeInterp = 0.5 * (computationPumpT[i-1] + computationPumpT[i])

      prevConj = np.conj(computationSignT[i-1, :])
      k1 = nlStep * (2 * np.abs(computationPumpT[i-1])**2 *  computationSignT[i-1, :]             + computationPumpT[i-1]**2 *  prevConj)
      k2 = nlStep * (2 * np.abs(pumpTimeInterp)**2        * (computationSignT[i-1, :] + 0.5 * k1) + pumpTimeInterp**2        * (prevConj + np.conj(0.5 * k1)))
      k3 = nlStep * (2 * np.abs(pumpTimeInterp)**2        * (computationSignT[i-1, :] + 0.5 * k2) + pumpTimeInterp**2        * (prevConj + np.conj(0.5 * k2)))
      k4 = nlStep * (2 * np.abs(computationPumpT[i])**2   * (computationSignT[i-1, :] + k3)       + computationPumpT[i]**2   * (prevConj + np.conj(k3)))

      temp = computationSignT[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

      # dispersion step
      computationSignF[i, :] = fft(temp) * dispStepSign
      computationSignT[i, :] = ifft(computationSignF[i, :])

    computationSignF[-1, :] *= np.exp(-0.5j * dispersionSign * dz)
    computationSignT[-1, :] = ifft(computationSignF[-1, :])


  # Green function matrices
  greenC = np.zeros((nFreqs, nFreqs), dtype=np.complex64)
  greenS = np.zeros((nFreqs, nFreqs), dtype=np.complex64)

  runPumpSimulation()

  for i in range(nFreqs):
    computationSignF[0, :] = 0
    computationSignF[0, i] = 1
    runSignalSimulation(computationSignF[0])

    greenC[i, :] += fftshift(computationSignF[-1, :] * 0.5)
    greenS[i, :] += fftshift(computationSignF[-1, :] * 0.5)

    computationSignF[0, :] = 0
    computationSignF[0, i] = 1j
    runSignalSimulation(computationSignF[0])

    greenC[i, :] -= fftshift(computationSignF[-1, :] * 0.5j)
    greenS[i, :] += fftshift(computationSignF[-1, :] * 0.5j)


  # Center Green's Functions
  for i in range(nFreqs):
    greenC[:, i] = fftshift(greenC[:, i])
    greenS[:, i] = fftshift(greenS[:, i])

  greenC = greenC.T
  greenS = greenS.T

  ### Results: ###

  # Plot pump propagation in time
  fig = plt.figure()
  ax = fig.add_subplot(2, 1, 1)
  plt.imshow(np.real(computationPumpT))
  plt.title("pump real profile")
  ax = fig.add_subplot(2, 1, 2)
  plt.plot(np.abs(computationPumpT[0]), label="pump init time")
  plt.plot(np.abs(computationPumpT[-1]), label="pump final time")
  plt.plot(fftshift(np.abs(computationPumpF[0])), label="pump init spec")
  plt.plot(fftshift(np.abs(computationPumpF[-1])), label="pump final spec")
  plt.legend()


  # Plot Green Functions
  fig = plt.figure()
  ax = fig.add_subplot(2, 2, 1)
  plt.imshow(np.abs(greenC), origin='lower')
  plt.title('$|C(\omega)|$')
  plt.colorbar()
  ax = fig.add_subplot(2, 2, 3)
  plt.imshow(np.angle(greenC), origin='lower', cmap="hsv")
  plt.title('$\\arg[C(\omega)]$')
  plt.colorbar()
  ax = fig.add_subplot(2, 2, 2)
  plt.imshow(np.abs(greenS), origin='lower')
  plt.title('$|S(\omega)|$')
  plt.colorbar()
  ax = fig.add_subplot(2, 2, 4)
  plt.imshow(np.angle(greenS), origin='lower', cmap="hsv")
  plt.title('$\\arg[S(\omega)]$')
  plt.colorbar()


  ##### Calculate Squeezing

  # X and P output transformation matrix [xf, pf] = Z [xi, pi]
  Z = np.block([[np.real(greenC + greenS), -np.imag(greenC - greenS)],
                [np.imag(greenC + greenS),  np.real(greenC - greenS)]]).astype(dtype=np.float_)

  assert not np.iscomplexobj(Z), "Complex object"

  # Covariance Matrix
  C = Z @ Z.T
  print("Det(C) =", np.linalg.det(C))

  # Covariance Matrix in time domain
  dftMat = np.conj(dft(nFreqs))
  gCtime = ifft2(fftshift(greenC)) * nFreqs
  gStime = dftMat.T @ fftshift(greenS) @ dftMat / nFreqs
  Ztime = np.block([[np.real(gCtime + gStime), -np.imag(gCtime - gStime)],
                    [np.imag(gCtime + gStime),  np.real(gCtime - gStime)]]).astype(dtype=np.float_)
  Ctime = Ztime @ Ztime.T

  # Normalized covariance matrix
  diagC = np.diag(C)
  normC = np.tile(diagC, (diagC.shape[0], 1))
  normalizedC = (C - np.eye(C.shape[0])) / np.sqrt(normC * normC.T)

  fig = plt.figure()
  ax = fig.add_subplot(1, 3, 1)
  plt.imshow(normalizedC)
  plt.title("normalized C")
  plt.colorbar()
  ax = fig.add_subplot(1, 3, 2)
  plt.imshow(C)
  plt.title("C")
  plt.colorbar()
  ax = fig.add_subplot(1, 3, 3)
  plt.imshow(Ctime)
  plt.title("$\~C$")
  plt.colorbar()


  # Squeezing of pulse as LO
  localOscillX = np.hstack([computationPumpF[-1].real,  computationPumpF[-1].imag]) / np.linalg.norm(computationPumpF[-1])
  localOscillP = np.hstack([computationPumpF[-1].imag, -computationPumpF[-1].real]) / np.linalg.norm(computationPumpF[-1])

  covMatrix = np.zeros((2, 2))
  covMatrix[0,0] = localOscillX.T @ C @ localOscillX
  covMatrix[1,1] = localOscillP.T @ C @ localOscillP
  covMatrix[0,1] = covMatrix[1,0] = (localOscillX.T @ C @ localOscillP + localOscillP.T @ C @ localOscillX) / 2

  variances = np.linalg.eigvals(covMatrix)
  uncertainty = np.sqrt(np.prod(variances))

  print("variances\t", variances)
  print("uncertainty\t", uncertainty)


  # Supermode squeezing

  # # check Z is symplectic
  # omega = sympmat(nFreqs)
  # fig = plt.figure()
  # ax = fig.add_subplot(1, 2, 1)
  # plt.imshow(np.real(Z @ omega @ Z.T), origin='lower')
  # plt.title('$Re(Z \Omega Z^T)$')
  # plt.colorbar()
  # ax = fig.add_subplot(1, 2, 2)
  # plt.imshow(np.imag(M @ omega @ M.T), origin='lower')
  # plt.title('$Im(Z \Omega Z^T)$')
  # plt.colorbar()

  O1, S, O2 = bloch_messiah(Z, tol=5e-5)
  print("O1 @ S @ O2 = Z", np.allclose(O1 @ S @ O2, Z, atol=5e-5))
  diagSqueezing = S.diagonal()
  fig = plt.figure()
  plt.plot(diagSqueezing[nFreqs:], "s-", markerfacecolor="none", label="squeezed variance")
  plt.plot(diagSqueezing[:nFreqs], "s-", markerfacecolor="none", label="anti-squeezed variance")
  plt.plot(diagSqueezing[nFreqs:] * diagSqueezing[:nFreqs], "s-", markerfacecolor="none", label="uncertainty product")
  plt.legend()


  # Test Greens Functions with an input
  # Signal profile
  taus = 1
  aInProfileTime = np.exp(-0.5 * (tau / taus)**2) # * 1j
  # aInProfileTime = 1 / np.cosh(tau/taus)
  aInProfileFreq = fftshift(fft(aInProfileTime))
  aOutProfileFreq = (greenC @ aInProfileFreq + greenS @ np.conj(aInProfileFreq)).flatten()
  aOutProfileTime = ifft(aOutProfileFreq)
  runSignalSimulation(ifftshift(aInProfileFreq))
  aOutProfileFSim = ifftshift(computationSignF[-1])
  print("aOut = C @ aIn + S @ a^t", np.allclose(aOutProfileFreq, aOutProfileFSim, atol=1e-4))

  fig = plt.figure()
  plt.plot(np.abs(aInProfileTime),  label="test signal init time")
  plt.plot(np.abs(aOutProfileTime), label="test signal final time Green")
  plt.plot(np.abs(computationSignT[-1]), label="test signal final time Prop")
  plt.plot(np.abs(aInProfileFreq),  label="test signal init spec")
  plt.plot(np.abs(aOutProfileFreq), label="test signal final spec Green")
  plt.plot(np.abs(ifftshift(computationSignF[-1])), label="test signal final spec Prop")
  plt.legend()


  # Check that Z is correct
  fProfInit = aInProfileFreq
  fProfFine = aOutProfileFSim
  xpInit = np.concatenate([fProfInit.real, fProfInit.imag])
  xpFine = np.concatenate([fProfFine.real, fProfFine.imag])
  xpFinZ = np.dot(Z, xpInit)
  print("xpFin = Z @ xpInit", np.allclose(xpFine, xpFinZ, atol=1e-4))
  fig = plt.figure()
  # plt.plot(np.abs(xpInit), label='$r_i$')
  plt.plot(np.abs(xpFine), label='$r_f$')
  plt.plot(np.abs(xpFinZ), label='$Z \cdot r_0$')
  plt.legend()


  plt.show()
