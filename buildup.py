if __name__ == "__main__":

  import numpy as np
  from numpy.fft import fft, ifft, fftshift
  import matplotlib.pyplot as plt

  # Equation defined in terms of dispersion and nonlinear lengh ratio N^2 = Lds / Lnl
  # The length z is given in units of dispersion length (of pump)
  # The time is given in units of initial width (of pump)

  # Total length (in units of pump dispersion length)
  z = 5

  # nonlinear and dispersion length scales of the pump
  NL = 1
  DS = 1 # should keep fixed at 1 to not mess up z

  # dispersion length of the signal
  DSs = 1.6

  # soliton order
  Nsquared = DS / NL

  # positive or negative dispersion for pump, relative dispersion for signal
  beta2  = -1 # should be +/- 1
  beta2s = 1

  # group velocity difference (relative to beta2 and pulse width)
  beta1  = 0
  beta1s = 0

  # Relative signal pulse width
  taus = np.sqrt(DSs * abs(beta2s) / DS)

  # initial chirp (pump)
  chirp = 0

  # time windowing and resolution
  Nt = 512
  tMax = 10
  dt = 2 * tMax / Nt

  # space resolution
  nZSteps = int(100 * z / np.min([DS, NL, DSs]))
  dz = z / nZSteps

  print(Nt, nZSteps)

  # time and frequency axes
  tau = np.arange(-Nt / 2, Nt / 2) * dt
  omega = np.pi / tMax * fftshift(np.arange(-Nt / 2, Nt / 2))

  # dispersion
  dispersionPump = 0.5 * beta2  * omega**2 - beta1  * omega
  dispersionSign = 0.5 * beta2s * omega**2 - beta1s * omega

  # initial time domain envelopes (Gaussian or Soliton Sech)
  env = 1 / np.cosh(tau) * np.exp(-0.5j * tau**2 * chirp)
  # env = np.exp(-0.5 * tau**2 * (1 + 1j * chirp))

  # sig = 1 / np.cosh(tau / taus)
  # sig = np.exp(-0.5 * (tau / taus)**2)
  sig = np.cos(tau / taus)


  # helpers
  nlStep = 1j * Nsquared * dz
  dispStepPump = np.exp(1j * dispersionPump * dz)
  dispStepSign = np.exp(1j * dispersionSign * dz)



  # Grids for PDE propagation
  computationGridPumpF  = np.zeros((nZSteps, Nt), dtype=np.complex64)
  computationGridPumpT  = np.zeros((nZSteps, Nt), dtype=np.complex64)
  computationGridSignalF = np.zeros((nZSteps, Nt), dtype=np.complex64)
  computationGridSignalT = np.zeros((nZSteps, Nt), dtype=np.complex64)


  # Start by computing pump propagation
  def runPumpSimulation():
    computationGridPumpF[0, :] = fft(env) * np.exp(0.5j * dispersionPump * dz)
    computationGridPumpT[0, :] = ifft(computationGridPumpF[0, :])

    for i in range(1, nZSteps):
      temp = computationGridPumpT[i-1, :] * np.exp(nlStep * np.abs(computationGridPumpT[i-1, :])**2)
      computationGridPumpF[i, :] = fft(temp) * dispStepPump
      computationGridPumpT[i, :] = ifft(computationGridPumpF[i, :])

    computationGridPumpF[-1, :] *= np.exp(-0.5j * dispersionPump * dz)
    computationGridPumpT[-1, :] = ifft(computationGridPumpF[-1, :])


  # Next, signal propagation
  def runSignalSimulation():
    computationGridSignalF[0, :] = fft(sig) * np.exp(0.5j * dispersionSign * dz)
    computationGridSignalT[0, :] = ifft(computationGridSignalF[0, :])

    for i in range(1, nZSteps):
      # do a Runge-Kutta step for the non-linear propagation
      pumpTimeInterp = 0.5 * (computationGridPumpT[i-1] + computationGridPumpT[i])

      prevConj = np.conj(computationGridSignalT[i-1, :])
      k1 = nlStep * (2 * np.abs(computationGridPumpT[i-1])**2 *  computationGridSignalT[i-1, :]             + computationGridPumpT[i-1]**2 *  prevConj)
      k2 = nlStep * (2 * np.abs(pumpTimeInterp)**2            * (computationGridSignalT[i-1, :] + 0.5 * k1) + pumpTimeInterp**2            * (prevConj + np.conj(0.5 * k1)))
      k3 = nlStep * (2 * np.abs(pumpTimeInterp)**2            * (computationGridSignalT[i-1, :] + 0.5 * k2) + pumpTimeInterp**2            * (prevConj + np.conj(0.5 * k2)))
      k4 = nlStep * (2 * np.abs(computationGridPumpT[i])**2   * (computationGridSignalT[i-1, :] + k3)       + computationGridPumpT[i]**2   * (prevConj + np.conj(k3)))

      temp = computationGridSignalT[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

      # dispersion step
      computationGridSignalF[i, :] = fft(temp) * dispStepSign
      computationGridSignalT[i, :] = ifft(computationGridSignalF[i, :])

    computationGridSignalF[-1, :] *= np.exp(-0.5j * dispersionSign * dz)
    computationGridSignalT[-1, :] = ifft(computationGridSignalF[-1, :])

  runPumpSimulation()
  runSignalSimulation()

  # Plot pump
  plt.plot(np.abs(env), label="initial env")
  plt.plot(np.abs(fftshift(fft(env))), label="initial spectrum")
  plt.plot(np.abs(computationGridPumpT[-1]), label="final env")
  plt.plot(fftshift(np.abs(computationGridPumpF[-1])), label="final spectrum")
  plt.legend()
  plt.show()

  # Plot Signal
  plt.plot(np.abs(sig), label="initial env")
  plt.plot(np.abs(fftshift(fft(sig))), label="initial spectrum")
  plt.plot(np.abs(computationGridSignalT[-1]), label="final env")
  plt.plot(fftshift(np.abs(computationGridSignalF[-1])), label="final spectrum")
  plt.legend()
  plt.show()

  # Plot Pump
  fig = plt.figure()
  ax = fig.add_subplot(1, 2, 1)
  plt.imshow(np.real(computationGridPumpF))
  plt.title('Pump Freq')
  ax = fig.add_subplot(1, 2, 2)
  plt.imshow(np.real(computationGridPumpT))
  plt.title('Pump Time')
  plt.show()

  # Plot Signal
  fig = plt.figure()
  ax = fig.add_subplot(1, 2, 1)
  plt.imshow(np.real(computationGridSignalF))
  plt.title('Signal Freq')
  ax = fig.add_subplot(1, 2, 2)
  plt.imshow(np.real(computationGridSignalT))
  plt.title('Signal Time')
  plt.show()
