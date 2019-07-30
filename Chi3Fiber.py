import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

class Chi3Fiber:
  """
  Class for numerically simulating chi(3) evolution of classical field as well as a signal or quantum field.
  """
  def __init__(self, relativeLength, nlLength, dispLength, beta2, beta2s, pulseType,
               beta1=0, beta1s=0, chirp=0, tMax=10, tPrecision=512, zPrecision=100):
    """
    :param relativeLength: Length of fiber in dispersion lengths, or nonlinear lengths if no dispersion.
    :param nlLength:       Nonlinear length in terms of dispersion length.
                           If no dispersion, must be 1. For no nonlinear effects, set to np.inf.
    :param dispLength:     Dispersion length. Must be kept at one or set to np.inf to remove dispersion.
    :param beta2:          Group velocity dispersion of the pump. Must be +/- 1.
    :param beta2s:         Group velocity dispersion of the signal, relative to the pump.
    :param pulseType:      0/False for Gaussian or 1/True for Hyperbolic Secant (soliton).
    :param beta1:          Group velocity difference for pump relative to simulation window.
    :param beta1s:         Group velocity difference for signal relative to simulation window.
    :param chirp:          Initial chirp for pump pulse.
    :param tMax:           Time window size in terms of pump width.
    :param tPrecision:     Number of time bins. Preferably power of 2 for better FFT performance.
    :param zPrecision:     Number of bins per unit length.
    """

    if not isinstance(relativeLength, (int, float)): raise TypeError("relativeLength")
    if not isinstance(nlLength,       (int, float)): raise TypeError("nlLength")
    if not isinstance(dispLength,     (int, float)): raise TypeError("dispLength")
    if not isinstance(beta2,  (int, float)): raise TypeError("beta2")
    if not isinstance(beta2s, (int, float)): raise TypeError("beta2s")
    if not isinstance(beta1,  (int, float)): raise TypeError("beta1")
    if not isinstance(beta1s, (int, float)): raise TypeError("beta1s")
    if not isinstance(chirp,  (int, float)): raise TypeError("chirp")
    if not isinstance(pulseType, (bool, int)): raise TypeError("pulseType")
    if not isinstance(tMax, int):       raise TypeError("tMax")
    if not isinstance(tPrecision, int): raise TypeError("tPrecision")
    if not isinstance(zPrecision, int): raise TypeError("zPrecision")

    self.setLengths(relativeLength, nlLength, dispLength, zPrecision)
    self.resetGrids(tPrecision, tMax)
    self.setDispersion(beta2, beta2s, beta1, beta1s)
    self.setPump(pulseType, chirp)

    # print("DS", self._DS, "NL", self._NL, "Nt", self._nFreqs, "Nz", self._nZSteps)


  def setLengths(self, relativeLength, nlLength, dispLength, zPrecision=100):
    # Equation defined in terms of dispersion and nonlinear lengh ratio N^2 = Lds / Lnl
    # The length z is given in units of dispersion length (of pump)
    # The time is given in units of initial width (of pump)

    # Total length (in units of pump dispersion length or, if infinite, nonlinear length)
    self._z = relativeLength

    # Nonlinear and dispersion length scales of the pump
    # DS should keep fixed at 1 to not mess up z unless no dispersion
    self._DS = dispLength

    # if no dispersion keep NL fixed at 1 to not mess up z, otherwise change relative to DS
    self._NL = nlLength

    self._noDispersion = True if self._DS == np.inf else False
    self._noNonlinear  = True if self._NL == np.inf else False

    if self._noDispersion:
      if self._NL != 1: raise ValueError("Non unit NL")
    else:
      if self._DS != 1: raise ValueError("Non unit DS")

    # Soliton order
    self._Nsquared = self._DS / self._NL
    if self._noDispersion: self._Nsquared = 1
    if self._noNonlinear:  self._Nsquared = 0

    # space resolution
    self._nZSteps = int(zPrecision * self._z / np.min([1, self._DS, self._NL]))
    self._dz = self._z / self._nZSteps

    # helper values
    self._nlStep = 1j * self._Nsquared * self._dz

    # TODO reset grids


  def resetGrids(self, nFreqs=None, tMax=None):

    # time windowing and resolution
    if nFreqs is not None:
      self._nFreqs = nFreqs
    if tMax is not None:
      self._tMax = tMax

    if not all(v is None for v in (nFreqs, tMax)):
      Nt = self._nFreqs
      dt = 2 * self._tMax / Nt

      # time and frequency axes
      self._tau = np.arange(-Nt / 2, Nt / 2) * dt
      self._omega = np.pi / self._tMax * fftshift(np.arange(-self._nFreqs / 2, self._nFreqs / 2))

      # TODO reset dispersion and pulse

    # Grids for PDE propagation
    self.pumpGridFreq = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex64)
    self.pumpGridTime = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex64)
    self.signalGridFreq = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex64)
    self.signalGridTime = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex64)


  def setDispersion(self, beta2, beta2s, beta1=0, beta1s=0):

    # positive or negative dispersion for pump (ie should be +/- 1), relative dispersion for signal
    self._beta2  = beta2
    self._beta2s = beta2s

    if abs(self._beta2) != 1 and not self._noDispersion:
      raise ValueError("Non unit beta2")

    # group velocity difference (relative to beta2 and pulse width)
    self._beta1  = beta1
    self._beta1s = beta1s

    # dispersion profile
    if self._noDispersion:
      self._dispersionPump = self._dispersionSign = 0
    else:
      self._dispersionPump = 0.5 * beta2  * self._omega**2 - beta1  * self._omega
      self._dispersionSign = 0.5 * beta2s * self._omega**2 - beta1s * self._omega

    # helper values
    self._dispStepPump = np.exp(1j * self._dispersionPump * self._dz)
    self._dispStepSign = np.exp(1j * self._dispersionSign * self._dz)


  def setPump(self, pulseType, chirp=0):
    # initial time domain envelopes (pick Gaussian or Soliton Hyperbolic Secant)
    if pulseType:
      self._env = 1 / np.cosh(self._tau) * np.exp(-0.5j * chirp * self._tau**2)
    else:
      self._env = np.exp(-0.5 * self._tau**2 * (1 + 1j * chirp))
    # TODO allow custom envelopes


  def runPumpSimulation(s):
    """
    Simulate propagation of pump field
    """
    s.pumpGridFreq[0, :] = fft(s._env) * np.exp(0.5j * s._dispersionPump * s._dz)
    s.pumpGridTime[0, :] = ifft(s.pumpGridFreq[0, :])

    for i in range(1, s._nZSteps):
      temp = s.pumpGridTime[i-1, :] * np.exp(s._nlStep * np.abs(s.pumpGridTime[i-1, :])**2)
      s.pumpGridFreq[i, :] = fft(temp) * s._dispStepPump
      s.pumpGridTime[i, :] = ifft(s.pumpGridFreq[i, :])

    s.pumpGridFreq[-1, :] *= np.exp(-0.5j * s._dispersionPump * s._dz)
    s.pumpGridTime[-1, :] = ifft(s.pumpGridFreq[-1, :])


  def runSignalSimulation(s, inputProf, freqSignal=True):
    """
    Simulate propagation of signal field
    :param inputProf: Frequency profile of input pulse
    :param freqSignal: input is in frequency domain if true, otherwise in time domain.
    """
    if freqSignal:
      s.signalGridFreq[0, :] = inputProf * np.exp(0.5j * s._dispersionSign * s._dz)
    else:
      s.signalGridFreq[0, :] = fft(inputProf) * np.exp(0.5j * s._dispersionSign * s._dz)

    s.signalGridTime[0, :] = ifft(s.signalGridFreq[0, :])

    for i in range(1, s._nZSteps):
      # Do a Runge-Kutta step for the non-linear propagation
      pumpTimeInterp = 0.5 * (s.pumpGridTime[i-1] + s.pumpGridTime[i])

      prevConj = np.conj(s.signalGridTime[i-1, :])
      k1 = s._nlStep * (2 * np.abs(s.pumpGridTime[i-1])**2 *  s.signalGridTime[i-1, :]             + s.pumpGridTime[i-1]**2 *  prevConj)
      k2 = s._nlStep * (2 * np.abs(pumpTimeInterp)**2      * (s.signalGridTime[i-1, :] + 0.5 * k1) + pumpTimeInterp**2      * (prevConj + np.conj(0.5 * k1)))
      k3 = s._nlStep * (2 * np.abs(pumpTimeInterp)**2      * (s.signalGridTime[i-1, :] + 0.5 * k2) + pumpTimeInterp**2      * (prevConj + np.conj(0.5 * k2)))
      k4 = s._nlStep * (2 * np.abs(s.pumpGridTime[i])**2   * (s.signalGridTime[i-1, :] + k3)       + s.pumpGridTime[i]**2   * (prevConj + np.conj(k3)))

      temp = s.signalGridTime[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

      # Dispersion step
      s.signalGridFreq[i, :] = fft(temp) * s._dispStepSign
      s.signalGridTime[i, :] = ifft(s.signalGridFreq[i, :])

    s.signalGridFreq[-1, :] *= np.exp(-0.5j * s._dispersionSign * s._dz)
    s.signalGridTime[-1, :] = ifft(s.signalGridFreq[-1, :])


  def computeGreensFunction(s):
    """
    Solve a(L, w) = C a(0, w) + S [a(0, w)]^t for C and S
    :return: Green's functions C, S
    """
    # Green function matrices
    greenC = np.zeros((s._nFreqs, s._nFreqs), dtype=np.complex64)
    greenS = np.zeros((s._nFreqs, s._nFreqs), dtype=np.complex64)

    s.runPumpSimulation()

    # Calculate Green's functions with real and imaginary impulse response
    for i in range(s._nFreqs):
      s.signalGridFreq[0, :] = 0
      s.signalGridFreq[0, i] = 1
      s.runSignalSimulation(s.signalGridFreq[0])
  
      greenC[i, :] += fftshift(s.signalGridFreq[-1, :] * 0.5)
      greenS[i, :] += fftshift(s.signalGridFreq[-1, :] * 0.5)
  
      s.signalGridFreq[0, :] = 0
      s.signalGridFreq[0, i] = 1j
      s.runSignalSimulation(s.signalGridFreq[0])

      greenC[i, :] -= fftshift(s.signalGridFreq[-1, :] * 0.5j)
      greenS[i, :] += fftshift(s.signalGridFreq[-1, :] * 0.5j)

    # Center Green's Functions
    for i in range(s._nFreqs):
      greenC[:, i] = fftshift(greenC[:, i])
      greenS[:, i] = fftshift(greenS[:, i])

    greenC = greenC.T
    greenS = greenS.T

    return greenC, greenS
