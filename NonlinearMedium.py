import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

class _NonlinearMedium:
  """
  Base class for numerically simulating the evolution of a classical field in nonlinear media with a signal or quantum field.
  """
  def __init__(self, relativeLength, nlLength, dispLength, beta2, beta2s, pulseType=0,
               beta1=0, beta1s=0, beta3=0, beta3s=0, chirp=0, tMax=10, tPrecision=512, zPrecision=100,
               customPump=None):
    """
    :param relativeLength: Length of fiber in dispersion lengths, or nonlinear lengths if no dispersion.
    :param nlLength:       Nonlinear length in terms of dispersion length.
                           If no dispersion, must be 1. For no nonlinear effects, set to np.inf.
    :param dispLength:     Dispersion length. Must be kept at one or set to np.inf to remove dispersion.
    :param beta2:          Group velocity dispersion of the pump. Must be +/- 1.
    :param beta2s:         Group velocity dispersion of the signal, relative to the pump.
    :param pulseType:      Pump profile, 0/False for Gaussian or 1/True for Hyperbolic Secant (soliton).
    :param beta1:          Group velocity difference for pump relative to simulation window.
    :param beta1s:         Group velocity difference for signal relative to simulation window.
    :param beta3:          Pump third order dispersion.
    :param beta3s:         Signal third order dispersion.
    :param chirp:          Initial chirp for pump pulse.
    :param tMax:           Time window size in terms of pump width.
    :param tPrecision:     Number of time bins. Preferably power of 2 for better FFT performance.
    :param zPrecision:     Number of bins per unit length.
    :param customPump:     Specify a pump profile in time domain.
    """

    self._checkInput(relativeLength, nlLength, dispLength, beta2, beta2s, pulseType,
                     beta1, beta1s, beta3, beta3s, chirp, tMax, tPrecision, zPrecision,
                     customPump)
    self.setLengths(relativeLength, nlLength, dispLength, zPrecision)
    self.resetGrids(tPrecision, tMax)
    self.setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s)
    self.setPump(pulseType, chirp, customPump)


  def _checkInput(self, relativeLength, nlLength, dispLength, beta2, beta2s, pulseType,
                  beta1, beta1s, beta3, beta3s, chirp, tMax, tPrecision, zPrecision,
                  customPump):

    if not isinstance(relativeLength, (int, float)): raise TypeError("relativeLength")
    if not isinstance(nlLength,       (int, float)): raise TypeError("nlLength")
    if not isinstance(dispLength,     (int, float)): raise TypeError("dispLength")
    if not isinstance(beta2,  (int, float)): raise TypeError("beta2")
    if not isinstance(beta2s, (int, float)): raise TypeError("beta2s")
    if not isinstance(beta1,  (int, float)): raise TypeError("beta1")
    if not isinstance(beta1s, (int, float)): raise TypeError("beta1s")
    if not isinstance(beta3,  (int, float)): raise TypeError("beta3")
    if not isinstance(beta3s, (int, float)): raise TypeError("beta3s")
    if not isinstance(chirp,  (int, float)): raise TypeError("chirp")
    if not isinstance(pulseType, (bool, int)): raise TypeError("pulseType")
    if not isinstance(tMax, int):       raise TypeError("tMax")
    if not isinstance(tPrecision, int): raise TypeError("tPrecision")
    if not isinstance(zPrecision, int): raise TypeError("zPrecision")
    if not isinstance(customPump, (type(None), np.ndarray)): raise TypeError("customPump")


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

    # Reset grids -- skip during construction
    try:
      self.resetGrids()
    except AttributeError:
      pass


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
      self.tau = ifftshift(np.arange(-Nt / 2, Nt / 2) * dt)
      self.omega = np.pi / self._tMax * fftshift(np.arange(-self._nFreqs / 2, self._nFreqs / 2))

      # Reset dispersion and pulse
      try:
        self.setDispersion(self._beta2, self._beta2s, self._beta1, self._beta1)
      except AttributeError:
        pass

    # Grids for PDE propagation
    self.pumpFreq = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex64)
    self.pumpTime = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex64)
    self.signalFreq = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex64)
    self.signalTime = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex64)


  def setDispersion(self, beta2, beta2s, beta1=0, beta1s=0, beta3=0, beta3s=0):

    # positive or negative dispersion for pump (ie should be +/- 1), relative dispersion for signal
    self._beta2  = beta2
    self._beta2s = beta2s

    # group velocity difference (relative to beta2 and pulse width)
    self._beta1  = beta1
    self._beta1s = beta1s

    # third order dispersion (relative to beta2 and pulse width, unless beta2 is zero, then should be +/- 1)
    self._beta3  = beta3
    self._beta3s = beta3s

    if abs(self._beta2) != 1 and not self._noDispersion:
      if self._beta2 != 0 or abs(self._beta3) != 1:
        raise ValueError("Non unit beta2")

    # dispersion profile
    if self._noDispersion:
      self._dispersionPump = self._dispersionSign = 0
    else:
      self._dispersionPump = 0.5 * beta2  * self.omega**2 + beta1  * self.omega + 1/6 * beta3  * self.omega**3
      self._dispersionSign = 0.5 * beta2s * self.omega**2 + beta1s * self.omega + 1/6 * beta3s * self.omega**3

    # helper values
    self._dispStepPump = np.exp(1j * self._dispersionPump * self._dz)
    self._dispStepSign = np.exp(1j * self._dispersionSign * self._dz)


  def setPump(self, pulseType=0, chirp=0, customPump=None):
    # initial time domain envelopes (pick Gaussian or Soliton Hyperbolic Secant or custom)
    if customPump is not None:
      if customPump.size != self._nFreqs:
        raise ValueError("Custom pump array length does not match number of frequency/time bins")
      self._env = customPump * np.exp(-0.5j * chirp * self.tau**2)
    else:
      if pulseType:
        self._env = 1 / np.cosh(self.tau) * np.exp(-0.5j * chirp * self.tau**2)
      else:
        self._env = np.exp(-0.5 * self.tau**2 * (1 + 1j * chirp))


  def runPumpSimulation(s):
    """
    Simulate propagation of pump field
    """
    pass


  def runSignalSimulation(s, inputProf, inTimeDomain=True):
    """
    Simulate propagation of signal field
    :param inputProf: Profile of input pulse. Can be time or frequency domain.
    Note: Frequency domain input is assumed to be "true" frequency with omega as its axis (since there are phase issues
    associated with FFT on a non-centered array, since FFT considers the center the first and last elements).
    :param inTimeDomain: Specify if input is in frequency or frequency domain. True for time, false for frequency.
    """
    pass


  def computeGreensFunction(s, inTimeDomain=False, runPump=True):
    """
    Solve a(L) = C a(0) + S [a(0)]^t for C and S
    :param inTimeDomain Compute the Green's function in time or frequency domain.
    :param runPump      Whether to run pump simulation beforehand.
    :return: Green's functions C, S
    """
    # Green function matrices
    greenC = np.zeros((s._nFreqs, s._nFreqs), dtype=np.complex64)
    greenS = np.zeros((s._nFreqs, s._nFreqs), dtype=np.complex64)

    if runPump: s.runPumpSimulation()

    grid = (s.signalTime if inTimeDomain else s.signalFreq)

    # Calculate Green's functions with real and imaginary impulse response
    for i in range(s._nFreqs):
      grid[0, :] = 0
      grid[0, i] = 1
      s.runSignalSimulation(grid[0], inTimeDomain)

      greenC[i, :] += grid[-1, :] * 0.5
      greenS[i, :] += grid[-1, :] * 0.5

      grid[0, :] = 0
      grid[0, i] = 1j
      s.runSignalSimulation(grid[0], inTimeDomain)

      greenC[i, :] -= grid[-1, :] * 0.5j
      greenS[i, :] += grid[-1, :] * 0.5j

    greenC = greenC.T
    greenS = greenS.T

    return fftshift(greenC), fftshift(greenS)


class Chi3(_NonlinearMedium):
  """
  Class for numerically simulating the evolution of a classical field in a chi(3) medium with a signal or quantum field.
  """
  def runPumpSimulation(s):
    __doc__ = _NonlinearMedium.runPumpSimulation.__doc__

    s.pumpFreq[0, :] = fft(s._env) * np.exp(0.5j * s._dispersionPump * s._dz)
    s.pumpTime[0, :] = ifft(s.pumpFreq[0, :])

    for i in range(1, s._nZSteps):
      temp = s.pumpTime[i-1, :] * np.exp(s._nlStep * np.abs(s.pumpTime[i-1, :])**2)
      s.pumpFreq[i, :] = fft(temp) * s._dispStepPump
      s.pumpTime[i, :] = ifft(s.pumpFreq[i, :])

    s.pumpFreq[-1, :] *= np.exp(-0.5j * s._dispersionPump * s._dz)
    s.pumpTime[-1, :] = ifft(s.pumpFreq[-1, :])


  def runSignalSimulation(s, inputProf, inTimeDomain=True):
    __doc__ = _NonlinearMedium.runSignalSimulation.__doc__

    inputProfFreq = (fft(inputProf) if inTimeDomain else inputProf)

    s.signalFreq[0, :] = inputProfFreq * np.exp(0.5j * s._dispersionSign * s._dz)
    s.signalTime[0, :] = ifft(s.signalFreq[0, :])

    for i in range(1, s._nZSteps):
      # Do a Runge-Kutta step for the non-linear propagation
      pumpTimeInterp = 0.5 * (s.pumpTime[i-1] + s.pumpTime[i])

      prevConj = np.conj(s.signalTime[i-1, :])
      k1 = s._nlStep * (2 * np.abs(s.pumpTime[i-1])**2 *  s.signalTime[i-1, :]             + s.pumpTime[i-1]**2 *  prevConj)
      k2 = s._nlStep * (2 * np.abs(pumpTimeInterp)**2  * (s.signalTime[i-1, :] + 0.5 * k1) + pumpTimeInterp**2  * (prevConj + np.conj(0.5 * k1)))
      k3 = s._nlStep * (2 * np.abs(pumpTimeInterp)**2  * (s.signalTime[i-1, :] + 0.5 * k2) + pumpTimeInterp**2  * (prevConj + np.conj(0.5 * k2)))
      k4 = s._nlStep * (2 * np.abs(s.pumpTime[i])**2   * (s.signalTime[i-1, :] + k3)       + s.pumpTime[i]**2   * (prevConj + np.conj(k3)))

      temp = s.signalTime[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

      # Dispersion step
      s.signalFreq[i, :] = fft(temp) * s._dispStepSign
      s.signalTime[i, :] = ifft(s.signalFreq[i, :])

    s.signalFreq[-1, :] *= np.exp(-0.5j * s._dispersionSign * s._dz)
    s.signalTime[-1, :] = ifft(s.signalFreq[-1, :])


class Chi2(_NonlinearMedium):
  def runPumpSimulation(s):
    __doc__ = _NonlinearMedium.runPumpSimulation.__doc__

    s.pumpFreq[0, :] = fft(s._env)
    s.pumpTime[0, :] = s._env

    for i in range(1, s._nZSteps):
      s.pumpFreq[i, :] = s.pumpFreq[0, :] * np.exp(1j * i * s._dispersionPump * s._dz)
      s.pumpTime[i, :] = ifft(s.pumpFreq[i, :])


  def runSignalSimulation(s, inputProf, inTimeDomain=True):
    __doc__ = _NonlinearMedium.runSignalSimulation.__doc__

    inputProfFreq = (fft(inputProf) if inTimeDomain else inputProf)

    s.signalFreq[0, :] = inputProfFreq * np.exp(0.5j * s._dispersionSign * s._dz)
    s.signalTime[0, :] = ifft(s.signalFreq[0, :])

    for i in range(1, s._nZSteps):
      # Do a Runge-Kutta step for the non-linear propagation
      pumpTimeInterp = 0.5 * (s.pumpTime[i-1] + s.pumpTime[i])

      prevConj = np.conj(s.signalTime[i-1, :])
      k1 = s._nlStep * s.pumpTime[i-1, :] *  prevConj
      k2 = s._nlStep * pumpTimeInterp     * (prevConj + np.conj(0.5 * k1))
      k3 = s._nlStep * pumpTimeInterp     * (prevConj + np.conj(0.5 * k2))
      k4 = s._nlStep * s.pumpTime[i,   :] * (prevConj + np.conj(k3))

      temp = s.signalTime[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

      # Dispersion step
      s.signalFreq[i, :] = fft(temp) * s._dispStepSign
      s.signalTime[i, :] = ifft(s.signalFreq[i, :])

    s.signalFreq[-1, :] *= np.exp(-0.5j * s._dispersionSign * s._dz)
    s.signalTime[-1, :] = ifft(s.signalFreq[-1, :])


class Cascade(_NonlinearMedium):
  """
  Class that cascades multiple media together
  """
  def __init__(self, sharedPump, media):

    if not isinstance(sharedPump, (bool, int)):
      raise TypeError("sharedPump must be boolean")

    if len(media) == 0:
      ValueError("Cascade must contain at least one medium")

    for i, medium in enumerate(media):
      if not isinstance(medium, _NonlinearMedium):
        raise TypeError("Argument %d is not a NonlinearMedium object" % i)

    self._nFreqs = media[0]._nFreqs
    self._tMax = media[0]._tMax
    for i, medium in enumerate(media):
      if medium._nFreqs != self._nFreqs or medium._tMax != self._tMax:
        ValueError("Medium %d does not have same time and frequency axes as the first" % i)

    self.media = [medium for medium in media]
    self.nMedia = len(self.media)
    self.sharedPump = bool(sharedPump)

    # initialize parent class values to shared/combined values of cascaded media
    self.tau = self.media[0].tau
    self.omega = self.media[0].omega
    self._nZSteps = np.sum([medium._nZSteps for medium in self.media])
    # initialize parent class values to null values
    # self._z = self._DS = self._NL = 0
    # self._noDispersion = self._noNonlinear = False
    # self._Nsquared = self._dz = self._nlStep = 0
    # self.pumpFreq = self.pumpTime = self.signalFreq = self.signalTime = None
    # self._beta2 = self._beta2s = self._beta1 = self._beta1s = self._beta3 = self._beta3s = 0
    # self._dispersionPump = self._dispersionSign = self._dispStepPump = self._dispStepSign = self._env = None


  def addMedium(self, medium):
    if not isinstance(medium, _NonlinearMedium):
      raise TypeError("Argument is not a NonlinearMedium object")

    if medium._nFreqs != self.media[0]._nFreqs or medium._tMax != self.media[0]._tMax:
      ValueError("Medium does not have same time and frequency axes as the first")

    self.media.append(medium)
    self.nMedia += 1
    self._nZSteps += medium._nZSteps


  def runPumpSimulation(self):
    __doc__ = _NonlinearMedium.runPumpSimulation.__doc__

    if not self.sharedPump:
      for medium in self.media:
        medium.runPumpSimulation()

    else:
      self.media[0].runPumpSimulation()
      for i in range(1, len(self.media)):
        self.media[i]._env = self.media[i-1].pumpTime[-1]
        self.media[i].runPumpSimulation()


  def runSignalSimulation(self, inputProf, inTimeDomain=True):
    __doc__ = _NonlinearMedium.runSignalSimulation.__doc__

    self.media[0].runSignalSimulation(inputProf, inTimeDomain)
    for i in range(1, len(self.media)):
      self.media[i].runSignalSimulation(self.media[i-1].signalFreq[-1], inTimeDomain=False)


  def setLengths(self, relativeLength, nlLength, dispLength, zPrecision=100):
    """Invalid"""
    pass

  def resetGrids(self, nFreqs=None, tMax=None):
    """Invalid"""
    pass

  def setDispersion(self, beta2, beta2s, beta1=0, beta1s=0, beta3=0, beta3s=0):
    """Invalid"""
    pass

  def setPump(self, pulseType=0, chirp=0, customPump=None):
    """Invalid"""
    pass

  def computeGreensFunction(s, inTimeDomain=False, runPump=True):
    __doc__ = _NonlinearMedium.computeGreensFunction.__doc__

    if runPump: s.runPumpSimulation()

    # Green function matrices
    greenC = np.eye(s._nFreqs, dtype=np.complex64)
    greenS = np.zeros((s._nFreqs, s._nFreqs), dtype=np.complex64)

    for medium in s.media:
      C, S = medium.computeGreensFunction(inTimeDomain=inTimeDomain, runPump=False)
      greenC = C @ greenC + S @ np.conj(greenS)
      greenS = C @ greenS + S @ np.conj(greenC)

    return greenC, greenS
