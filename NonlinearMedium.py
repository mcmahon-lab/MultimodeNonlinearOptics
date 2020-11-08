import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import threading as th

from inspect import getmembers, isfunction

def inherit_docstrings(skip=None):
  if not isinstance(skip, (type(None), list, tuple, str)): raise TypeError("What function are you trying to skip?")
  if skip and isinstance(skip, str): skip = [skip]

  def wrapper(cls):
    parents = [parent for parent in cls.__mro__[1:]]
    redundantParent = [False] * len(parents)

    for i in range(len(parents)):
      for j in range(i+1, len(parents)):
        if issubclass(parents[i], parents[j]):
          redundantParent[j] = True

    for name, func in getmembers(cls, isfunction):
      if skip and name in skip: continue

      docStrings = [func.__doc__] if func.__doc__ else []
      for i, parent in enumerate(parents):
        if redundantParent[i]: continue

        if hasattr(parent, name) and func is not getattr(parent, name) and getattr(parent, name).__doc__:
          docStrings.append(getattr(parent, name).__doc__)

      func.__doc__ = "".join(docStrings[::-1])
    return cls
  return wrapper

class _NonlinearMedium:
  """
  Base class for numerically simulating the evolution of a classical field in nonlinear media with a signal or quantum field.
  """
  def __init__(self, relativeLength, nlLength, dispLength, beta2, beta2s, pulseType=0,
               beta1=0, beta1s=0, beta3=0, beta3s=0, diffBeta0=0, chirp=0, rayleighLength=np.inf, tMax=10, tPrecision=512, zPrecision=100,
               customPump=None):
    """
    :param relativeLength: Length of fiber in dispersion lengths, or nonlinear lengths if no dispersion.
    :param nlLength:       Nonlinear length in terms of dispersion length.
                           If no dispersion, must be 1. For no nonlinear effects, set to np.inf.
    :param dispLength:     Dispersion length. Must be kept at one or set to np.inf to remove dispersion.
    :param beta2:          Group velocity dispersion of the pump. Must be +/- 1.
    :param beta2s:         Group velocity dispersion of the signal, relative to the pump.
    :param pulseType:      Pump profile, 0/False for Gaussian, 1/True for Hyperbolic Secant or 3 for Sinc.
    :param beta1:          Group velocity difference for pump relative to simulation window.
    :param beta1s:         Group velocity difference for signal relative to simulation window.
    :param beta3:          Pump third order dispersion.
    :param beta3s:         Signal third order dispersion.
    :param diffBeta0:      Wave-vector mismatch of the simulated process.
    :param chirp:          Initial chirp for pump pulse, in terms of propagation length.
    :param rayleighLength: Rayleigh length of propagation, assumes focused at medium's center.
    :param tMax:           Time window size in terms of pump width.
    :param tPrecision:     Number of time bins. Preferably power of 2 for better FFT performance.
    :param zPrecision:     Number of bins per unit length.
    :param customPump:     Specify a pump profile in time domain.
    """

    self._checkInput(relativeLength, nlLength, dispLength, beta2, beta2s, pulseType,
                     beta1, beta1s, beta3, beta3s, diffBeta0, chirp, tMax, tPrecision, zPrecision, rayleighLength,
                     customPump)
    self._setLengths(relativeLength, nlLength, dispLength, zPrecision, rayleighLength)
    self._resetGrids(tPrecision, tMax)
    self._setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s, diffBeta0)
    self.setPump(pulseType, chirp, customPump)


  def _checkInput(self, relativeLength, nlLength, dispLength, beta2, beta2s, pulseType,
                  beta1, beta1s, beta3, beta3s, diffBeta0, chirp, tMax, tPrecision, zPrecision, rayleighLength,
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
    if not isinstance(diffBeta0, (int, float)): raise TypeError("diffBeta0")
    if not isinstance(chirp,  (int, float)): raise TypeError("chirp")
    if not isinstance(rayleighLength,  (int, float)): raise TypeError("rayleighLength")
    if not isinstance(pulseType, (bool, int)): raise TypeError("pulseType")
    if not isinstance(tMax, int):       raise TypeError("tMax")
    if not isinstance(tPrecision, int): raise TypeError("tPrecision")
    if not isinstance(zPrecision, int): raise TypeError("zPrecision")
    if not isinstance(customPump, (type(None), np.ndarray)): raise TypeError("customPump")


  def _setLengths(self, relativeLength, nlLength, dispLength, zPrecision, rayleighLength):
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
    _Nsquared = self._DS / self._NL
    if self._noDispersion: _Nsquared = 1
    if self._noNonlinear:  _Nsquared = 0

    # space resolution
    self._nZSteps = int(zPrecision * self._z / np.min([1, self._DS, self._NL]))
    self._dz = self._z / self._nZSteps

    # helper values
    self._nlStep = 1j * _Nsquared * self._dz

    self._rayleighLength = rayleighLength


  def _resetGrids(self, nFreqs, tMax):

    # time windowing and resolution
    if nFreqs % 2 != 0 or nFreqs <= 0 or self._nZSteps <= 0 or tMax <= 0:
      raise ValueError("Invalid PDE grid size")

    self._nFreqs = nFreqs
    self._tMax = tMax

    Nt = self._nFreqs

    # time and frequency axes
    self.tau = (2 * self._tMax / Nt) * ifftshift(np.arange(-Nt // 2, Nt // 2, dtype=np.float64))
    self.omega = (-np.pi / self._tMax) * fftshift(np.arange(-Nt // 2, Nt // 2, dtype=np.float64))

    # Grids for PDE propagation
    self.pumpFreq = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex128)
    self.pumpTime = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex128)
    self.signalFreq = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex128)
    self.signalTime = np.zeros((self._nZSteps, self._nFreqs), dtype=np.complex128)


  def _setDispersion(self, beta2, beta2s, beta1=0, beta1s=0, beta3=0, beta3s=0, diffBeta0=0):

    # positive or negative dispersion for pump (ie should be +/- 1), relative dispersion for signal
    self._beta2  = beta2
    self._beta2s = beta2s

    # group velocity difference (relative to beta2 and pulse width)
    self._beta1  = beta1
    self._beta1s = beta1s

    # third order dispersion (relative to beta2 and pulse width, unless beta2 is zero, then should be +/- 1)
    self._beta3  = beta3
    self._beta3s = beta3s

    # signal phase mis-match
    self._diffBeta0 = diffBeta0

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


  def setPump(self, pulseType=0, chirpLength=0, customPump=None):
    # initial time domain envelopes (pick Gaussian, Hyperbolic Secant or custom, Sinc)
    if customPump is not None:
      if customPump.size != self._nFreqs:
        raise ValueError("Custom pump array length does not match number of frequency/time bins")
      self._env = customPump
    else:
      if pulseType == 1:
        self._env = 1 / np.cosh(self.tau)
      elif pulseType == 2:
        self._env = np.sin(self.tau) / self.tau
        self._env[np.isnan(self._env)] = 1
      else:
        self._env = np.exp(-0.5 * self.tau**2)

    if chirpLength != 0:
      self._env = ifft(fft(self._env) * np.exp(0.5j * self._beta2 * chirpLength * self.omega**2))


  def runPumpSimulation(s):
    """
    Simulate propagation of pump field
    """
    s.pumpFreq[0, :] = fft(s._env)
    s.pumpTime[0, :] = s._env

    for i in range(1, s._nZSteps):
      s.pumpFreq[i, :] = s.pumpFreq[0, :] * np.exp(1j * i * s._dispersionPump * s._dz)
      s.pumpTime[i, :] = ifft(s.pumpFreq[i, :])

    if s._rayleighLength != np.inf:
      relativeStrength = 1 / np.sqrt(1 + (np.linspace(-0.5 * s._z, 0.5 * s._z, s._nZSteps) / s._rayleighLength)**2)
      s.pumpFreq *= relativeStrength[:, np.newaxis]
      s.pumpTime *= relativeStrength[:, np.newaxis]


  def runSignalSimulation(s, inputProf, inTimeDomain=True):
    """
    Simulate propagation of signal field
    :param inputProf: Profile of input pulse. Can be time or frequency domain.
    Note: Frequency domain input is assumed to be "true" frequency with self.omega as its axis
    (since FFT considers the center frequency as the first and last elements).
    :param inTimeDomain: Specify if input is in frequency or frequency domain. True for time, false for frequency.
    """
    if inputProf.shape != (s._nFreqs,):
      raise ValueError("inputProf array size does not match number of frequency/time bins")
    s._runSignalSimulation(inputProf, inTimeDomain, s.signalFreq, s.signalTime)


  def _runSignalSimulation(s, inputProf, inTimeDomain, signalFreq, signalTime):
    pass


  def computeGreensFunction(s, inTimeDomain=False, runPump=True, nThreads=1):
    """
    Solve a(L) = C a(0) + S [a(0)]^t for C and S
    :param inTimeDomain Compute the Green's function in time or frequency domain.
    :param runPump      Whether to run pump simulation beforehand.
    :return: Green's functions C, S
    """
    if nThreads > s._nFreqs:
      raise ValueError("Too many threads requested!")

    if runPump: s.runPumpSimulation()

    # Green function matrices
    greenC = np.zeros((s._nFreqs, s._nFreqs), dtype=np.complex128)
    greenS = np.zeros((s._nFreqs, s._nFreqs), dtype=np.complex128)

    # Calculate Green's functions with real and imaginary impulse response
    def calcGreensPart(usingMemberGrids, start, stop):

      if usingMemberGrids:
        gridFreq, gridTime = s.signalFreq, s.signalTime
      else:
        gridFreq, gridTime = np.empty_like(s.signalFreq), np.empty_like(s.signalTime)

      grid = gridTime if inTimeDomain else gridFreq

      for i in range(start, stop):
        grid[0, :] = 0
        grid[0, i] = 1
        s._runSignalSimulation(grid[0], inTimeDomain, gridFreq, gridTime)

        greenC[i, :] += grid[-1, :] * 0.5
        greenS[i, :] += grid[-1, :] * 0.5

        grid[0, :] = 0
        grid[0, i] = 1j
        s._runSignalSimulation(grid[0], inTimeDomain, gridFreq, gridTime)

        greenC[i, :] -= grid[-1, :] * 0.5j
        greenS[i, :] += grid[-1, :] * 0.5j

    # run n-1 separate threads, run part on this process
    threads = [th.Thread(target=calcGreensPart, args=(False, (i * s._nFreqs) // nThreads, ((i + 1) * s._nFreqs) // nThreads))
               for i in range(1, nThreads)]
    for thread in threads: thread.start()
    calcGreensPart(True, 0, s._nFreqs // nThreads)
    for thread in threads: thread.join()

    greenC = fftshift(greenC.T)
    greenS = fftshift(greenS.T)

    return greenC, greenS


  def batchSignalSimulation(s, inputProfs, inTimeDomain=False, runPump=True, nThreads=1):
    """
    Run multiple signal simulations.
    :param inputProfs   Profiles of input pulses. Can be time or frequency domain.
    :param inTimeDomain Specify if input is in frequency or frequency domain. True for time, false for frequency.
    :param runPump      Whether to run pump simulation beforehand.
    :return: Signal profiles at the output of the medium
    """
    nInputs, inCols = inputProfs.shape
    # TODO For SFG accepts single or double input but returns only one, need to generalize or expand
    if inCols % s._nFreqs != 0 or inCols / s._nFreqs == 0 or inCols / s._nFreqs > 2:
      raise ValueError("Signals not of correct length!")

    if nThreads > nInputs:
      raise ValueError("Too many threads requested!")

    if runPump: s.runPumpSimulation()

    # Signal outputs
    outSignals = np.empty((nInputs, s._nFreqs), dtype=np.complex128)

    # Calculate Green's functions with real and imaginary impulse response
    def calcBatch(usingMemberGrids, start, stop):

      if usingMemberGrids:
        gridFreq, gridTime = s.signalFreq, s.signalTime
      else:
        gridFreq, gridTime = np.empty_like(s.signalFreq), np.empty_like(s.signalTime)

      grid = gridTime if inTimeDomain else gridFreq

      for i in range(start, stop):
        s._runSignalSimulation(inputProfs[i], inTimeDomain, gridFreq, gridTime)
        outSignals[i, :] = grid[-1, :]

    # run n-1 separate threads, run part on this process
    threads = [th.Thread(target=calcBatch, args=(False, (i * nInputs) // nThreads, ((i + 1) * nInputs) // nThreads))
               for i in range(1, nThreads)]
    for thread in threads: thread.start()
    calcBatch(True, 0, nInputs // nThreads)
    for thread in threads: thread.join()

    return outSignals


class _NLM2ModeExtension:
  """
  Extension class for extending _NonlinearMedium to allow simulation of processes involving two different modes.
  """
  def __init__(self, medium, nlLengthOrig, beta2o, beta1o, beta3o):
    """
    :param nlLengthOrig:   Like nlLength but with respect to the original signal.
    :param beta2o:         Second order dispersion of the original signal's frequency
    :param beta1o:         Group velocity difference for original signal relative to simulation window.
    :param beta3o:         Original signal third order dispersion.
    """
    # Store a reference of the actual _NonlinearMedium object, to access variables and methods
    self._checkInputs(medium, nlLengthOrig, beta2o, beta1o, beta3o)
    self.m = medium
    self.setLengths(nlLengthOrig)
    self.resetGrids()
    self.setDispersion(beta2o, beta1o, beta3o)


  def _checkInputs(self, medium, nlLengthOrig, beta2o, beta1o, beta3o):

    if not isinstance(nlLengthOrig, (int, float)): raise TypeError("nlLengthOrig")
    if not isinstance(beta2o,  (int, float)): raise TypeError("beta2o")
    if not isinstance(beta1o,  (int, float)): raise TypeError("beta1o")
    if not isinstance(beta3o,  (int, float)): raise TypeError("beta3o")
    if not issubclass(type(medium), _NonlinearMedium): raise TypeError("medium")


  def setLengths(self, nlLengthOrig):
    # self._NLo = nlLengthOrig
    if self.m._noDispersion:
      self._nlStepO = 1j * self.m._NL / nlLengthOrig * self.m._dz
    elif self.m._noNonlinear:
      self._nlStepO = 0
    else:
      self._nlStepO = 1j * self.m._DS / nlLengthOrig * self.m._dz


  def resetGrids(self):
    self.originalFreq = np.zeros((self.m._nZSteps, self.m._nFreqs), dtype=np.complex128)
    self.originalTime = np.zeros((self.m._nZSteps, self.m._nFreqs), dtype=np.complex128)


  def setDispersion(self, beta2o, beta1o, beta3o):
    self._beta2o = beta2o
    self._beta1o = beta1o
    self._beta3o = beta3o

    if self.m._noDispersion:
      self._dispersionOrig.setZero(self.m._nFreqs)
    else:
      self._dispersionOrig = self.m.omega * (beta1o + self.m.omega * (0.5 * beta2o + self.m.omega * beta3o / 6))

    self._dispStepOrig = np.exp(1j * self._dispersionOrig * self.m._dz)


  def runSignalSimulation(self, inputProf, inTimeDomain=True):
    """
    Simulate propagation of signal field
    :param inputProf: Profile of input pulse. Can be time or frequency domain.
    Note: Frequency domain input is assumed to be "true" frequency with self.omega as its axis
    (since FFT considers the center frequency as the first and last elements).
    :param inTimeDomain: Specify if input is in frequency or frequency domain. True for time, false for frequency.
    """
    if inputProf.shape != (self.m._nFreqs,) and inputProf.shape != (2 * self.m._nFreqs,):
      raise ValueError("inputProf array size does not match number of frequency/time bins")
    self.m._runSignalSimulation(inputProf, inTimeDomain, self.m.signalFreq, self.m.signalTime)


  def computeTotalGreen(s, inTimeDomain=False, runPump=True, nThreads=1):
    """
    Solve a(L) = C a(0) + S [a(0)]^t for C and S for the combined system of the generated and original signals
    :param inTimeDomain Compute the Green's function in time or frequency domain.
    :param runPump      Whether to run pump simulation beforehand.
    :return: Green's functions C, S for the spectrum including both generated and original signals
    """
    _nFreqs = s.m._nFreqs
    _nZSteps = s.m._nZSteps

    if nThreads > 2 * _nFreqs:
      raise ValueError("Too many threads requested!")

    if runPump: s.m.runPumpSimulation()

    # Green function matrices
    greenC = np.zeros((2 * _nFreqs, 2 * _nFreqs), dtype=np.complex128)
    greenS = np.zeros((2 * _nFreqs, 2 * _nFreqs), dtype=np.complex128)

    # Calculate Green's functions with real and imaginary impulse response
    def calcGreensPart(usingMemberGrids, start, stop):

      if usingMemberGrids:
        gridFreq, gridTime = s.m.signalFreq, s.m.signalTime
      else:
        gridFreq = np.empty((2 * _nZSteps, _nFreqs), dtype=np.complex128)
        gridTime = np.empty((2 * _nZSteps, _nFreqs), dtype=np.complex128)

      if usingMemberGrids:
        outputOriginal, outputSignal = (s.originalTime[-1], s.m.signalTime[-1]) if inTimeDomain \
          else (s.originalFreq[-1], s.m.signalFreq[-1])
      else:
        outputOriginal, outputSignal = (gridTime[_nZSteps-1], gridTime[-1]) if inTimeDomain \
          else (gridFreq[_nZSteps-1], gridFreq[-1])

      impulse = np.zeros(2 * _nFreqs, dtype=np.complex128)

      for i in range(start, stop):
        impulse[i] = 1
        s.m._runSignalSimulation(impulse, inTimeDomain, gridFreq, gridTime)

        greenC[i, :_nFreqs] += outputSignal   * 0.5
        greenC[i, _nFreqs:] += outputOriginal * 0.5
        greenS[i, :_nFreqs] += outputSignal   * 0.5
        greenS[i, _nFreqs:] += outputOriginal * 0.5

        impulse[i] = 1j
        s.m._runSignalSimulation(impulse, inTimeDomain, gridFreq, gridTime)

        greenC[i, :_nFreqs] -= outputSignal   * 0.5j
        greenC[i, _nFreqs:] -= outputOriginal * 0.5j
        greenS[i, :_nFreqs] += outputSignal   * 0.5j
        greenS[i, _nFreqs:] += outputOriginal * 0.5j

        impulse[i] = 0

    # run n-1 separate threads, run part on this process
    threads = [th.Thread(target=calcGreensPart, args=(False, (2 * i * _nFreqs) // nThreads, ((i + 1) * 2 * _nFreqs) // nThreads))
               for i in range(1, nThreads)]
    for thread in threads: thread.start()
    calcGreensPart(True, 0, (2 * _nFreqs) // nThreads)
    for thread in threads: thread.join()

    # Transpose, then need to fftshift each frequency block
    greenC = greenC.T
    greenC[_nFreqs:, _nFreqs:] = fftshift(greenC[_nFreqs:, _nFreqs:])
    greenC[_nFreqs:, :_nFreqs] = fftshift(greenC[_nFreqs:, :_nFreqs])
    greenC[:_nFreqs, _nFreqs:] = fftshift(greenC[:_nFreqs, _nFreqs:])
    greenC[:_nFreqs, :_nFreqs] = fftshift(greenC[:_nFreqs, :_nFreqs])

    greenS = greenS.T
    greenS[_nFreqs:, _nFreqs:] = fftshift(greenS[_nFreqs:, _nFreqs:])
    greenS[_nFreqs:, :_nFreqs] = fftshift(greenS[_nFreqs:, :_nFreqs])
    greenS[:_nFreqs, _nFreqs:] = fftshift(greenS[:_nFreqs, _nFreqs:])
    greenS[:_nFreqs, :_nFreqs] = fftshift(greenS[:_nFreqs, :_nFreqs])

    return greenC, greenS


@inherit_docstrings()
class Chi3(_NonlinearMedium):
  """
  Class for numerically simulating the evolution of a pump and quantum field undergoing self
  phase modulation in a chi(3) medium.
  """
  def __init__(self, relativeLength, nlLength, dispLength, beta2, pulseType=0,
               beta3=0, chirp=0, rayleighLength=np.inf, tMax=10, tPrecision=512, zPrecision=100, customPump=None):
    # same as base class except pump and signal dispersion must be identical, and no zero or first order dispersion
    super().__init__(relativeLength, nlLength, dispLength, beta2, beta2, pulseType,
                     0, 0, beta3, beta3, 0, chirp, rayleighLength, tMax, tPrecision, zPrecision,
                     customPump)


  def runPumpSimulation(s):

    s.pumpFreq[0, :] = fft(s._env) * np.exp(0.5j * s._dispersionPump * s._dz)
    s.pumpTime[0, :] = ifft(s.pumpFreq[0, :])

    for i in range(1, s._nZSteps):
      temp = s.pumpTime[i-1, :] * np.exp(s._nlStep * np.abs(s.pumpTime[i-1, :])**2)
      s.pumpFreq[i, :] = fft(temp) * s._dispStepPump
      s.pumpTime[i, :] = ifft(s.pumpFreq[i, :])

    s.pumpFreq[-1, :] *= np.exp(-0.5j * s._dispersionPump * s._dz)
    s.pumpTime[-1, :] = ifft(s.pumpFreq[-1, :])

    if s._rayleighLength != np.inf:
      relativeStrength = 1 / np.sqrt(1 + (np.linspace(-0.5 * s._z, 0.5 * s._z, s._nZSteps) / s._rayleighLength)**2)
      s.pumpFreq *= relativeStrength[:, np.newaxis]
      s.pumpTime *= relativeStrength[:, np.newaxis]


  def _runSignalSimulation(s, inputProf, inTimeDomain, signalFreq, signalTime):

    inputProfFreq = (fft(inputProf) if inTimeDomain else inputProf)

    signalFreq[0, :] = inputProfFreq * np.exp(0.5j * s._dispersionSign * s._dz)
    signalTime[0, :] = ifft(signalFreq[0, :])

    for i in range(1, s._nZSteps):
      # Do a Runge-Kutta step for the non-linear propagation
      pumpTimeInterp = 0.5 * (s.pumpTime[i-1] + s.pumpTime[i])

      prevConj = np.conj(signalTime[i-1, :])
      k1 = s._nlStep * (2 * np.abs(s.pumpTime[i-1])**2 *  signalTime[i-1, :]             + s.pumpTime[i-1]**2 *  prevConj)
      k2 = s._nlStep * (2 * np.abs(pumpTimeInterp)**2  * (signalTime[i-1, :] + 0.5 * k1) + pumpTimeInterp**2  * (prevConj + np.conj(0.5 * k1)))
      k3 = s._nlStep * (2 * np.abs(pumpTimeInterp)**2  * (signalTime[i-1, :] + 0.5 * k2) + pumpTimeInterp**2  * (prevConj + np.conj(0.5 * k2)))
      k4 = s._nlStep * (2 * np.abs(s.pumpTime[i])**2   * (signalTime[i-1, :] + k3)       + s.pumpTime[i]**2   * (prevConj + np.conj(k3)))

      temp = signalTime[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

      # Dispersion step
      signalFreq[i, :] = fft(temp) * s._dispStepSign
      signalTime[i, :] = ifft(signalFreq[i, :])

    signalFreq[-1, :] *= np.exp(-0.5j * s._dispersionSign * s._dz)
    signalTime[-1, :] = ifft(signalFreq[-1, :])


@inherit_docstrings()
class _Chi2(_NonlinearMedium):
  """
  Base class for numerically simulating the evolution of a classical field in a chi(2) medium with a signal or quantum field.
  """
  def __init__(self, relativeLength, nlLength, dispLength, beta2, beta2s, pulseType=0,
               beta1=0, beta1s=0, beta3=0, beta3s=0, diffBeta0=0, chirp=0, rayleighLength=np.inf, tMax=10,
               tPrecision=512, zPrecision=100, customPump=None, poling=None):
    """
    :param poling:         Poling profile to simulate, specifying relative domain lengths.
    """
    super().__init__(relativeLength, nlLength, dispLength, beta2, beta2s, pulseType,
                     beta1, beta1s, beta3, beta3s, diffBeta0, chirp, rayleighLength, tMax, tPrecision, zPrecision,
                     customPump)

    self._setPoling(poling)


  def _setPoling(self, poling):
    if not isinstance(poling, (type(None), np.ndarray, list)): raise TypeError("poling")

    if poling is None or len(poling) <= 1:
      self.poling = np.ones(self._nZSteps)
    else:
      poleDomains = np.cumsum(poling, dtype=np.float64)
      poleDomains *= self._nZSteps / poleDomains[-1]

      self.poling = np.empty(self._nZSteps)
      prevInd = 0
      direction = 1
      for currInd in poleDomains:
        if currInd < prevInd: raise ValueError("Poling period too small for simulation resolution")
        self.poling[prevInd:int(currInd)] = direction

        if int(currInd) < self._nZSteps: # interpolate indices on the boundary
          self.poling[int(currInd)] = direction * (2 * abs(currInd % 1) - 1)

        direction *= -1
        prevInd = int(currInd) + 1


@inherit_docstrings()
class Chi2PDC(_Chi2):
  """
  Class for numerically simulating the evolution of a parametric down-conversion or DOPA process of a pump and signal
  or quantum field in a chi(2) medium.
  """
  def _runSignalSimulation(s, inputProf, inTimeDomain, signalFreq, signalTime):

    inputProfFreq = (fft(inputProf) if inTimeDomain else inputProf)

    signalFreq[0, :] = inputProfFreq * np.exp(0.5j * s._dispersionSign * s._dz)
    signalTime[0, :] = ifft(signalFreq[0, :])

    for i in range(1, s._nZSteps):
      # Do a Runge-Kutta step for the non-linear propagation
      pumpTimeInterp = 0.5 * (s.pumpTime[i-1] + s.pumpTime[i])

      prevPolDir = s.poling[i-1]
      currPolDir = s.poling[i]
      intmPolDir = 0.5 * (prevPolDir + currPolDir)

      prevMismatch = np.exp(1j * s._diffBeta0 * ((i- 1) * s._dz))
      intmMismatch = np.exp(1j * s._diffBeta0 * ((i-.5) * s._dz))
      currMismatch = np.exp(1j * s._diffBeta0 * ( i     * s._dz))

      k1 = (prevPolDir * s._nlStep * prevMismatch) * s.pumpTime[i-1] * np.conj(signalTime[i-1])
      k2 = (intmPolDir * s._nlStep * intmMismatch) * pumpTimeInterp  * np.conj(signalTime[i-1] + 0.5 * k1)
      k3 = (intmPolDir * s._nlStep * intmMismatch) * pumpTimeInterp  * np.conj(signalTime[i-1] + 0.5 * k2)
      k4 = (currPolDir * s._nlStep * currMismatch) * s.pumpTime[i]   * np.conj(signalTime[i-1] + k3)

      temp = signalTime[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

      # Dispersion step
      signalFreq[i, :] = fft(temp) * s._dispStepSign
      signalTime[i, :] = ifft(signalFreq[i, :])

    signalFreq[-1, :] *= np.exp(-0.5j * s._dispersionSign * s._dz)
    signalTime[-1, :] = ifft(signalFreq[-1, :])


@inherit_docstrings()
class Chi2SHG(_Chi2):
  """
  Class for numerically simulating the process of second harmonic generation in the pump undepleted approximation in a chi(2) medium.
  """
  def _runSignalSimulation(s, inputProf, inTimeDomain, signalFreq, signalTime):

    inputProfFreq = (fft(inputProf) if inTimeDomain else inputProf)

    signalFreq[0, :] = inputProfFreq * np.exp(0.5j * s._dispersionSign * s._dz)
    signalTime[0, :] = ifft(signalFreq[0, :])

    for i in range(1, s._nZSteps):
      # Do a Runge-Kutta step for the non-linear propagation
      pumpTimeInterp = 0.5 * (s.pumpTime[i-1] + s.pumpTime[i])

      prevPolDir = s.poling[i-1]
      currPolDir = s.poling[i]
      intmPolDir = 0.5 * (prevPolDir + currPolDir)

      prevMismatch = np.exp(1j * s._diffBeta0 * ((i- 1) * s._dz))
      intmMismatch = np.exp(1j * s._diffBeta0 * ((i-.5) * s._dz))
      currMismatch = np.exp(1j * s._diffBeta0 * ( i     * s._dz))

      k1 = (prevPolDir * s._nlStep * prevMismatch) * s.pumpTime[i-1]**2
      k2 = (intmPolDir * s._nlStep * intmMismatch) * pumpTimeInterp**2
      k3 = (intmPolDir * s._nlStep * intmMismatch) * pumpTimeInterp**2
      k4 = (currPolDir * s._nlStep * currMismatch) * s.pumpTime[i]**2

      temp = signalTime[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

      # Dispersion step
      signalFreq[i, :] = fft(temp) * s._dispStepSign
      signalTime[i, :] = ifft(signalFreq[i, :])

    signalFreq[-1, :] *= np.exp(-0.5j * s._dispersionSign * s._dz)
    signalTime[-1, :] = ifft(signalFreq[-1, :])


@inherit_docstrings()
class Chi2SFG(_NLM2ModeExtension, _Chi2):
  """
  Class for numerically simulating the evolution of a sum frequency generation process of a pump and two signals
  or quantum fields in a chi(2) medium.
  """
  def __init__(self, relativeLength, nlLength, nlLengthOrig, dispLength, beta2, beta2s, beta2o, pulseType=0,
               beta1=0, beta1s=0, beta1o=0, beta3=0, beta3s=0, beta3o=0, diffBeta0=0, diffBeta0o=0, chirp=0,
               rayleighLength=np.inf, tMax=10, tPrecision=512, zPrecision=100, customPump=None, poling=None):
    """
    :param diffBeta0o:     Wave-vector mismatch of PDC process with the original signal and pump.
    """
    _Chi2.__init__(self, relativeLength, nlLength, dispLength, beta2, beta2s, pulseType,
                   beta1, beta1s, beta3, beta3s, diffBeta0, chirp, rayleighLength, tMax,
                   tPrecision, zPrecision, customPump, poling)

    _NLM2ModeExtension.__init__(self, self, nlLengthOrig, beta2o, beta1o, beta3o)

    if not isinstance(diffBeta0o, (int, float)): raise TypeError("diffBeta0o")
    self._diffBeta0o = diffBeta0o


  def _runSignalSimulation(s, inputProf, inTimeDomain, signalFreq, signalTime):

    # Hack: If we are using grids that are the member variables of the class, then proceed normally.
    # However if called from computeGreensFunction we need a workaround to use only one grid.
    usingMemberGrids = (signalFreq is s.signalFreq)
    O = 0 if usingMemberGrids else s._nZSteps # offset
    if not usingMemberGrids:
      if signalFreq.shape != (2 * s._nZSteps, s._nFreqs): signalFreq.resize((2 * s._nZSteps, s._nFreqs), refcheck=False)
      if signalTime.shape != (2 * s._nZSteps, s._nFreqs): signalTime.resize((2 * s._nZSteps, s._nFreqs), refcheck=False)
    originalFreq = s.originalFreq if usingMemberGrids else signalFreq[:s._nZSteps, :]
    originalTime = s.originalTime if usingMemberGrids else signalTime[:s._nZSteps, :]

    if inputProf.size == s._nFreqs:
      # Takes as input the signal in the first frequency and outputs in the second frequency
      inputProfFreq = (fft(inputProf) if inTimeDomain else inputProf)

      originalFreq[0, :] = inputProfFreq * np.exp(0.5j * s._dispersionOrig * s._dz)
      originalTime[0, :] = ifft(originalFreq[0, :])

      signalFreq[O, :] = 0
      signalTime[O, :] = 0

    elif inputProf.size == 2 * s._nFreqs:
      # input array spanning both frequencies, ordered as signal then original
      if inTimeDomain:
        signalFreq[O, :]   = fft(inputProf[:s._nFreqs]) * np.exp(0.5j * s._dispersionSign * s._dz)
        originalFreq[0, :] = fft(inputProf[s._nFreqs:]) * np.exp(0.5j * s._dispersionOrig * s._dz)

      else:
        signalFreq[O, :]   = inputProf[:s._nFreqs] * np.exp(0.5j * s._dispersionSign * s._dz)
        originalFreq[0, :] = inputProf[s._nFreqs:] * np.exp(0.5j * s._dispersionOrig * s._dz)

      signalTime[O, :]   = ifft(signalFreq[O, :])
      originalTime[0, :] = ifft(originalFreq[0, :])

    else:
      raise ValueError("inputProf array size does not match number of frequency/time bins")

    for i in range(1, s._nZSteps):
      # Do a Runge-Kutta step for the non-linear propagation
      pumpTimeInterp = 0.5 * (s.pumpTime[i-1] + s.pumpTime[i])

      conjPumpInterpTime = np.conj(pumpTimeInterp)

      prevPolDir = s.poling[i-1]
      currPolDir = s.poling[i]
      intmPolDir = 0.5 * (prevPolDir + currPolDir)

      prevMismatch = np.exp(1j * s._diffBeta0 * (i- 1) * s._dz)
      intmMismatch = np.exp(1j * s._diffBeta0 * (i-.5) * s._dz)
      currMismatch = np.exp(1j * s._diffBeta0 *  i     * s._dz)
      prevInvMsmch = 1 / prevMismatch
      intmInvMsmch = 1 / intmMismatch
      currInvMsmch = 1 / currMismatch
      prevMismatcho = np.exp(1j * s._diffBeta0o * (i- 1) * s._dz)
      intmMismatcho = np.exp(1j * s._diffBeta0o * (i-.5) * s._dz)
      currMismatcho = np.exp(1j * s._diffBeta0o *  i     * s._dz)

      k1 = (prevPolDir * s._nlStepO) * (prevInvMsmch  * np.conj(s.pumpTime[i-1]) *  signalTime[O+i-1]             + prevMismatcho * s.pumpTime[i-1] * np.conj(originalTime[i-1]))
      l1 = (prevPolDir * s._nlStep   *  prevMismatch) * s.pumpTime[i-1]          *  originalTime[i-1]
      k2 = (intmPolDir * s._nlStepO) * (intmInvMsmch  * conjPumpInterpTime       * (signalTime[O+i-1] + 0.5 * l1) + intmMismatcho * pumpTimeInterp  * np.conj(originalTime[i-1] + 0.5 * k1))
      l2 = (intmPolDir * s._nlStep   *  intmMismatch) * pumpTimeInterp           * (originalTime[i-1] + 0.5 * k1)
      k3 = (intmPolDir * s._nlStepO) * (intmInvMsmch  * conjPumpInterpTime       * (signalTime[O+i-1] + 0.5 * l2) + intmMismatcho * pumpTimeInterp  * np.conj(originalTime[i-1] + 0.5 * k2))
      l3 = (intmPolDir * s._nlStep   *  intmMismatch) * pumpTimeInterp           * (originalTime[i-1] + 0.5 * k2)
      k4 = (currPolDir * s._nlStepO) * (currInvMsmch  * np.conj(s.pumpTime[i])   * (signalTime[O+i-1] + l3)       + currMismatcho * s.pumpTime[i]   * np.conj(originalTime[i-1] + k3))
      l4 = (currPolDir * s._nlStep   *  currMismatch) * s.pumpTime[i]            * (originalTime[i-1] + k3)

      tempOrig = originalTime[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
      tempSign = signalTime[O+i-1] + (l1 + 2 * l2 + 2 * l3 + l4) / 6

      # Dispersion step
      signalFreq[O+i, :] = fft(tempSign) * s._dispStepSign
      signalTime[O+i, :] = ifft(signalFreq[O+i, :])

      originalFreq[i, :] = fft(tempOrig) * s._dispStepOrig
      originalTime[i, :] = ifft(originalFreq[i, :])

    signalFreq[-1, :] *= np.exp(-0.5j * s._dispersionSign * s._dz)
    signalTime[-1, :] = ifft(signalFreq[-1, :])

    originalFreq[-1, :] *= np.exp(-0.5j * s._dispersionOrig * s._dz)
    originalTime[-1, :] = ifft(originalFreq[-1, :])


@inherit_docstrings()
class Chi2PDCII(_NLM2ModeExtension, _Chi2):
  """
  Class for numerically simulating the evolution of a Type II process of a pump and two signals
  or quantum fields of opposite polarization in a chi(2) medium.
  """
  def __init__(self, relativeLength, nlLength, nlLengthOrig, nlLengthI, dispLength, beta2, beta2s, beta2o, pulseType=0,
               beta1=0, beta1s=0, beta1o=0, beta3=0, beta3s=0, beta3o=0, diffBeta0=0, diffBeta0o=0, chirp=0,
               rayleighLength=np.inf, tMax=10, tPrecision=512, zPrecision=100, customPump=None, poling=None):
    """
    :param diffBeta0o:     Wave-vector mismatch of PDC process with the original signal and pump.
    :param nlLengthI:      Strength of type I nonlinear process over length dz; DOPA process of original signal
    """
    _Chi2.__init__(self, relativeLength, nlLength, dispLength, beta2, beta2s, pulseType,
                   beta1, beta1s, beta3, beta3s, diffBeta0, chirp, rayleighLength, tMax,
                   tPrecision, zPrecision, customPump, poling)

    _NLM2ModeExtension.__init__(self, self, nlLengthOrig, beta2o, beta1o, beta3o)

    if not isinstance(diffBeta0o, (int, float)): raise TypeError("diffBeta0o")
    if not isinstance(nlLengthI,  (int, float)): raise TypeError("nlLengthI")

    self._diffBeta0o = diffBeta0o

    if self._noDispersion:
      self._nlStepI = 1j * self._NL / nlLengthI * self._dz
    elif self._noNonlinear:
      self._nlStepI = 0
    else:
      self._nlStepI = 1j * self._DS / nlLengthI * self._dz


  def _runSignalSimulation(s, inputProf, inTimeDomain, signalFreq, signalTime):

    # Hack: If we are using grids that are the member variables of the class, then proceed normally.
    # However if called from computeGreensFunction we need a workaround to use only one grid.
    usingMemberGrids = (signalFreq is s.signalFreq)
    O = 0 if usingMemberGrids else s._nZSteps # offset
    if not usingMemberGrids:
      if signalFreq.shape != (2 * s._nZSteps, s._nFreqs): signalFreq.resize((2 * s._nZSteps, s._nFreqs), refcheck=False)
      if signalTime.shape != (2 * s._nZSteps, s._nFreqs): signalTime.resize((2 * s._nZSteps, s._nFreqs), refcheck=False)
    originalFreq = s.originalFreq if usingMemberGrids else signalFreq[:s._nZSteps, :]
    originalTime = s.originalTime if usingMemberGrids else signalTime[:s._nZSteps, :]

    if inputProf.size == s._nFreqs:
      # Takes as input the signal in the first frequency and outputs in the second frequency
      inputProfFreq = (fft(inputProf) if inTimeDomain else inputProf)

      originalFreq[0, :] = inputProfFreq * np.exp(0.5j * s._dispersionOrig * s._dz)
      originalTime[0, :] = ifft(originalFreq[0, :])

      signalFreq[O, :] = 0
      signalTime[O, :] = 0

    elif inputProf.size == 2 * s._nFreqs:
      # input array spanning both frequencies, ordered as signal then original
      if inTimeDomain:
        signalFreq[O, :]   = fft(inputProf[:s._nFreqs]) * np.exp(0.5j * s._dispersionSign * s._dz)
        originalFreq[0, :] = fft(inputProf[s._nFreqs:]) * np.exp(0.5j * s._dispersionOrig * s._dz)

      else:
        signalFreq[O, :]   = inputProf[:s._nFreqs] * np.exp(0.5j * s._dispersionSign * s._dz)
        originalFreq[0, :] = inputProf[s._nFreqs:] * np.exp(0.5j * s._dispersionOrig * s._dz)

      signalTime[O, :]   = ifft(signalFreq[O, :])
      originalTime[0, :] = ifft(originalFreq[0, :])

    else:
      raise ValueError("inputProf array size does not match number of frequency/time bins")

    for i in range(1, s._nZSteps):
      # Do a Runge-Kutta step for the non-linear propagation
      pumpTimeInterp = 0.5 * (s.pumpTime[i-1] + s.pumpTime[i])

      prevPolDir = s.poling[i-1]
      currPolDir = s.poling[i]
      intmPolDir = 0.5 * (prevPolDir + currPolDir)

      prevMismatch = np.exp(1j * s._diffBeta0 * (i- 1) * s._dz)
      intmMismatch = np.exp(1j * s._diffBeta0 * (i-.5) * s._dz)
      currMismatch = np.exp(1j * s._diffBeta0 *  i     * s._dz)
      prevMismatcho = np.exp(1j * s._diffBeta0o * (i- 1) * s._dz)
      intmMismatcho = np.exp(1j * s._diffBeta0o * (i-.5) * s._dz)
      currMismatcho = np.exp(1j * s._diffBeta0o *  i     * s._dz)

      k1 =  prevPolDir * ((s._nlStepO * prevMismatch)  * s.pumpTime[i-1] * np.conj(signalTime[O+i-1])            + (s._nlStepI * prevMismatcho) * s.pumpTime[i-1] * np.conj(originalTime[i-1]))
      l1 = (prevPolDir *   s._nlStep  *  prevMismatch) * s.pumpTime[i-1] * np.conj(originalTime[i-1])
      k2 =  intmPolDir * ((s._nlStepO * intmMismatch)  * pumpTimeInterp  * np.conj(signalTime[O+i-1] + 0.5 * l1) + (s._nlStepI * intmMismatcho) * pumpTimeInterp  * np.conj(originalTime[i-1] + 0.5 * k1))
      l2 = (intmPolDir *   s._nlStep  *  intmMismatch) * pumpTimeInterp  * np.conj(originalTime[i-1] + 0.5 * k1)
      k3 =  intmPolDir * ((s._nlStepO * intmMismatch)  * pumpTimeInterp  * np.conj(signalTime[O+i-1] + 0.5 * l2) + (s._nlStepI * intmMismatcho) * pumpTimeInterp  * np.conj(originalTime[i-1] + 0.5 * k2))
      l3 = (intmPolDir *   s._nlStep  *  intmMismatch) * pumpTimeInterp  * np.conj(originalTime[i-1] + 0.5 * k2)
      k4 =  currPolDir * ((s._nlStepO * currMismatch)  * s.pumpTime[i]   * np.conj(signalTime[O+i-1] + l3)       + (s._nlStepI * currMismatcho) * s.pumpTime[i]   * np.conj(originalTime[i-1] + k3))
      l4 = (currPolDir *   s._nlStep  *  currMismatch) * s.pumpTime[i]   * np.conj(originalTime[i-1] + k3)

      tempOrig = originalTime[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
      tempSign = signalTime[O+i-1] + (l1 + 2 * l2 + 2 * l3 + l4) / 6

      # Dispersion step
      signalFreq[O+i, :] = fft(tempSign) * s._dispStepSign
      signalTime[O+i, :] = ifft(signalFreq[O+i, :])

      originalFreq[i, :] = fft(tempOrig) * s._dispStepOrig
      originalTime[i, :] = ifft(originalFreq[i, :])

    signalFreq[-1, :] *= np.exp(-0.5j * s._dispersionSign * s._dz)
    signalTime[-1, :] = ifft(signalFreq[-1, :])

    originalFreq[-1, :] *= np.exp(-0.5j * s._dispersionOrig * s._dz)
    originalTime[-1, :] = ifft(originalFreq[-1, :])


@inherit_docstrings(skip="__init__")
class Cascade(_NonlinearMedium):
  """
  Class that cascades multiple media together.
  """
  def __init__(self, sharedPump, media):
    """
    :param sharedPump: Is the pump shared across media or are they independently pumped.
    :param media:      Collection of nonlinear media objects.
    """
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
    # self._dz = self._nlStep = 0
    # self.pumpFreq = self.pumpTime = self.signalFreq = self.signalTime = None
    # self._beta2 = self._beta2s = self._beta1 = self._beta1s = self._beta3 = self._beta3s = 0
    # self._dispersionPump = self._dispersionSign = self._dispStepPump = self._dispStepSign = self._env = None


  def addMedium(self, medium):
    """
    Append medium
    :param medium: Additional medium to append to cascaded process.
    """
    if not isinstance(medium, _NonlinearMedium):
      raise TypeError("Argument is not a NonlinearMedium object")

    if medium._nFreqs != self._nFreqs or medium._tMax != self._tMax:
      ValueError("Medium does not have same time and frequency axes as the first")

    self.media.append(medium)
    self.nMedia += 1
    self._nZSteps += medium._nZSteps


  def runPumpSimulation(self):

    if not self.sharedPump:
      for medium in self.media:
        medium.runPumpSimulation()

    else:
      self.media[0].runPumpSimulation()
      for i in range(1, len(self.media)):
        self.media[i]._env = self.media[i-1].pumpTime[-1]
        self.media[i].runPumpSimulation()


  def runSignalSimulation(self, inputProf, inTimeDomain=True):

    if inputProf.size != self._nFreqs:
      raise ValueError("inputProf array size does not match number of frequency/time bins")

    self.media[0].runSignalSimulation(inputProf, inTimeDomain)
    for i in range(1, len(self.media)):
      self.media[i].runSignalSimulation(self.media[i-1].signalFreq[-1], inTimeDomain=False)


  def _setLengths(self, relativeLength, nlLength, dispLength, zPrecision, rayleighLength):
    """Invalid"""
    pass

  def _resetGrids(self, nFreqs=None, tMax=None):
    """Invalid"""
    pass

  def _setDispersion(self, beta2, beta2s, beta1=0, beta1s=0, beta3=0, beta3s=0, diffBeta0=0):
    """Invalid"""
    pass


  def setPump(self, pulseType=0, chirp=0, customPump=None):

    if self.sharedPump:
      self.media[0].setPump(pulseType, chirp, customPump)
    else:
      for medium in self.media:
        medium.setPump(pulseType, chirp, customPump)


  def computeGreensFunction(s, inTimeDomain=False, runPump=True, nThreads=1):
    # TODO for large cascades or short media: option for directly feeding signals to avoid many matrix multiplications

    if runPump: s.runPumpSimulation()

    # Green function matrices
    greenC = np.eye(s._nFreqs, dtype=np.complex128)
    greenS = np.zeros((s._nFreqs, s._nFreqs), dtype=np.complex128)

    for medium in s.media:
      newC, newS = medium.computeGreensFunction(inTimeDomain=inTimeDomain, runPump=runPump, nThreads=nThreads)
      greenC, greenS = newC @ greenC + newS @ np.conj(greenS),\
                       newC @ greenS + newS @ np.conj(greenC)

    return greenC, greenS

  def batchSignalSimulation(self, inputProfs, inTimeDomain=False, runPump=True, nThreads=1):

    if runPump: self.runPumpSimulation()

    outSignals = self.media[0].batchSignalSimulation(inputProfs, inTimeDomain, False, nThreads)

    for medium in self.media[1:]:
      outSignals = medium.batchSignalSimulation(outSignals, inTimeDomain, False, nThreads)

    return outSignals
