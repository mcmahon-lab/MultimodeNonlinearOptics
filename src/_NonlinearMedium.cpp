#include "_NonlinearMedium.hpp"
#include <stdexcept>
#include <limits>
#include <thread>

Eigen::FFT<double> _NonlinearMedium::fftObj = Eigen::FFT<double>();

_NonlinearMedium::_NonlinearMedium(uint nSignalModes, uint nPumpModes, bool canBePoled, uint nFieldModes,
                                   double relativeLength, std::initializer_list<double> nlLength,
                                   std::initializer_list<double> beta2, std::initializer_list<double> beta2s,
                                   const Eigen::Ref<const Arraycd>& customPump, PulseType pulseType,
                                   std::initializer_list<double> beta1, std::initializer_list<double> beta1s,
                                   std::initializer_list<double> beta3, std::initializer_list<double> beta3s,
                                   std::initializer_list<double> diffBeta0, double rayleighLength, double tMax, uint tPrecision,
                                   uint zPrecision, IntensityProfile intensityProfile, double chirp, double delay,
                                   const Eigen::Ref<const Arrayd>& poling) :
  _nSignalModes(nSignalModes), _nPumpModes(nPumpModes), _nFieldModes(nFieldModes)
{
  if (intensityProfile == IntensityProfile::Constant) rayleighLength = std::numeric_limits<double>::infinity();

  setLengths(relativeLength, nlLength, zPrecision, rayleighLength, beta2, beta2s, beta1, beta1s, beta3, beta3s);
  resetGrids(tPrecision, tMax);
  setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s, diffBeta0);

  if (canBePoled)
    setPoling(poling);

  _intensityProfile = (_rayleighLength != std::numeric_limits<double>::infinity() ?
      intensityProfile : IntensityProfile::Constant);

  if (_nPumpModes > 0) {
    _envelope.resize(_nPumpModes);
    if (customPump.size() != 0)
      setPump(customPump, chirp, delay);
    else
      setPump(pulseType, chirp, delay);
    for (uint m = 1; m < _nPumpModes; m++)
      _envelope[m].setZero(_nFreqs);
  }
}


void _NonlinearMedium::setLengths(double relativeLength, const std::vector<double>& nlLength, uint zPrecision,
                                  double rayleighLength, const std::vector<double>& beta2, const std::vector<double>& beta2s,
                                  const std::vector<double>& beta1, const std::vector<double>& beta1s,
                                  const std::vector<double>& beta3, const std::vector<double>& beta3s) {
  // Equations are normalized to either the dispersion or nonlinear length scales L_ds, L_nl
  // The total length z is given in units of dispersion length or nonlinear length, whichever is set to unit length
  // Therefore, one length scale must be kept fixed at 1. The time scale is given in units of initial width of pump.

  bool negativeLength = false;

  negativeLength |= (relativeLength <= 0 || rayleighLength <= 0);
  for (double nl : nlLength)
    negativeLength |= (nl <= 0);

  if (negativeLength) throw std::invalid_argument("Non-positive length scale");

  bool allNonUnit = true;

  for (double b : beta1)  allNonUnit &= (std::abs(b) != 1);
  for (double b : beta2)  allNonUnit &= (std::abs(b) != 1);
  for (double b : beta3)  allNonUnit &= (std::abs(b) != 1);
  for (double b : beta1s) allNonUnit &= (std::abs(b) != 1);
  for (double b : beta2s) allNonUnit &= (std::abs(b) != 1);
  for (double b : beta3s) allNonUnit &= (std::abs(b) != 1);

  for (double nl : nlLength)
    allNonUnit &= (nl != 1);

  allNonUnit &= (relativeLength != 1);
  allNonUnit &= (rayleighLength != 1);

  if (allNonUnit) throw std::invalid_argument("No unit length scale provided: please normalize variables");

  _z = relativeLength;

  auto absComp = [](double a, double b) {return (std::abs(a) < std::abs(b));};
  double minDispLength = 1 / std::abs(std::max({(beta2.empty() ? 0 : *std::max_element(beta2.begin(),  beta2.end(),  absComp)),
                                                (beta2s.empty()? 0 : *std::max_element(beta2s.begin(), beta2s.end(), absComp)),
                                                (beta1.empty() ? 0 : *std::max_element(beta1.begin(),  beta1.end(),  absComp)),
                                                (beta1s.empty()? 0 : *std::max_element(beta1s.begin(), beta1s.end(), absComp)),
                                                (beta3.empty() ? 0 : *std::max_element(beta3.begin(),  beta3.end(),  absComp)),
                                                (beta3s.empty()? 0 : *std::max_element(beta3s.begin(), beta3s.end(), absComp))}, absComp));

  // space resolution. Note: pump step is smaller to calculate the value for intermediate RK4 steps
  _nZSteps = static_cast<uint>(zPrecision * _z / std::min({1., minDispLength, rayleighLength,
                                                           (nlLength.empty()? 0 : *std::min_element(nlLength.begin(), nlLength.end()))}));
  _nZStepsP = 2 * _nZSteps - 1;
  _dz = _z / (_nZSteps - 1);
  _dzp = _z / (_nZStepsP - 1);

  // step sizes for the RK in the simulation
  _nlStep.resize(nlLength.size());
  for (uint process = 0; process < nlLength.size(); process++) {
    if (nlLength[process] <= 0)
      throw std::invalid_argument("Invalid nonlinear length scale");
    _nlStep[process] = 1._I / nlLength[process] * _dz;
  }

  _rayleighLength = rayleighLength;
}


void _NonlinearMedium::resetGrids(uint nFreqs, double tMax) {

  // time windowing and resolution
  if (nFreqs % 2 != 0 || nFreqs == 0)
    throw std::invalid_argument("Invalid number of Frequencies");
  if (_nZSteps == 0)
    throw std::invalid_argument("Zero steps");
  if (tMax <= 0)
    throw std::invalid_argument("Negative time span");

  _nFreqs = nFreqs;
  _tMax = tMax;

  int Nt = static_cast<int>(_nFreqs);

  // time and frequency axes
  _tau = 2 * tMax / Nt * Arrayd::LinSpaced(Nt, -Nt / 2, Nt / 2 - 1);
  _tau = fftshift(_tau);
  _omega = -M_PI / _tMax * Arrayd::LinSpaced(Nt, -Nt / 2, Nt / 2 - 1);
  _omega = fftshift(_omega);

  // Grids for PDE propagation
  pumpFreq.resize(_nPumpModes);
  pumpTime.resize(_nPumpModes);
  for (uint m = 0; m < _nPumpModes; m++) {
    pumpFreq[m].setZero(_nZStepsP, _nFreqs);
    pumpTime[m].setZero(_nZStepsP, _nFreqs);
  }

  field.resize(_nFieldModes);
  for (uint m = 0; m < _nFieldModes; m++) {
    field[m].setZero(_nZStepsP, _nFreqs);
  }

  signalFreq.resize(_nSignalModes);
  signalTime.resize(_nSignalModes);
  for (uint m = 0; m < _nSignalModes; m++) {
    signalFreq[m].resize(_nZSteps, _nFreqs);
    signalTime[m].resize(_nZSteps, _nFreqs);
  }
}


void _NonlinearMedium::setDispersion(const std::vector<double>& beta2, const std::vector<double>& beta2s,
                                     const std::vector<double>& beta1, const std::vector<double>& beta1s,
                                     const std::vector<double>& beta3, const std::vector<double>& beta3s,
                                     std::initializer_list<double> diffBeta0) {

  // Pump group velocity dispersion
  _beta2 = beta2;
  _beta1 = beta1;

  // signal phase mismatch
  _diffBeta0 = diffBeta0;

  // dispersion profile
  _dispersionPump.resize(_nPumpModes);
  for (uint m = 0; m < _nPumpModes; m++)
    _dispersionPump[m] = _omega * (beta1[m] + _omega * (0.5 * beta2[m] + _omega * beta3[m] / 6));

  _dispersionSign.resize(_nSignalModes);
  for (uint m = 0; m < _nSignalModes; m++)
    _dispersionSign[m] = _omega * (beta1s[m] + _omega * (0.5 * beta2s[m] + _omega * beta3s[m] / 6));

  // incremental phases for each simulation step
  _dispStepPump.resize(_nPumpModes);
  for (uint m = 0; m < _nPumpModes; m++)
    _dispStepPump[m] = ((1._I * _dzp) * _dispersionPump[m]).exp();

  _dispStepSign.resize(_nSignalModes);
  for (uint m = 0; m < _nSignalModes; m++)
    _dispStepSign[m] = ((1._I * _dz) * _dispersionSign[m]).exp();
}


void _NonlinearMedium::setPump(PulseType pulseType, double chirpLength, double delayLength, uint pumpIndex) {
  if (pumpIndex >= _nPumpModes)
    throw std::invalid_argument("Invalid pump index");

  // initial time domain envelopes (pick Gaussian, Hyperbolic Secant, Sinc)
  switch (pulseType) {
    default:
    case PulseType::Gaussian:
      _envelope[pumpIndex] = (-0.5 * _tau.square()).exp().cast<std::complex<double>>();
      break;
    case PulseType::Sech:
      _envelope[pumpIndex] = (1 / _tau.cosh()).cast<std::complex<double>>();
      break;
    case PulseType::Sinc:
      _envelope[pumpIndex] = (_tau.sin() / _tau).cast<std::complex<double>>();
      _envelope[pumpIndex](0) = 1;
  }

  if (chirpLength != 0 || delayLength != 0) {
    Arraycd fftTemp(_nFreqs);
    FFT(fftTemp, _envelope[pumpIndex]);
    fftTemp *= (1._I * (_beta1[pumpIndex] * delayLength + 0.5 * _beta2[pumpIndex] * chirpLength * _omega) * _omega).exp();
    IFFT(_envelope[pumpIndex], fftTemp);
  }
}


void _NonlinearMedium::setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength, double delayLength, uint pumpIndex) {
  // custom initial time domain envelope
  if (customPump.size() != _nFreqs)
    throw std::invalid_argument("Custom pump array length does not match number of frequency/time bins");
  if (pumpIndex >= _nPumpModes)
    throw std::invalid_argument("Invalid pump index");

  _envelope[pumpIndex] = customPump;

  if (chirpLength != 0 || delayLength != 0) {
    Arraycd fftTemp(_nFreqs);
    FFT(fftTemp, _envelope[pumpIndex]);
    fftTemp *= (1._I * (_beta1[pumpIndex] * delayLength + 0.5 * _beta2[pumpIndex] * chirpLength * _omega) * _omega).exp();
    IFFT(_envelope[pumpIndex], fftTemp);
  }
}


void _NonlinearMedium::runPumpSimulation() {
  for (uint m = 0; m < _nPumpModes; m++) {
    FFTi(pumpFreq[m], _envelope[m], 0, 0);
    pumpTime[m].row(0) = _envelope[m];

    for (uint i = 1; i < _nZStepsP; i++) {
      pumpFreq[m].row(i) = pumpFreq[m].row(0) * (1._I * (i * _dzp) * _dispersionPump[m]).exp();
      IFFTi(pumpTime[m], pumpFreq[m], i, i);
    }

    if (_intensityProfile != IntensityProfile::Constant) {
      Eigen::VectorXd relativeStrength;
      switch (_intensityProfile) {
        case IntensityProfile::GaussianBeam:
        default:
          relativeStrength = 1 / (1 + (Arrayd::LinSpaced(_nZStepsP, -0.5 * _z, 0.5 * _z) / _rayleighLength).square()).sqrt();
          break;
        case IntensityProfile::GaussianApodization:
          relativeStrength = (-0.5 * (Arrayd::LinSpaced(_nZStepsP, -0.5 * _z, 0.5 * _z) / _rayleighLength).square()).exp();
      }

      pumpFreq[m].colwise() *= relativeStrength.array();
      pumpTime[m].colwise() *= relativeStrength.array();
    }
  }
}


void _NonlinearMedium::runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain, uint inputMode) {
  if (inputProf.size() % _nFreqs != 0 || inputProf.size() / _nFreqs == 0 || inputProf.size() / _nFreqs > _nSignalModes)
    throw std::invalid_argument("inputProf array size does not match number of frequency/time bins");
  if (inputMode >= _nSignalModes)
    throw std::invalid_argument("inputModes does not match any mode in the system");

  dispatchSignalSim(inputProf, inTimeDomain, inputMode, signalFreq, signalTime, false);
}


std::pair<Array2Dcd, Array2Dcd>
_NonlinearMedium::computeGreensFunction(bool inTimeDomain, bool runPump, uint nThreads, bool normalize,
                                        const std::vector<uint8_t>& useInput, const std::vector<uint8_t>& useOutput) {
  // Determine which input and output modes to compute. If no input/output modes specified, computes all modes.
  uint nInputModes = 0, nOutputModes = 0;
  std::vector<uint> inputs, outputs;
  if (useInput.size() > _nSignalModes)
    throw std::invalid_argument("List of requested inputs indices longer than number of modes!");
  if (useOutput.size() > _nSignalModes)
    throw std::invalid_argument("List of requested output indices longer than number of modes!");

  if (!useInput.empty()) {
    for (auto value : useInput)
      nInputModes += (value < _nSignalModes);
    if (nInputModes == 0)
      throw std::invalid_argument("Requested no valid inputs!");
  }
  else
    nInputModes = _nSignalModes;

  if (!useOutput.empty()) {
    for (auto value : useOutput)
      nOutputModes += (value < _nSignalModes);
    if (nOutputModes == 0)
      throw std::invalid_argument("Requested no valid outputs!");
  }
  else
    nOutputModes = _nSignalModes;

  inputs.reserve(nInputModes);
  outputs.reserve(nOutputModes);
  for (uint m = 0; m < _nSignalModes; m++) {
    if (useInput.empty() || useInput[m])
      inputs.emplace_back(m);
    if (useOutput.empty() || useOutput[m])
      outputs.emplace_back(m);
  }

  if (nThreads > _nFreqs * nInputModes)
    throw std::invalid_argument("Too many threads requested!");

  if (runPump) runPumpSimulation();

  // Green function matrices -- Note: hopefully large enough to avoid dirtying cache?
  Array2Dcd greenC;
  Array2Dcd greenS;
  greenC.setZero(nInputModes * _nFreqs, nOutputModes * _nFreqs);
  greenS.setZero(nInputModes * _nFreqs, nOutputModes * _nFreqs);

  // run n-1 separate threads
  std::vector<std::thread> threads;
  threads.reserve(nThreads - 1);

  // Calculate Green's functions with real and imaginary impulse response
  auto calcGreensPart = [&, inTimeDomain, _nFreqs=_nFreqs](uint start, uint stop) {
    // As a trick for memory efficiency here we use a single array instead of 2D time and frequency grids
    std::vector<Array2Dcd> gridFreq(_nSignalModes);
    for (uint m = 0; m < _nSignalModes; m++) gridFreq[m].resize(1, _nFreqs);
    std::vector<Array2Dcd> gridTime(_nSignalModes);
    for (uint m = 0; m < _nSignalModes; m++) gridTime[m].resize(1, _nFreqs);

    auto& grid = inTimeDomain ? gridTime : gridFreq;

    for (uint i = start; i < stop; i++) {
      uint im = i / _nFreqs;

      grid[inputs[im]].row(0) = 0;
      grid[inputs[im]](0, i % _nFreqs) = 1;
      dispatchSignalSim(grid[inputs[im]].row(0), inTimeDomain, inputs[im], gridFreq, gridTime, true);

      for (uint om = 0; om < nOutputModes; om++) {
        greenC.row(i).segment(om*_nFreqs, _nFreqs) += 0.5 * grid[outputs[om]].bottomRows<1>();
        greenS.row(i).segment(om*_nFreqs, _nFreqs) += 0.5 * grid[outputs[om]].bottomRows<1>();
      }

      grid[inputs[im]].row(0) = 0;
      grid[inputs[im]](0, i % _nFreqs) = 1._I;
      dispatchSignalSim(grid[inputs[im]].row(0), inTimeDomain, inputs[im], gridFreq, gridTime, true);

      for (uint om = 0; om < nOutputModes; om++) {
        greenC.row(i).segment(om*_nFreqs, _nFreqs) -= 0.5_I * grid[outputs[om]].bottomRows<1>();
        greenS.row(i).segment(om*_nFreqs, _nFreqs) += 0.5_I * grid[outputs[om]].bottomRows<1>();
      }
    }
  };

  // Spawn threads. One batch will be processed in original thread.
  for (uint i = 1; i < nThreads; i++) {
    threads.emplace_back(calcGreensPart, (i * _nFreqs * nInputModes) / nThreads, ((i + 1) * _nFreqs * nInputModes) / nThreads);
  }
  calcGreensPart(0, (_nFreqs * nInputModes) / nThreads);
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }

  // Transpose and shift individual sub-blocks so that frequencies or times are contiguous
  // If normalizing mode amplitudes, appropriately scale the conversion sub-matrices (nonlinear lengths must be ordered correctly)
  greenC.transposeInPlace();
  for (uint im = 0; im < nOutputModes; im++)
    for (uint om = 0; om < nOutputModes; om++)
      greenC.block(om * _nFreqs, im * _nFreqs, _nFreqs, _nFreqs) = normalize && im != om ?
          fftshift2(greenC.block(om * _nFreqs, im * _nFreqs, _nFreqs, _nFreqs)) * sqrt(_nlStep[im] / _nlStep[om]):
          fftshift2(greenC.block(om * _nFreqs, im * _nFreqs, _nFreqs, _nFreqs));

  greenS.transposeInPlace();
  for (uint im = 0; im < nOutputModes; im++)
    for (uint om = 0; om < nOutputModes; om++)
      greenS.block(om * _nFreqs, im * _nFreqs, _nFreqs, _nFreqs) = normalize && im != om ?
          fftshift2(greenS.block(om * _nFreqs, im * _nFreqs, _nFreqs, _nFreqs)) * sqrt(_nlStep[im] / _nlStep[om]):
          fftshift2(greenS.block(om * _nFreqs, im * _nFreqs, _nFreqs, _nFreqs));

  return std::make_pair(std::move(greenC), std::move(greenS));
}


Array2Dcd _NonlinearMedium::batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs, bool inTimeDomain,
                                                  bool runPump, uint nThreads, uint inputMode, const std::vector<uint8_t>& useOutput) {

  auto nInputs = inputProfs.rows();
  auto inCols  = inputProfs.cols();
  if (inCols % _nFreqs != 0 || inCols / _nFreqs == 0 || inCols / _nFreqs > _nSignalModes)
    throw std::invalid_argument("Signals not of correct length!");

  if (nThreads > nInputs)
    throw std::invalid_argument("Too many threads requested!");

  if (inputMode >= _nSignalModes)
    throw std::invalid_argument("inputModes does not match any mode in the system");

  // Determine which output modes to return. If none specified, returns all modes.
  uint nOutputModes = 0;
  std::vector<uint> outputs;
  if (!useOutput.empty()) {
    for (auto value : useOutput)
      nOutputModes += (value != 0);
    if (nOutputModes == 0)
      throw std::invalid_argument("Requested no outputs!");
  }
  else
    nOutputModes = _nSignalModes;
  outputs.reserve(nOutputModes);
  for (uint m = 0; m < _nSignalModes; m++)
    if (useOutput.empty() || useOutput[m])
      outputs.emplace_back(m);

  if (runPump) runPumpSimulation();

  // Signal outputs -- Note: hopefully large enough to avoid dirtying cache?
  Array2Dcd outSignals(nInputs, nOutputModes * _nFreqs);

  // run n-1 separate threads
  std::vector<std::thread> threads;
  threads.reserve(nThreads - 1);

  // Calculate all signal propagations
  auto calcBatch = [&, inTimeDomain, _nFreqs=_nFreqs] (uint start, uint stop) {
    // As a trick for memory efficiency here we use a single array instead of 2D time and frequency grids
    std::vector<Array2Dcd> gridFreq(_nSignalModes);
    for (uint m = 0; m < _nSignalModes; m++) gridFreq[m].resize(1, _nFreqs);
    std::vector<Array2Dcd> gridTime(_nSignalModes);
    for (uint m = 0; m < _nSignalModes; m++) gridTime[m].resize(1, _nFreqs);

    auto& grid = inTimeDomain ? gridTime : gridFreq;

    for (uint i = start; i < stop; i++) {
      dispatchSignalSim(inputProfs.row(i), inTimeDomain, inputMode, gridFreq, gridTime, true);
      for (uint om = 0; om < nOutputModes; om++)
        outSignals.row(i).segment(om*_nFreqs, _nFreqs) = grid[outputs[om]].bottomRows<1>();
    }
  };

  // Spawn threads. One batch will be processed in original thread.
  for (uint i = 1; i < nThreads; i++) {
    threads.emplace_back(calcBatch, (i * nInputs) / nThreads, ((i + 1) * nInputs) / nThreads);
  }
  calcBatch(0, nInputs / nThreads);
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }

  return outSignals;
}


inline Arrayd _NonlinearMedium::fftshift(const Arrayd& input) {
  Arrayd out(input.rows(), input.cols());
  auto half = input.cols() / 2;
  out.head(half) = input.tail(half);
  out.tail(half) = input.head(half);
  return out;
}


inline Array2Dcd _NonlinearMedium::fftshift2(const Array2Dcd& input) {
  Array2Dcd out(input.rows(), input.cols());

  auto halfCols = input.cols() / 2;
  auto halfRows = input.rows() / 2;

  out.topLeftCorner(halfRows, halfCols) = input.bottomRightCorner(halfRows, halfCols);
  out.topRightCorner(halfRows, halfCols) = input.bottomLeftCorner(halfRows, halfCols);
  out.bottomLeftCorner(halfRows, halfCols) = input.topRightCorner(halfRows, halfCols);
  out.bottomRightCorner(halfRows, halfCols) = input.topLeftCorner(halfRows, halfCols);
  return out;
}


void _NonlinearMedium::setPoling(const Eigen::Ref<const Arrayd>& poling) {
  if (poling.cols() <= 1)
    _poling.setOnes(_nZSteps);
  else {
    if ((poling <= 0).any())
      throw std::invalid_argument("Poling contains invalid domain length");

    Arrayd poleDomains(poling.cols());
    // cumulative sum
    poleDomains(0) = poling(0);
    for (uint i = 1; i < poling.cols(); i++) poleDomains(i) = poling(i) + poleDomains(i-1);

    poleDomains *= _nZSteps / poleDomains(poleDomains.cols()-1);

    _poling.resize(_nZSteps);
    uint prevInd = 0;
    int direction = 1;
    for (uint i = 0; i < poleDomains.cols(); i++) {
      const double currInd = poleDomains(i);
      const uint currIndRound = static_cast<uint>(currInd);

      if (currInd < prevInd)
        throw std::invalid_argument("Poling period too small for simulation resolution");

      _poling.segment(prevInd, currIndRound - prevInd) = direction;

      if (currIndRound < _nZSteps) // interpolate indices corresponding to steps on the boundary of two domains
        _poling(currIndRound) = direction * (2 * std::abs(std::fmod(currInd, 1)) - 1);

      direction *= - 1;
      prevInd = currIndRound + 1;
    }
  }
}


void _NonlinearMedium::setPump(const _NonlinearMedium& other, uint modeIndex, double delayLength, uint pumpIndex) {
  if (other._nFreqs != _nFreqs || other._tMax != _tMax)
    throw std::invalid_argument("Medium does not have same time and frequency axes as this one");

  if (modeIndex >= _nSignalModes)
    throw std::invalid_argument("Mode index larger than number of modes in medium");

  if (pumpIndex >= _nPumpModes)
    throw std::invalid_argument("Invalid pump index");

  if (other._nZSteps < _nZSteps)
    throw std::invalid_argument("Medium does not have sufficient resolution to be used with this one");

  auto delay = 1._I * _beta1[pumpIndex] * delayLength * _omega;
  for (uint i = 0; i < _nZStepsP - 1; i++) {
    double j_ = i * (static_cast<double>(other._nZSteps - 1) / (_nZStepsP - 1)); // integer overflow danger
    uint j = static_cast<uint>(j_);
    double frac = j_ - j;

    pumpFreq[pumpIndex].row(i) = ((1 - frac) * other.signalFreq[modeIndex].row(j) + frac * other.signalFreq[modeIndex].row(j+1))
        * ((1._I * (i * _dzp)) * _dispersionPump[pumpIndex] - (1._I * (j_ + 0.5) * other._dz) * other._dispersionSign[modeIndex] + delay).exp();

    IFFTi(pumpTime[pumpIndex], pumpFreq[pumpIndex], i, i);
  }
  pumpFreq[pumpIndex].bottomRows<1>() = other.signalFreq[modeIndex].bottomRows<1>()
      * ((1._I * _z) * _dispersionPump[pumpIndex] - (1._I * other._z) * other._dispersionSign[modeIndex] + delay).exp();

  IFFTi(pumpTime[pumpIndex], pumpFreq[pumpIndex], _nZSteps-1, _nZSteps-1);
}
