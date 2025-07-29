#include "_NonlinearMedium.hpp"
#include <stdexcept>
#include <limits>
#include <thread>

Eigen::FFT<double> _NonlinearMedium::fftObj = Eigen::FFT<double>();

_NonlinearMedium::_NonlinearMedium(uint nSignalModes, uint nDimensions, uint nPumpModes, bool canBePoled, uint nFieldModes,
                                   double relativeLength, std::initializer_list<double> nlLength,
                                   std::initializer_list<double> beta2, std::initializer_list<double> beta2s,
                                   const Eigen::Ref<const Arraycd>& customPump, PulseType pulseType,
                                   std::initializer_list<double> beta1, std::initializer_list<double> beta1s,
                                   std::initializer_list<double> beta3, std::initializer_list<double> beta3s,
                                   std::initializer_list<double> diffBeta0, double rayleighLength,
                                   std::initializer_list<double> tMax, std::initializer_list<uint> tPrecision,
                                   uint zPrecision, uint ratioStepsToRecord, IntensityProfile intensityProfile,
                                   double chirp, double delay, const Eigen::Ref<const Arrayd>& poling) :
  _nSignalModes(nSignalModes), _nPumpModes(nPumpModes), _nFieldModes(nFieldModes), _nDimensions(nDimensions)
{
  if (intensityProfile == IntensityProfile::Constant) rayleighLength = std::numeric_limits<double>::infinity();

  setLengths(relativeLength, nlLength, zPrecision, rayleighLength, beta2, beta2s, beta1, beta1s, beta3, beta3s);
  resetGrids(tPrecision, tMax, ratioStepsToRecord);
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


void _NonlinearMedium::resetGrids(const std::vector<uint>& nFreqs, const std::vector<double>& tMax, uint ratioStepsToRecord) {

  // time windowing and resolution
  for (auto f : nFreqs)
    if (f % 2 != 0 || f == 0)
      throw std::invalid_argument("Invalid number of Frequencies");
  if (_nZSteps == 0)
    throw std::invalid_argument("Zero steps");
  for (auto t : tMax)
    if (t <= 0)
      throw std::invalid_argument("Negative time span");

  _nFreqsPerDim = nFreqs;
  _nFreqs = 1; // _nFreqs is the total number of points for >1D simulations
  for (auto nF : nFreqs) {
    _nFreqs *= nF;
  }
  _tMax = tMax;

  // time and frequency axes
  _tau.resize(_nDimensions);
  _omega.resize(_nDimensions);
  for (uint d = 0; d < _nDimensions; d++) {
    int Nt = static_cast<int>(nFreqs[d]);

    _tau[d] = 2 * tMax[d] / Nt * Arrayd::LinSpaced(Nt, -Nt / 2, Nt / 2 - 1);
    _tau[d] = fftshift(_tau[d]);
    _omega[d] = -M_PI / tMax[d] * Arrayd::LinSpaced(Nt, -Nt / 2, Nt / 2 - 1);
    _omega[d] = fftshift(_omega[d]);
  }

  _ratioStepsToRecord = ratioStepsToRecord;

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
  uint gridSize = _nZSteps / _ratioStepsToRecord;
  gridSize += (_nZSteps % _ratioStepsToRecord? 1 : 0);
  for (uint m = 0; m < _nSignalModes; m++) {
    signalFreq[m].resize(gridSize, _nFreqs);
    signalTime[m].resize(gridSize, _nFreqs);
  }
}


void _NonlinearMedium::setDispersion(const std::vector<double>& beta2, const std::vector<double>& beta2s,
                                     const std::vector<double>& beta1, const std::vector<double>& beta1s,
                                     const std::vector<double>& beta3, const std::vector<double>& beta3s,
                                     std::initializer_list<double> diffBeta0) {
  uint expectedSize = _nSignalModes * _nDimensions;
  if (beta1s.size() != expectedSize || beta2s.size() != expectedSize || beta3s.size() != expectedSize)
    throw std::invalid_argument("Incorrect number of parameters for given number of signal modes");
  expectedSize = _nPumpModes * _nDimensions;
  if (beta1.size() != expectedSize || beta2.size() != expectedSize || beta3.size() != expectedSize)
    throw std::invalid_argument("Incorrect number of parameters for given number of pump modes");

  // Pump group velocity dispersion
  _beta2 = beta2;
  _beta1 = beta1;

  // signal phase mismatch
  _diffBeta0 = diffBeta0;

  // dispersion profile
  auto updateIndex = [&](std::vector<uint>& index) {
    index[_nDimensions - 1] += 1;
    for (uint d = _nDimensions - 1; d > 0; d--) {
      if (index[d] == _nFreqsPerDim[d]) {
        index[d] = 0;
        index[d-1] += 1;
      } else break;
    }
  };
  std::vector<uint> strides(_nDimensions);
  strides[0] = _nFreqs / _nFreqsPerDim[0];
  for (uint d = 1; d < _nDimensions; d++) strides[d] = strides[d-1] / _nFreqsPerDim[d];

  _dispersionPump.resize(_nPumpModes);
  for (uint m = 0; m < _nPumpModes; m++) { // iterate over the modes
    _dispersionPump[m].setZero(_nFreqs);
    std::vector<uint> index(_nDimensions); // represents the multidimensional index
    for (uint i = 0; i < _nFreqs; i++) { // iterate over each pixel, if it is on the boundary of some dimension(s), apply corresponding dispersion profile
      for (uint d = 0; d < _nDimensions; d++) {
        if (index[d] == 0) {
          uint betaInd = _nDimensions * m + d;
          Eigen::Map<Arrayd, 0, Eigen::InnerStride<Eigen::Dynamic>>
              stridedView(_dispersionPump[m].data() + i, _nFreqsPerDim[d], Eigen::InnerStride(strides[d]));
          stridedView += _omega[d] * (beta1[betaInd] + _omega[d] * (0.5 * beta2[betaInd] + _omega[d] * beta3[betaInd] / 6));
        }
      }
      updateIndex(index);
    }
  }
  _dispersionSign.resize(_nSignalModes);
  for (uint m = 0; m < _nSignalModes; m++) { // iterate over the modes
    _dispersionSign[m].setZero(_nFreqs);
    std::vector<uint> index(_nDimensions); // represents the multidimensional index
    for (uint i = 0; i < _nFreqs; i++) { // iterate over each pixel, if it is on the boundary of some dimension(s), apply corresponding dispersion profile
      for (uint d = 0; d < _nDimensions; d++) {
        if (index[d] == 0) {
          uint betaInd = _nDimensions * m + d;
          Eigen::Map<Arrayd, 0, Eigen::InnerStride<Eigen::Dynamic>>
              stridedView(_dispersionSign[m].data() + i, _nFreqsPerDim[d], Eigen::InnerStride(strides[d]));
          stridedView += _omega[d] * (beta1s[betaInd] + _omega[d] * (0.5 * beta2s[betaInd] + _omega[d] * beta3s[betaInd] / 6));
        }
      }
      updateIndex(index);
    }
  }

  // incremental phases for each simulation step
  _dispStepPump.resize(_nPumpModes);
  for (uint m = 0; m < _nPumpModes; m++) {
    _dispStepPump[m] = ((1._I * _dzp) * _dispersionPump[m]).exp();
  }

  _dispStepSign.resize(_nSignalModes);
  for (uint m = 0; m < _nSignalModes; m++) {
    _dispStepSign[m] = ((1._I * _dz) * _dispersionSign[m]).exp();
  }
}


void _NonlinearMedium::setPump(PulseType pulseType, double chirpLength, double delayLength, uint pumpIndex) {
  if (pumpIndex >= _nPumpModes)
    throw std::invalid_argument("Invalid pump index");

  // initial time domain envelopes (pick Gaussian, Hyperbolic Secant, Sinc)
  _envelope[pumpIndex] = Arraycd::Ones(_nFreqs);
  Eigen::Map<Array2Dcd> envMultiDimView(_envelope[pumpIndex].data(), _nFreqsPerDim[0], _envelope[pumpIndex].size() / _nFreqsPerDim[0]);

  switch (pulseType) {
    default:
    case PulseType::Gaussian:
      envMultiDimView.rowwise() *= (-0.5 * _tau[0].square()).exp().cast<std::complex<double>>();
      break;
    case PulseType::Sech:
      envMultiDimView.rowwise() *= (1 / _tau[0].cosh()).cast<std::complex<double>>();
      break;
    case PulseType::Sinc:
      envMultiDimView.rowwise() *= (_tau[0].sin() / _tau[0]).cast<std::complex<double>>();
      envMultiDimView.col(0) = 1;
      break;
  }

  if (_nDimensions > 1) {
    switch (pulseType) {
      default:
      case PulseType::Gaussian:
        envMultiDimView.colwise() *= (-0.5 * _tau[1].square()).exp().cast<std::complex<double>>().transpose();
        break;
      case PulseType::Sech:
        envMultiDimView.colwise() *= (1 / _tau[1].cosh()).cast<std::complex<double>>().transpose();
        break;
      case PulseType::Sinc:
        envMultiDimView.colwise() *= (_tau[1].sin() / _tau[1]).cast<std::complex<double>>().transpose();
        envMultiDimView.row(0) = 1;
        break;
    }
  }

  if (chirpLength != 0 || delayLength != 0) {
    Arraycd fftTemp(_nFreqs);
    if (_nDimensions == 1)      FFT(fftTemp, _envelope[pumpIndex]);
    else if (_nDimensions == 1) FFT2(fftTemp, _envelope[pumpIndex]);

    fftTemp.rowwise() *= (1._I * (_beta1[_nDimensions*pumpIndex+0] * delayLength + 0.5 * _beta2[_nDimensions*pumpIndex+0] * chirpLength * _omega[0]) * _omega[0]).exp();
    if (_nDimensions > 1)
      fftTemp.colwise() *= (1._I * (_beta1[_nDimensions*pumpIndex+1] * delayLength + 0.5 * _beta2[_nDimensions*pumpIndex+1] * chirpLength * _omega[1]) * _omega[1]).exp().transpose();

    if (_nDimensions == 1)      IFFT(_envelope[pumpIndex], fftTemp);
    else if (_nDimensions == 2) IFFT2(fftTemp, fftTemp);
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
    if (_nDimensions == 1)      FFT(fftTemp, _envelope[pumpIndex]);
    else if (_nDimensions == 1) FFT2(fftTemp, _envelope[pumpIndex]);

    fftTemp.rowwise() *= (1._I * (_beta1[_nDimensions*pumpIndex+0] * delayLength + 0.5 * _beta2[_nDimensions*pumpIndex+0] * chirpLength * _omega[0]) * _omega[0]).exp();
    if (_nDimensions > 1)
      fftTemp.colwise() *= (1._I * (_beta1[_nDimensions*pumpIndex+1] * delayLength + 0.5 * _beta2[_nDimensions*pumpIndex+1] * chirpLength * _omega[1]) * _omega[1]).exp().transpose();

    if (_nDimensions == 1)      IFFT(_envelope[pumpIndex], fftTemp);
    else if (_nDimensions == 2) IFFT2(fftTemp, fftTemp);
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

  dispatchSignalSim(inputProf, inTimeDomain, inputMode, signalFreq, signalTime, _ratioStepsToRecord);
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
      dispatchSignalSim(grid[inputs[im]].row(0), inTimeDomain, inputs[im], gridFreq, gridTime, _nZSteps);

      for (uint om = 0; om < nOutputModes; om++) {
        greenC.row(i).segment(om*_nFreqs, _nFreqs) += 0.5 * grid[outputs[om]].bottomRows<1>();
        greenS.row(i).segment(om*_nFreqs, _nFreqs) += 0.5 * grid[outputs[om]].bottomRows<1>();
      }

      grid[inputs[im]].row(0) = 0;
      grid[inputs[im]](0, i % _nFreqs) = 1._I;
      dispatchSignalSim(grid[inputs[im]].row(0), inTimeDomain, inputs[im], gridFreq, gridTime, _nZSteps);

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
      dispatchSignalSim(inputProfs.row(i), inTimeDomain, inputMode, gridFreq, gridTime, _nZSteps);
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
  if (other._nFreqsPerDim != _nFreqsPerDim || other._tMax != _tMax)
    throw std::invalid_argument("Medium does not have same time and frequency axes as this one");

  if (modeIndex >= _nSignalModes)
    throw std::invalid_argument("Mode index larger than number of modes in medium");

  if (pumpIndex >= _nPumpModes)
    throw std::invalid_argument("Invalid pump index");

  if (other._nZSteps < _nZSteps)
    throw std::invalid_argument("Medium does not have sufficient resolution to be used with this one");

  Arraycd delay = 1._I * Arraycd::Ones(_nFreqs);
  delay.rowwise() *= _beta1[pumpIndex * _nDimensions + 0] * delayLength * _omega[0];
  if (_nDimensions > 1) {
    Eigen::Map<Array2Dcd> delayMultiDimView(delay.data(), _nFreqsPerDim[0], _nFreqsPerDim[1]);
    delayMultiDimView.colwise() *= (_beta1[pumpIndex * _nDimensions + 1] * delayLength * _omega[1]).transpose();
  }

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
