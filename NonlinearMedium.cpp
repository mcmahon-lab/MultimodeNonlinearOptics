#include "NonlinearMedium.hpp"
#include <stdexcept>
#include <limits>
#include <thread>

inline constexpr std::complex<double> operator"" _I(long double c) {return std::complex<double> {0, static_cast<double>(c)};}

// This way is verified to be the most efficient, avoiding allocation of temporaries.
// Note: it seems that some compilers will throw a taking address of temporary error in EigenFFT.
// This is due to array->matrix casting, the code will work if disabling the warning and compiling.
#define FFT(output, input) { \
  fftObj.fwd(fftTemp, (input).matrix()); \
  output = fftTemp.array(); }
#define FFTtimes(output, input, phase) { \
  fftObj.fwd(fftTemp, (input).matrix()); \
  output = fftTemp.array() * phase; }
#define IFFT(output, input) { \
  fftObj.inv(fftTemp, (input).matrix()); \
  output = fftTemp.array(); }


_NonlinearMedium::_NonlinearMedium(uint nSignalmodes, bool canBePoled, double relativeLength, std::initializer_list<double> nlLength,
                                   double beta2, std::initializer_list<double> beta2s, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                                   double beta1, std::initializer_list<double> beta1s, double beta3, std::initializer_list<double> beta3s,
                                   std::initializer_list<double> diffBeta0, double rayleighLength, double tMax, uint tPrecision, uint zPrecision,
                                   double chirp, double delay, const Eigen::Ref<const Arrayd>& poling) :
  _nSignalModes(nSignalmodes)
{
  setLengths(relativeLength, nlLength, zPrecision, rayleighLength, beta2, beta2s, beta1, beta1s, beta3, beta3s);
  resetGrids(tPrecision, tMax);
  setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s, diffBeta0);
  if (canBePoled)
    setPoling(poling);
  if (customPump.size() != 0)
    setPump(customPump, chirp, delay);
  else
    setPump(pulseType, chirp, delay);
}


void _NonlinearMedium::setLengths(double relativeLength, const std::vector<double>& nlLength, uint zPrecision,
                                  double rayleighLength, double beta2, const std::vector<double>& beta2s, double beta1,
                                  const std::vector<double>& beta1s, double beta3, const std::vector<double>& beta3s) {
  // Equations are normalized to either the dispersion or nonlinear length scales L_ds, L_nl
  // The total length z is given in units of dispersion length or nonlinear length, whichever is set to unit length
  // Therefore, one length scale must be kept fixed at 1. The time scale is given in units of initial width of pump.

  bool negativeLength = false;

  negativeLength |= (std::abs(relativeLength) <= 0 || std::abs(rayleighLength) <= 0);
  for (double nl : nlLength)
    negativeLength |= (nl <= 0);

  if (negativeLength) throw std::invalid_argument("Non-positive length scale");

  bool allNonUnit = true;

  allNonUnit &= (std::abs(beta1) != 1 && std::abs(beta2) != 1 && std::abs(beta3) != 1);
  for (double b : beta1s) allNonUnit &= (std::abs(b) != 1);
  for (double b : beta2s) allNonUnit &= (std::abs(b) != 1);
  for (double b : beta3s) allNonUnit &= (std::abs(b) != 1);

  for (double nl : nlLength)
    allNonUnit &= (nl != 1);

  if (allNonUnit) throw std::invalid_argument("No unit length scale provided: please normalize variables");

  _z = relativeLength;

  auto absComp = [](double a, double b) {return (std::abs(a) < std::abs(b));};
  double maxDispLength = 1 / std::abs(std::max({beta2, *std::max_element(beta2s.begin(), beta2s.end(), absComp),
                                                beta1, *std::max_element(beta1s.begin(), beta1s.end(), absComp),
                                                beta3, *std::max_element(beta3s.begin(), beta3s.end(), absComp)}, absComp));

  // space resolution
  _nZSteps = static_cast<uint>(zPrecision * _z / std::min({1., maxDispLength, rayleighLength,
                                                           *std::min_element(nlLength.begin(), nlLength.end())}));
  _dz = _z / _nZSteps;

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

  int Nt = _nFreqs;

  // time and frequency axes
  _tau = 2 * tMax / Nt * Arrayd::LinSpaced(Nt, -Nt / 2, Nt / 2 - 1);
  _tau = fftshift(_tau);
  _omega = -M_PI / _tMax * Arrayd::LinSpaced(Nt, -Nt / 2, Nt / 2 - 1);
  _omega = fftshift(_omega);

  // Grids for PDE propagation
  pumpFreq.resize(_nZSteps, _nFreqs);
  pumpTime.resize(_nZSteps, _nFreqs);

  signalFreq.resize(_nSignalModes);
  signalTime.resize(_nSignalModes);
  for (uint m = 0; m < _nSignalModes; m++) {
    signalFreq[m].resize(_nZSteps, _nFreqs);
    signalTime[m].resize(_nZSteps, _nFreqs);
  }
}


void _NonlinearMedium::setDispersion(double beta2, const std::vector<double>& beta2s, double beta1, const std::vector<double>& beta1s,
                                     double beta3, const std::vector<double>& beta3s, std::initializer_list<double> diffBeta0) {

  // Pump group velocity dispersion
  _beta2  = beta2;
  _beta1  = beta1;

  // signal phase mis-match
  _diffBeta0 = diffBeta0;

  // dispersion profile; all beta coefficients must be normalized with respect to some dispersion length scale
  _dispersionSign.resize(_nSignalModes);
  _dispersionPump = _omega * (beta1  + _omega * (0.5 * beta2  + _omega * beta3  / 6));
  for (uint m = 0; m < _nSignalModes; m++)
    _dispersionSign[m] = _omega * (beta1s[m] + _omega * (0.5 * beta2s[m] + _omega * beta3s[m] / 6));

  // incremental phases for each simulation step
  _dispStepPump = (1._I * _dispersionPump * _dz).exp();
  _dispStepSign.resize(_nSignalModes);
  for (uint m = 0; m < _nSignalModes; m++)
    _dispStepSign[m] = (1._I * _dispersionSign[m] * _dz).exp();
}


void _NonlinearMedium::setPump(int pulseType, double chirpLength, double delayLength) {
  // initial time domain envelopes (pick Gaussian, Hyperbolic Secant, Sinc)
  if (pulseType == 1)
    _env = (1 / _tau.cosh()).cast<std::complex<double>>();
  else if (pulseType == 2) {
    _env = (_tau.sin() / _tau).cast<std::complex<double>>();
    _env(0) = 1;
  }
  else
    _env = (-0.5 * _tau.square()).exp().cast<std::complex<double>>();

  if (chirpLength != 0 || delayLength != 0) {
    RowVectorcd fftTemp(_nFreqs);
    FFTtimes(fftTemp, _env, (1._I * (_beta1 * delayLength + 0.5 * _beta2 * chirpLength * _omega) * _omega).exp())
    IFFT(_env, fftTemp)
  }
}

void _NonlinearMedium::setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength, double delayLength) {
  // custom initial time domain envelope
  if (customPump.size() != _nFreqs)
    throw std::invalid_argument("Custom pump array length does not match number of frequency/time bins");
  _env = customPump;

  if (chirpLength != 0 || delayLength != 0) {
    RowVectorcd fftTemp(_nFreqs);
    FFTtimes(fftTemp, _env, (1._I * (_beta1 * delayLength + 0.5 * _beta2 * chirpLength * _omega) * _omega).exp())
    IFFT(_env, fftTemp)
  }
}


void _NonlinearMedium::runPumpSimulation() {
  RowVectorcd fftTemp(_nFreqs);

  FFT(pumpFreq.row(0), _env)
  pumpTime.row(0) = _env;

  for (uint i = 1; i < _nZSteps; i++) {
    pumpFreq.row(i) = pumpFreq.row(0) * (1._I * (i * _dz) * _dispersionPump).exp();
    IFFT(pumpTime.row(i), pumpFreq.row(i))
  }

  if (_rayleighLength != std::numeric_limits<double>::infinity()) {
    Eigen::VectorXd relativeStrength = 1 / (1 + (Arrayd::LinSpaced(_nZSteps, -0.5 * _z, 0.5 * _z) / _rayleighLength).square()).sqrt();
    pumpFreq.colwise() *= relativeStrength.array();
    pumpTime.colwise() *= relativeStrength.array();
  }
}


void _NonlinearMedium::runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain, uint inputMode) {
  if (inputProf.size() % _nFreqs != 0 || inputProf.size() / _nFreqs == 0 || inputProf.size() / _nFreqs > _nSignalModes)
    throw std::invalid_argument("inputProf array size does not match number of frequency/time bins");
  if (inputMode >= _nSignalModes)
    throw std::invalid_argument("inputModes does not match any mode in the system");

  runSignalSimulation(inputProf, inTimeDomain, inputMode, signalFreq, signalTime);
}


std::pair<Array2Dcd, Array2Dcd>
_NonlinearMedium::computeGreensFunction(bool inTimeDomain, bool runPump, uint nThreads, bool normalize,
                                        const std::vector<char>& useInput, const std::vector<char>& useOutput) {
  // Determine which input and output modes to compute. If no input/output modes specified, computes all modes.
  uint nInputModes = 0, nOutputModes = 0;
  std::vector<uint> inputs, outputs;
  if (useInput.size() > _nSignalModes)
    throw std::invalid_argument("List of requested inputs indices longer than number of modes!");
  if (useOutput.size() > _nSignalModes)
    throw std::invalid_argument("List of requested output indices longer than number of modes!");

  if (!useInput.empty()) {
    for (auto value : useInput)
      nInputModes += (value != 0);
    if (nInputModes == 0)
      throw std::invalid_argument("Requested no inputs!");
  }
  else
    nInputModes = _nSignalModes;

  if (!useOutput.empty()) {
    for (auto value : useOutput)
      nOutputModes += (value != 0);
    if (nOutputModes == 0)
      throw std::invalid_argument("Requested no outputs!");
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

  // Each thread needs a separate computation grid to avoid interfering with other threads
  std::vector<std::vector<Array2Dcd>> grids(2 * (nThreads - 1));

  // Calculate Green's functions with real and imaginary impulse response
  auto calcGreensPart = [&, inTimeDomain, _nZSteps=_nZSteps, _nFreqs=_nFreqs]
                        (std::vector<Array2Dcd>& gridFreq, std::vector<Array2Dcd>& gridTime, uint start, uint stop) {
    if (gridFreq.size() == 0) {
      gridFreq.resize(_nSignalModes);
      for (uint m = 0; m < _nSignalModes; m++) gridFreq[m].resize(_nZSteps, _nFreqs);
    }
    if (gridTime.size() == 0) {
      gridTime.resize(_nSignalModes);
      for (uint m = 0; m < _nSignalModes; m++) gridTime[m].resize(_nZSteps, _nFreqs);
    }
    auto& grid = inTimeDomain ? gridTime : gridFreq;

    for (uint i = start; i < stop; i++) {
      uint im = i / _nFreqs;

      grid[inputs[im]].row(0) = 0;
      grid[inputs[im]](0, i % _nFreqs) = 1;
      runSignalSimulation(grid[inputs[im]].row(0), inTimeDomain, inputs[im], gridFreq, gridTime);

      for (uint om = 0; om < nOutputModes; om++) {
        greenC.row(i).segment(om*_nFreqs, _nFreqs) += 0.5 * grid[outputs[om]].bottomRows<1>();
        greenS.row(i).segment(om*_nFreqs, _nFreqs) += 0.5 * grid[outputs[om]].bottomRows<1>();
      }

      grid[inputs[im]].row(0) = 0;
      grid[inputs[im]](0, i % _nFreqs) = 1._I;
      runSignalSimulation(grid[inputs[im]].row(0), inTimeDomain, inputs[im], gridFreq, gridTime);

      for (uint om = 0; om < nOutputModes; om++) {
        greenC.row(i).segment(om*_nFreqs, _nFreqs) -= 0.5_I * grid[outputs[om]].bottomRows<1>();
        greenS.row(i).segment(om*_nFreqs, _nFreqs) += 0.5_I * grid[outputs[om]].bottomRows<1>();
      }
    }
  };

  // Spawn threads. One batch will be processed in original thread.
  for (uint i = 1; i < nThreads; i++) {
    threads.emplace_back(calcGreensPart, std::ref(grids[2*i-2]), std::ref(grids[2*i-1]),
                         (i * _nFreqs * nInputModes) / nThreads, ((i + 1) * _nFreqs * nInputModes) / nThreads);
  }
  calcGreensPart(signalFreq, signalTime, 0, (_nFreqs * nInputModes) / nThreads);
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
                                                  bool runPump, uint nThreads, uint inputMode, const std::vector<char>& useOutput) {

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

  // Each thread needs a separate computation grid to avoid interfering with other threads
  std::vector<std::vector<Array2Dcd>> grids(2 * (nThreads - 1));

  // Calculate all signal propagations
  auto calcBatch = [&, inTimeDomain, _nZSteps=_nZSteps, _nFreqs=_nFreqs]
                   (std::vector<Array2Dcd>& gridFreq, std::vector<Array2Dcd>& gridTime, uint start, uint stop) {
    if (gridFreq.size() == 0) {
      gridFreq.resize(_nSignalModes);
      for (uint m = 0; m < _nSignalModes; m++) gridFreq[m].resize(_nZSteps, _nFreqs);
    }
    if (gridTime.size() == 0) {
      gridTime.resize(_nSignalModes);
      for (uint m = 0; m < _nSignalModes; m++) gridTime[m].resize(_nZSteps, _nFreqs);
    }
    auto& grid = inTimeDomain ? gridTime : gridFreq;

    for (uint i = start; i < stop; i++) {
      runSignalSimulation(inputProfs.row(i), inTimeDomain, inputMode, gridFreq, gridTime);
      for (uint om = 0; om < nOutputModes; om++)
        outSignals.row(i).segment(om*_nFreqs, _nFreqs) = grid[outputs[om]].bottomRows<1>();
    }
  };

  // Spawn threads. One batch will be processed in original thread.
  for (uint i = 1; i < nThreads; i++) {
    threads.emplace_back(calcBatch, std::ref(grids[2*i-2]), std::ref(grids[2*i-1]),
                         (i * nInputs) / nThreads, ((i + 1) * nInputs) / nThreads);
  }
  calcBatch(signalFreq, signalTime, 0, nInputs / nThreads);
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


template<class T>
void _NonlinearMedium::signalSimulationTemplate(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                                std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime) {
  RowVectorcd fftTemp(_nFreqs);

  // Can specify: input to any 1 mode by passing a length N array, or an input to the first x consecutive modes with a length x*N array
  uint nInputChannels = inputProf.size() / _nFreqs;
  if (nInputChannels > 1) inputMode = 0;
  if (T::_nSignalModes <= 1) inputMode = 0; // compiler guarantee

  if (inTimeDomain)
    for (uint m = 0; m < T::_nSignalModes; m++) {
      if (m == inputMode) {
        signalFreq[m].row(0) = inputProf.segment(0, _nFreqs); // hack: fft on inputProf sometimes fails
        FFTtimes(signalFreq[m].row(0), signalFreq[m].row(0), ((0.5_I * _dz) * _dispersionSign[m]).exp())
      }
      else if (inputMode < 1 && m < nInputChannels) {
        signalFreq[m].row(0) = inputProf.segment(m*_nFreqs, _nFreqs); // hack: fft on inputProf sometimes fails
        FFTtimes(signalFreq[m].row(0), signalFreq[m].row(0), ((0.5_I * _dz) * _dispersionSign[m]).exp())
      }
      else
        signalFreq[m].row(0) = 0;
    }
  else
    for (uint m = 0; m < T::_nSignalModes; m++) {
      if (m == inputMode)
        signalFreq[m].row(0) = inputProf.segment(0, _nFreqs) * ((0.5_I * _dz) * _dispersionSign[m]).exp();
      else if (inputMode < 1 && m < nInputChannels)
        signalFreq[m].row(0) = inputProf.segment(m*_nFreqs, _nFreqs) * ((0.5_I * _dz) * _dispersionSign[m]).exp();
      else
        signalFreq[m].row(0) = 0;
    }
  for (uint m = 0; m < T::_nSignalModes; m++) {
    if (m == inputMode || m < nInputChannels)
      IFFT(signalTime[m].row(0), signalFreq[m].row(0))
    else
      signalTime[m].row(0) = 0;
  }

  Arraycd interpP(_nFreqs), temp(_nFreqs);
  std::vector<Arraycd> k1(T::_nSignalModes), k2(T::_nSignalModes), k3(T::_nSignalModes), k4(T::_nSignalModes);
  for (uint m = 0; m < T::_nSignalModes; m++) {
    k1[m].resize(_nFreqs); k2[m].resize(_nFreqs); k3[m].resize(_nFreqs); k4[m].resize(_nFreqs);
  }
  for (uint i = 1; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the non-linear propagation
    const auto& prevP = pumpTime.row(i-1);
    const auto& currP = pumpTime.row(i);

    interpP = 0.5 * (prevP + currP);

    static_cast<T*>(this)->DiffEq(i, k1, k2, k3, k4, prevP, currP, interpP, signalTime);

    for (uint m = 0; m < T::_nSignalModes; m++) {
      temp = signalTime[m].row(i - 1) + (k1[m] + 2 * k2[m] + 2 * k3[m] + k4[m]) / 6;

      // Dispersion step
      FFTtimes(signalFreq[m].row(i), temp, _dispStepSign[m])
      IFFT(signalTime[m].row(i), signalFreq[m].row(i))
    }
  }

  for (uint m = 0; m < T::_nSignalModes; m++) {
    signalFreq[m].row(_nZSteps - 1) *= ((-0.5_I * _dz) * _dispersionSign[m]).exp();
    IFFT(signalTime[m].row(_nZSteps - 1), signalFreq[m].row(_nZSteps - 1))
  }
}


Chi3::Chi3(double relativeLength, double nlLength, double beta2, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
           double beta3, double rayleighLength, double tMax, uint tPrecision, uint zPrecision, double chirp) :
  _NonlinearMedium(_nSignalModes, false, relativeLength, {nlLength}, beta2, {beta2}, customPump, pulseType,
                   0, {0}, beta3, {beta3}, {}, rayleighLength, tMax, tPrecision, zPrecision, chirp, 0)
{}


void Chi3::runPumpSimulation() {
  RowVectorcd fftTemp(_nFreqs);

  FFTtimes(pumpFreq.row(0), _env, ((0.5_I * _dz) * _dispersionPump).exp())
  IFFT(pumpTime.row(0), pumpFreq.row(0))

  Eigen::VectorXd relativeStrength = 1 / (1 + (Arrayd::LinSpaced(_nZSteps, -0.5 * _z, 0.5 * _z) / _rayleighLength).square()).sqrt();

  Arraycd temp(_nFreqs);
  for (uint i = 1; i < _nZSteps; i++) {
    temp = pumpTime.row(i-1) * (_nlStep[0] * relativeStrength(i-1) * pumpTime.row(i-1).abs2()).exp();
    FFTtimes(pumpFreq.row(i), temp, _dispStepPump)
    IFFT(pumpTime.row(i), pumpFreq.row(i))
  }

  pumpFreq.row(_nZSteps-1) *= ((-0.5_I * _dz) * _dispersionPump).exp();
  IFFT(pumpTime.row(_nZSteps-1), pumpFreq.row(_nZSteps-1))
}


void Chi3::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, std::vector<Arraycd>& k4,
                  const Arraycd& prevP, const Arraycd& currP, const Arraycd& interpP, const std::vector<Array2Dcd>& signal) {
  const auto& prev = signal[0].row(i-1);
  k1[0] = _nlStep[0] * (2 * prevP.abs2()   *  prev                + prevP.square()   *  prev.conjugate());
  k2[0] = _nlStep[0] * (2 * interpP.abs2() * (prev + 0.5 * k1[0]) + interpP.square() * (prev + 0.5 * k1[0]).conjugate());
  k3[0] = _nlStep[0] * (2 * interpP.abs2() * (prev + 0.5 * k2[0]) + interpP.square() * (prev + 0.5 * k2[0]).conjugate());
  k4[0] = _nlStep[0] * (2 * currP.abs2()   * (prev + k3[0])       + currP.square()   * (prev + k3[0]).conjugate());
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
    for (int i = 1; i < poling.cols(); i++) poleDomains(i) = poling(i) + poleDomains(i-1);

    poleDomains *= _nZSteps / poleDomains(poleDomains.cols()-1);

    _poling.resize(_nZSteps);
    uint prevInd = 0;
    int direction = 1;
    for (int i = 0; i < poleDomains.cols(); i++) {
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


Chi2PDC::Chi2PDC(double relativeLength, double nlLength, double beta2, double beta2s,
                 const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                 double beta1, double beta1s, double beta3, double beta3s, double diffBeta0,
                 double rayleighLength, double tMax, uint tPrecision, uint zPrecision, double chirp, double delay,
                 const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(_nSignalModes, true, relativeLength, {nlLength}, beta2, {beta2s}, customPump, pulseType, beta1, {beta1s}, beta3,
                   {beta3s}, {diffBeta0}, rayleighLength, tMax, tPrecision, zPrecision, chirp, delay, poling) {}


void Chi2PDC::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, std::vector<Arraycd>& k4,
                     const Arraycd& prevP, const Arraycd& currP, const Arraycd& interpP, const std::vector<Array2Dcd>& signal) {
  const auto& prev = signal[0].row(i-1);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));

  k1[0] = (prevPolDir * _nlStep[0] * prevMismatch) * prevP   *  prev.conjugate();
  k2[0] = (intmPolDir * _nlStep[0] * intmMismatch) * interpP * (prev + 0.5 * k1[0]).conjugate();
  k3[0] = (intmPolDir * _nlStep[0] * intmMismatch) * interpP * (prev + 0.5 * k2[0]).conjugate();
  k4[0] = (currPolDir * _nlStep[0] * currMismatch) * currP   * (prev + k3[0]).conjugate();
}


#ifdef DEPLETESHG
Chi2SHG::Chi2SHG(double relativeLength, double nlLength, double nlLengthP, double beta2, double beta2s,
#else
Chi2SHG::Chi2SHG(double relativeLength, double nlLength, double beta2, double beta2s,
#endif
                 const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                 double beta1, double beta1s, double beta3, double beta3s, double diffBeta0,
                 double rayleighLength, double tMax, uint tPrecision, uint zPrecision, double chirp, double delay,
                 const Eigen::Ref<const Arrayd>& poling) :

#ifdef DEPLETESHG
  _NonlinearMedium(_nSignalModes, true, relativeLength, {nlLength, nlLengthP}, beta2, {beta2s}, customPump, pulseType,
#else
  _NonlinearMedium(_nSignalModes, true, relativeLength, {nlLength}, beta2, {beta2s}, customPump, pulseType,
#endif
                   beta1, {beta1s}, beta3, {beta3s}, {diffBeta0}, rayleighLength, tMax, tPrecision, zPrecision, chirp, delay, poling)
{}


void Chi2SHG::runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                  std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime) {
  RowVectorcd fftTemp(_nFreqs);

#ifdef DEPLETESHG
  // NOTE: DEPLETESHG not thread safe due to use of single pumpTime and pumpFreq arrays
  FFTtimes(pumpFreq.row(0), _env, ((0.5_I * _dz) * _dispersionPump).exp())
  IFFT(pumpTime.row(0), pumpFreq.row(0))
#endif

  if (inTimeDomain)
    FFTtimes(signalFreq[0].row(0), inputProf, ((0.5_I * _dz) * _dispersionSign[0]).exp())
  else
    signalFreq[0].row(0) = inputProf * ((0.5_I * _dz) * _dispersionSign[0]).exp();
  IFFT(signalTime[0].row(0), signalFreq[0].row(0))

  Arraycd interpP(_nFreqs), k1(_nFreqs), k2(_nFreqs), k3(_nFreqs), k4(_nFreqs), temp(_nFreqs);
#ifdef DEPLETESHG
  Arraycd l1(_nFreqs), l2(_nFreqs), l3(_nFreqs), l4(_nFreqs), tempPump(_nFreqs);
#endif
  for (uint i = 1; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the non-linear propagation
    const auto& prevP = pumpTime.row(i-1);
#ifdef DEPLETESHG
    const auto& prevS = signalTime[0].row(i-1);
#else
    const auto& currP = pumpTime.row(i);
    interpP = 0.5 * (prevP + currP);
#endif

    const double prevPolDir = _poling(i-1);
    const double currPolDir = _poling(i);
    const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

    const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
    const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
    const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));

    k1 = (prevPolDir * _nlStep[0] * prevMismatch) * prevP.square();
#ifdef DEPLETESHG
    l1 = (prevPolDir * _nlStep[1] / prevMismatch) *  prevS * prevP.conjugate();
    k2 = (intmPolDir * _nlStep[0] * intmMismatch) * (prevP + 0.5 * l1).square();
    l2 = (intmPolDir * _nlStep[1] / intmMismatch) * (prevS + 0.5 * k1) * (prevP + 0.5 * l1).conjugate();
    k3 = (intmPolDir * _nlStep[0] * intmMismatch) * (prevP + 0.5 * l2).square();
    l3 = (intmPolDir * _nlStep[1] / intmMismatch) * (prevS + 0.5 * k2) * (prevP + 0.5 * l2).conjugate();
    k4 = (currPolDir * _nlStep[0] * currMismatch) * (prevP + l3).square();
    l4 = (currPolDir * _nlStep[1] / currMismatch) * (prevS + k3) * (prevP + l3).conjugate();
#else
    k2 = (intmPolDir * _nlStep[0] * intmMismatch) * interpP.square();
    k3 = (intmPolDir * _nlStep[0] * intmMismatch) * interpP.square();
    k4 = (currPolDir * _nlStep[0] * currMismatch) * currP.square();
#endif
    temp = signalTime[0].row(i-1) + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    FFTtimes(signalFreq[0].row(i), temp, _dispStepSign[0])
    IFFT(signalTime[0].row(i), signalFreq[0].row(i))

#ifdef DEPLETESHG
    tempPump = pumpTime.row(i-1) + (l1 + 2 * l2 + 2 * l3 + l4) / 6;
    FFTtimes(pumpFreq.row(i), tempPump, _dispStepPump)
    IFFT(pumpTime.row(i), pumpFreq.row(i))
#endif
  }

  signalFreq[0].row(_nZSteps-1) *= ((-0.5_I * _dz) * _dispersionSign[0]).exp();
  IFFT(signalTime[0].row(_nZSteps-1), signalFreq[0].row(_nZSteps-1))
#ifdef DEPLETESHG
  pumpFreq.row(_nZSteps-1) *= ((-0.5_I * _dz) * _dispersionPump).exp();
  IFFT(pumpTime.row(_nZSteps-1), pumpFreq.row(_nZSteps-1))
#endif
}


Chi2SFGPDC::Chi2SFGPDC(double relativeLength, double nlLength, double nlLengthOrig, double beta2, double beta2s, double beta2o,
                       const Eigen::Ref<const Arraycd>& customPump, int pulseType, double beta1, double beta1s, double beta1o,
                       double beta3, double beta3s, double beta3o, double diffBeta0, double diffBeta0o, double rayleighLength,
                       double tMax, uint tPrecision, uint zPrecision, double chirp, double delay, const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(_nSignalModes, true, relativeLength, {nlLength, nlLengthOrig}, beta2, {beta2s, beta2o},
                   customPump, pulseType, beta1, {beta1s, beta1o}, beta3, {beta3s, beta3o}, {diffBeta0, diffBeta0o},
                   rayleighLength, tMax, tPrecision, zPrecision, chirp, delay, poling) {}


void Chi2SFGPDC::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, std::vector<Arraycd>& k4,
                        const Arraycd& prevP, const Arraycd& currP, const Arraycd& interpP, const std::vector<Array2Dcd>& signal) {
  auto& prevS = signal[0].row(i-1);
  auto& prevO = signal[1].row(i-1);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevInvMsmch = 1. / prevMismatch;
  const std::complex<double> intmInvMsmch = 1. / intmMismatch;
  const std::complex<double> currInvMsmch = 1. / currMismatch;
  const std::complex<double> prevMismatcho = std::exp(1._I * _diffBeta0[1] * ((i- 1) * _dz));
  const std::complex<double> intmMismatcho = std::exp(1._I * _diffBeta0[1] * ((i-.5) * _dz));
  const std::complex<double> currMismatcho = std::exp(1._I * _diffBeta0[1] * ( i     * _dz));

  k1[0] = (prevPolDir * _nlStep[0]  *  prevMismatch) * prevP               *  prevO;
  k1[1] = (prevPolDir * _nlStep[1]) * (prevInvMsmch  * prevP.conjugate()   *  prevS                + prevMismatcho * prevP   *  prevO.conjugate());
  k2[0] = (intmPolDir * _nlStep[0]  *  intmMismatch) * interpP             * (prevO + 0.5 * k1[1]);
  k2[1] = (intmPolDir * _nlStep[1]) * (intmInvMsmch  * interpP.conjugate() * (prevS + 0.5 * k1[0]) + intmMismatcho * interpP * (prevO + 0.5 * k1[1]).conjugate());
  k3[0] = (intmPolDir * _nlStep[0]  *  intmMismatch) * interpP             * (prevO + 0.5 * k2[1]);
  k3[1] = (intmPolDir * _nlStep[1]) * (intmInvMsmch  * interpP.conjugate() * (prevS + 0.5 * k2[0]) + intmMismatcho * interpP * (prevO + 0.5 * k2[1]).conjugate());
  k4[0] = (currPolDir * _nlStep[0]  *  currMismatch) * currP               * (prevO + k3[1]);
  k4[1] = (currPolDir * _nlStep[1]) * (currInvMsmch  * currP.conjugate()   * (prevS + k3[0])       + currMismatcho * currP   * (prevO + k3[1]).conjugate());
}


Chi2SFG::Chi2SFG(double relativeLength, double nlLength, double nlLengthOrig, double beta2, double beta2s, double beta2o,
                 const Eigen::Ref<const Arraycd>& customPump, int pulseType, double beta1, double beta1s, double beta1o,
                 double beta3, double beta3s, double beta3o, double diffBeta0, double rayleighLength,
                 double tMax, uint tPrecision, uint zPrecision, double chirp, double delay, const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(_nSignalModes, true, relativeLength, {nlLength, nlLengthOrig}, beta2, {beta2s, beta2o},
                   customPump, pulseType, beta1, {beta1s, beta1o}, beta3, {beta3s, beta3o}, {diffBeta0},
                   rayleighLength, tMax, tPrecision, zPrecision, chirp, delay, poling) {}


void Chi2SFG::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, std::vector<Arraycd>& k4,
                        const Arraycd& prevP, const Arraycd& currP, const Arraycd& interpP, const std::vector<Array2Dcd>& signal) {
  auto& prevS = signal[0].row(i-1);
  auto& prevO = signal[1].row(i-1);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevInvMsmch = 1. / prevMismatch;
  const std::complex<double> intmInvMsmch = 1. / intmMismatch;
  const std::complex<double> currInvMsmch = 1. / currMismatch;

  k1[0] = (prevPolDir * _nlStep[0] * prevMismatch) * prevP               *  prevO;
  k1[1] = (prevPolDir * _nlStep[1] * prevInvMsmch) * prevP.conjugate()   *  prevS;
  k2[0] = (intmPolDir * _nlStep[0] * intmMismatch) * interpP             * (prevO + 0.5 * k1[1]);
  k2[1] = (intmPolDir * _nlStep[1] * intmInvMsmch) * interpP.conjugate() * (prevS + 0.5 * k1[0]);
  k3[0] = (intmPolDir * _nlStep[0] * intmMismatch) * interpP             * (prevO + 0.5 * k2[1]);
  k3[1] = (intmPolDir * _nlStep[1] * intmInvMsmch) * interpP.conjugate() * (prevS + 0.5 * k2[0]);
  k4[0] = (currPolDir * _nlStep[0] * currMismatch) * currP               * (prevO + k3[1]);
  k4[1] = (currPolDir * _nlStep[1] * currInvMsmch) * currP.conjugate()   * (prevS + k3[0]);
}


Chi2PDCII::Chi2PDCII(double relativeLength, double nlLength, double nlLengthOrig, double nlLengthI,
                     double beta2, double beta2s, double beta2o, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                     double beta1, double beta1s, double beta1o, double beta3, double beta3s, double beta3o,
                     double diffBeta0, double diffBeta0o, double rayleighLength, double tMax, uint tPrecision, uint zPrecision,
                     double chirp, double delay, const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(_nSignalModes, true, relativeLength, {nlLength, nlLengthOrig, nlLengthI}, beta2, {beta2s, beta2o},
                   customPump, pulseType, beta1, {beta1s, beta1o}, beta3, {beta3s, beta3o}, {diffBeta0, diffBeta0o},
                   rayleighLength, tMax, tPrecision, zPrecision, chirp, delay, poling) {}


void Chi2PDCII::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, std::vector<Arraycd>& k4,
                       const Arraycd& prevP, const Arraycd& currP, const Arraycd& interpP, const std::vector<Array2Dcd>& signal) {

  auto& prevS = signal[0].row(i-1);
  auto& prevO = signal[1].row(i-1);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevMismatcho = std::exp(1._I * _diffBeta0[1] * ((i- 1) * _dz));
  const std::complex<double> intmMismatcho = std::exp(1._I * _diffBeta0[1] * ((i-.5) * _dz));
  const std::complex<double> currMismatcho = std::exp(1._I * _diffBeta0[1] * ( i     * _dz));

  k1[0] = (prevPolDir *   _nlStep[0] * prevMismatch) * prevP   *  prevO.conjugate();
  k1[1] =  prevPolDir * ((_nlStep[1] * prevMismatch) * prevP   *  prevS.conjugate()                + (_nlStep[2] * prevMismatcho) * prevP   *  prevO.conjugate());
  k2[0] = (intmPolDir *   _nlStep[0] * intmMismatch) * interpP * (prevO + 0.5 * k1[1]).conjugate();
  k2[1] =  intmPolDir * ((_nlStep[1] * intmMismatch) * interpP * (prevS + 0.5 * k1[0]).conjugate() + (_nlStep[2] * intmMismatcho) * interpP * (prevO + 0.5 * k1[1]).conjugate());
  k3[0] = (intmPolDir *   _nlStep[0] * intmMismatch) * interpP * (prevO + 0.5 * k2[1]).conjugate();
  k3[1] =  intmPolDir * ((_nlStep[1] * intmMismatch) * interpP * (prevS + 0.5 * k2[0]).conjugate() + (_nlStep[2] * intmMismatcho) * interpP * (prevO + 0.5 * k2[1]).conjugate());
  k4[0] = (currPolDir *   _nlStep[0] * currMismatch) * currP   * (prevO + k3[1]).conjugate();
  k4[1] =  currPolDir * ((_nlStep[1] * currMismatch) * currP   * (prevS + k3[0]).conjugate()       + (_nlStep[2] * currMismatcho) * currP   * (prevO + k3[1]).conjugate());
}


Chi2SFGII::Chi2SFGII(double relativeLength, double nlLengthSignZ, double nlLengthSignY, double nlLengthOrigZ, double nlLengthOrigY,
                     double beta2, double beta2sz, double beta2sy, double beta2oz, double beta2oy,
                     const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                     double beta1, double beta1sz, double beta1sy, double beta1oz, double beta1oy,
                     double beta3, double beta3sz, double beta3sy, double beta3oz, double beta3oy,
                     double diffBeta0z, double diffBeta0y, double diffBeta0s, double rayleighLength,
                     double tMax, uint tPrecision, uint zPrecision, double chirp, double delay, const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(_nSignalModes, true, relativeLength, {nlLengthSignZ, nlLengthSignY, nlLengthOrigZ, nlLengthOrigY},
                   beta2, {beta2sz, beta2sy, beta2oz, beta2oy}, customPump, pulseType, beta1, {beta1sz, beta1sy, beta1oz,  beta1oy},
                   beta3, {beta3sz, beta3sy, beta3oz, beta3oy}, {diffBeta0z, diffBeta0y, diffBeta0s},
                   rayleighLength, tMax, tPrecision, zPrecision, chirp, delay, poling) {}


void Chi2SFGII::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, std::vector<Arraycd>& k4,
                       const Arraycd& prevP, const Arraycd& currP, const Arraycd& interpP, const std::vector<Array2Dcd>& signal) {

  auto& prevSz = signal[0].row(i-1);
  auto& prevSy = signal[1].row(i-1);
  auto& prevOz = signal[2].row(i-1);
  auto& prevOy = signal[3].row(i-1);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const std::complex<double> prevMismatchCz = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchCz = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatchCz = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevInvMsmchCz = 1. / prevMismatchCz;
  const std::complex<double> intmInvMsmchCz = 1. / intmMismatchCz;
  const std::complex<double> currInvMsmchCz = 1. / currMismatchCz;

  const std::complex<double> prevMismatchCy = std::exp(1._I * _diffBeta0[1] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchCy = std::exp(1._I * _diffBeta0[1] * ((i-.5) * _dz));
  const std::complex<double> currMismatchCy = std::exp(1._I * _diffBeta0[1] * ( i     * _dz));
  const std::complex<double> prevInvMsmchCy = 1. / prevMismatchCy;
  const std::complex<double> intmInvMsmchCy = 1. / intmMismatchCy;
  const std::complex<double> currInvMsmchCy = 1. / currMismatchCy;

  const std::complex<double> prevMismatchSq = std::exp(1._I * _diffBeta0[2] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchSq = std::exp(1._I * _diffBeta0[2] * ((i-.5) * _dz));
  const std::complex<double> currMismatchSq = std::exp(1._I * _diffBeta0[2] * ( i     * _dz));

  k1[0] = (prevPolDir * _nlStep[0]  * prevMismatchCz) * prevP * prevOy;
  k1[1] = (prevPolDir * _nlStep[1]  * prevMismatchCy) * prevP * prevOz;
  k1[2] = (prevPolDir * _nlStep[2]) * (prevInvMsmchCy * prevP.conjugate() * prevSy + prevMismatchSq * prevP * prevOy.conjugate());
  k1[3] = (prevPolDir * _nlStep[3]) * (prevInvMsmchCz * prevP.conjugate() * prevSz + prevMismatchSq * prevP * prevOz.conjugate());

  k2[0] = (intmPolDir * _nlStep[0]  * intmMismatchCz) * interpP * (prevOy + 0.5 * k1[3]);
  k2[1] = (intmPolDir * _nlStep[1]  * intmMismatchCy) * interpP * (prevOz + 0.5 * k1[2]);
  k2[2] = (intmPolDir * _nlStep[2]) * (intmInvMsmchCy * interpP.conjugate() * (prevSy + 0.5 * k1[1]) + intmMismatchSq * interpP * (prevOy + 0.5 * k1[3]).conjugate());
  k2[3] = (intmPolDir * _nlStep[3]) * (intmInvMsmchCz * interpP.conjugate() * (prevSz + 0.5 * k1[0]) + intmMismatchSq * interpP * (prevOz + 0.5 * k1[2]).conjugate());

  k3[0] = (intmPolDir * _nlStep[0]  * intmMismatchCz) * interpP * (prevOy + 0.5 * k2[3]);
  k3[1] = (intmPolDir * _nlStep[1]  * intmMismatchCy) * interpP * (prevOz + 0.5 * k2[2]);
  k3[2] = (intmPolDir * _nlStep[2]) * (intmInvMsmchCy * interpP.conjugate() * (prevSy + 0.5 * k2[1]) + intmMismatchSq * interpP * (prevOy + 0.5 * k2[3]).conjugate());
  k3[3] = (intmPolDir * _nlStep[3]) * (intmInvMsmchCz * interpP.conjugate() * (prevSz + 0.5 * k2[0]) + intmMismatchSq * interpP * (prevOz + 0.5 * k2[2]).conjugate());

  k4[0] = (prevPolDir * _nlStep[0]  * currMismatchCz) * currP * (prevOy + k3[3]);
  k4[1] = (prevPolDir * _nlStep[1]  * currMismatchCy) * currP * (prevOz + k3[2]);
  k4[2] = (prevPolDir * _nlStep[2]) * (currInvMsmchCy * currP.conjugate() * (prevSy + k3[1]) + currMismatchSq * currP * (prevOy + k3[3]).conjugate());
  k4[3] = (prevPolDir * _nlStep[3]) * (currInvMsmchCz * currP.conjugate() * (prevSz + k3[0]) + currMismatchSq * currP * (prevOz + k3[2]).conjugate());
}


Cascade::Cascade(bool sharePump, const std::vector<std::reference_wrapper<_NonlinearMedium>>& inputMedia,
                const std::vector<std::map<uint, uint>>& modeConnections) {

  if (inputMedia.empty())
    throw std::invalid_argument("Cascade must contain at least one medium");

  _nFreqs = inputMedia[0].get()._nFreqs;
  _tMax = inputMedia[0].get()._tMax;

  media.reserve(inputMedia.size());
  for (auto& medium : inputMedia) {
    if (medium.get()._nFreqs != _nFreqs or medium.get()._tMax != _tMax)
      throw std::invalid_argument("Medium does not have same time and frequency axes as the first");
    media.emplace_back(medium);
    _nZSteps += medium.get()._nZSteps;
  }

  if (modeConnections.size() != media.size() - 1)
    throw std::invalid_argument("Must have one connection per pair of adjacent media");
  uint i = 0;
  for (auto connection = modeConnections.begin(); connection != modeConnections.end(); ++connection, ++i) {
    if (connection->empty())
      throw std::invalid_argument("No connection!");
    for (auto& signalMap : *connection) {
        if (signalMap.first >= media[i].get()._nSignalModes || signalMap.second >= media[i+1].get()._nSignalModes)
          throw std::invalid_argument("Invalid connections, out of range");
    }
    connections.emplace_back(*connection);
  }

  sharedPump = sharePump;
}


void Cascade::addMedium(_NonlinearMedium& medium, const std::map<uint, uint>& connection) {
  if (medium._nFreqs != _nFreqs or medium._tMax != _tMax)
    throw std::invalid_argument("Medium does not have same time and frequency axes as the first");

  if (connection.empty())
    throw std::invalid_argument("No connection!");
  for (auto& signalMap : connection) {
    // todo check uniqueness of values? is this function necessary?
    if (signalMap.first >= medium._nSignalModes || signalMap.second >= medium._nSignalModes)
      throw std::invalid_argument("Invalid connections, out of range");
  }

  media.emplace_back(medium);
  connections.emplace_back(connection);
}


void Cascade::setPump(int pulseType, double chirpLength, double delayLength) {
  if (sharedPump)
    media[0].get().setPump(pulseType, chirpLength, delayLength);
  else {
    for (auto& medium : media)
      medium.get().setPump(pulseType, chirpLength, delayLength);
  }
}


void Cascade::setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength, double delayLength) {
  if (sharedPump)
    media[0].get().setPump(customPump, chirpLength, delayLength);
  else {
    for (auto& medium : media)
      medium.get().setPump(customPump, chirpLength, delayLength);
  }
}


void Cascade::runPumpSimulation() {
  if (not sharedPump) {
    for (auto& medium : media) {
      medium.get().runPumpSimulation();
    }
  }
  else {
    media[0].get().runPumpSimulation();
    for (uint i = 1; i < media.size(); i++) {
      media[i].get()._env = media[i-1].get().pumpTime.bottomRows<1>();
      media[i].get().runPumpSimulation();
    }
  }
}


void Cascade::runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain, uint inputMode) {
  if (inputProf.size() != _nFreqs)
    throw std::invalid_argument("inputProf array size does not match number of frequency/time bins");

  media[0].get().runSignalSimulation(inputProf, inTimeDomain, inputMode);
  for (uint i = 1; i < media.size(); i++) {
    // TODO connect signal channels / specify modes
    media[i].get().runSignalSimulation(media[i-1].get().signalFreq[inputMode].bottomRows<1>(), false);
  }
}


std::pair<Array2Dcd, Array2Dcd>
Cascade::computeGreensFunction(bool inTimeDomain, bool runPump, uint nThreads, bool normalize,
                               const std::vector<char>& useInput, const std::vector<char>& useOutput) {

  if (runPump) runPumpSimulation();

  // Green function matrices
  Array2Dcd greenC;
  Array2Dcd greenS;
  greenC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(_nFreqs, _nFreqs);
  greenS.setZero(_nFreqs, _nFreqs);

  Array2Dcd tempC, tempS;
  for (auto& medium : media) {
    // TODO useInput and useOutput need to be defined based on connections
    auto CandS = medium.get().computeGreensFunction(inTimeDomain, false, nThreads, normalize);
    tempC = std::get<0>(CandS).matrix() * greenC.matrix() + std::get<1>(CandS).matrix() * greenS.conjugate().matrix();
    tempS = std::get<0>(CandS).matrix() * greenS.matrix() + std::get<1>(CandS).matrix() * greenC.conjugate().matrix();
    greenC.swap(tempC);
    greenS.swap(tempS);
  }

  return std::make_pair(std::move(greenC), std::move(greenS));
}


Array2Dcd Cascade::batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs,
                                         bool inTimeDomain, bool runPump, uint nThreads,
                                         uint inputMode, const std::vector<char>& useOutput) {
  if (runPump) runPumpSimulation();

  Array2Dcd outSignals = media[0].get().batchSignalSimulation(inputProfs, inTimeDomain, false, nThreads);

  for (uint i = 1; i < media.size(); i++) {
    // TODO connect signal channels / specify modes
    outSignals = media[i].get().batchSignalSimulation(outSignals, inTimeDomain, false, nThreads);
  }

  return outSignals;
}
