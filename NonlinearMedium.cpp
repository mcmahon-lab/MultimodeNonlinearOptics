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


_NonlinearMedium::_NonlinearMedium(uint nSignalmodes, double relativeLength, std::initializer_list<double> nlLength, double dispLength,
                                   double beta2, std::initializer_list<double> beta2s, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                                   double beta1, std::initializer_list<double> beta1s, double beta3, std::initializer_list<double> beta3s, std::initializer_list<double> diffBeta0,
                                   double chirp, double rayleighLength, double tMax, uint tPrecision, uint zPrecision) :
  _nSignalModes(nSignalmodes)
{
  setLengths(relativeLength, nlLength, dispLength, zPrecision, rayleighLength);
  resetGrids(tPrecision, tMax);
  setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s, diffBeta0);
  if (customPump.size() != 0)
    setPump(customPump, chirp);
  else
    setPump(pulseType, chirp);
}


void _NonlinearMedium::setLengths(double relativeLength, const std::vector<double>& nlLength, double dispLength, uint zPrecision,
                                  double rayleighLength) {
  // Equation defined in terms of dispersion and nonlinear lengh ratio N^2 = L_ds / L_nl
  // The length z is given in units of dispersion length (of pump)
  // The time is given in units of initial width (of pump)
  // L_ds must be kept fixed at 1
  // if no dispersion, L_nl must fixed at 1, otherwise, in units of l_ds

  // Total length (in units of pump dispersion length or, if infinite, nonlinear length)
  _z = relativeLength;

  _noDispersion = (dispLength == std::numeric_limits<double>::infinity());
  _noNonlinear = true;
  for (double nl : nlLength)
    _noNonlinear &= (nl == std::numeric_limits<double>::infinity());

  if (_noDispersion) {
    bool allNonUnit = true;
    for (double nl : nlLength)
      allNonUnit &= (nl != 1);
    if (allNonUnit) throw std::invalid_argument("Non unit NL");
  }
  else
    if (dispLength != 1) throw std::invalid_argument("Non unit DS");

  // Soliton order for each process
  std::vector<double> Nsquared(nlLength.size());
  for (uint process = 0; process < nlLength.size(); process++) {
    Nsquared[process] = dispLength / nlLength[process];
    if (_noDispersion) Nsquared[process] = 1;
    if (_noNonlinear)  Nsquared[process] = 0;
  }

  // space resolution
  _nZSteps = static_cast<uint>(zPrecision * _z / std::min({1., dispLength,
                                                           *std::min_element(nlLength.begin(), nlLength.end())}));
  _dz = _z / _nZSteps;

  // helper values
  _nlStep.resize(nlLength.size());
  for (uint process = 0; process < nlLength.size(); process++)
    _nlStep[process] = 1._I * Nsquared[process] * _dz;

  _rayleighLength = rayleighLength;
}


void _NonlinearMedium::resetGrids(uint nFreqs, double tMax) {

  // time windowing and resolution
  if (nFreqs % 2 != 0 || nFreqs == 0 || _nZSteps == 0 || tMax <= 0)
    throw std::invalid_argument("Invalid PDE grid size");

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

  // positive or negative dispersion for pump (ie should be +/- 1), relative dispersion for signal
  _beta2  = beta2;

  if (std::abs(beta2) != 1 && !_noDispersion)
    throw std::invalid_argument("Non unit beta2");

  // signal's phase mis-match
  _diffBeta0 = diffBeta0;

  // dispersion profile
  // group velocity difference, beta1, and beta3 are normalized with beta2 and pulse width
  _dispersionSign.resize(_nSignalModes);
  if (_noDispersion) {
    _dispersionPump.setZero(_nFreqs);
    for (uint m = 0; m < _nSignalModes; m++)
      _dispersionSign[m].setZero(_nFreqs);
  }
  else {
    _dispersionPump = _omega * (beta1  + _omega * (0.5 * beta2  + _omega * beta3  / 6));
    for (uint m = 0; m < _nSignalModes; m++)
      _dispersionSign[m] = _omega * (beta1s[m] + _omega * (0.5 * beta2s[m] + _omega * beta3s[m] / 6));
  }

  // helper values
  _dispStepPump = (1._I * _dispersionPump * _dz).exp();
  _dispStepSign.resize(_nSignalModes);
  for (uint m = 0; m < _nSignalModes; m++)
    _dispStepSign[m] = (1._I * _dispersionSign[m] * _dz).exp();
}


void _NonlinearMedium::setPump(int pulseType, double chirpLength) {
  // initial time domain envelopes (pick Gaussian, Hyperbolic Secant, Sinc)
  if (pulseType == 1)
    _env = (1 / _tau.cosh()).cast<std::complex<double>>();
  else if (pulseType == 2) {
    _env = (_tau.sin() / _tau).cast<std::complex<double>>();
    _env(0) = 1;
  }
  else
    _env = (-0.5 * _tau.square()).exp().cast<std::complex<double>>();

  if (chirpLength != 0) {
    RowVectorcd fftTemp(_nFreqs);
    FFTtimes(fftTemp, _env, (0.5_I * _beta2 * chirpLength * _omega.square()).exp())
    IFFT(_env, fftTemp)
  }
}

void _NonlinearMedium::setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength) {
  // custom initial time domain envelope
  if (customPump.size() != _nFreqs)
    throw std::invalid_argument("Custom pump array length does not match number of frequency/time bins");
  _env = customPump;

  if (chirpLength != 0) {
    RowVectorcd fftTemp(_nFreqs);
    FFTtimes(fftTemp, _env, (0.5_I * _beta2 * chirpLength * _omega.square()).exp())
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
  if (inputProf.size() > _nSignalModes * _nFreqs || inputProf.size() % _nFreqs)
    throw std::invalid_argument("inputProf array size does not match number of frequency/time bins");
  if (inputMode >= _nSignalModes)
    throw std::invalid_argument("inputModes does not match any mode in the system");

  runSignalSimulation(inputProf, inTimeDomain, inputMode, signalFreq, signalTime);
}


std::pair<Array2Dcd, Array2Dcd> _NonlinearMedium::computeGreensFunction(bool inTimeDomain, bool runPump, uint nThreads,
                                                                        const std::vector<char>& useInput,
                                                                        const std::vector<char>& useOutput) {

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
  inputs.reserve(nInputModes);
  for (uint m = 0; m < _nSignalModes; m++)
    if (useInput.empty() || useInput[m])
      inputs.emplace_back(m);

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


  if (nThreads > _nFreqs * nInputModes)
    throw std::invalid_argument("Too many threads requested!");

  if (runPump) runPumpSimulation();

  // Green function matrices -- Note: hopefully large enough to avoid dirtying cache?
  Array2Dcd greenC;
  Array2Dcd greenS;
  greenC.setZero(nInputModes * _nFreqs, nOutputModes * _nFreqs);
  greenS.setZero(nInputModes * _nFreqs, nOutputModes * _nFreqs);

  // run n-1 separate threads, run part on this process
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

  for (uint i = 1; i < nThreads; i++) {
    threads.emplace_back(calcGreensPart, std::ref(grids[2*i-2]), std::ref(grids[2*i-1]),
                         (i * _nFreqs * nInputModes) / nThreads, ((i + 1) * _nFreqs * nInputModes) / nThreads);
  }
  calcGreensPart(signalFreq, signalTime, 0, (_nFreqs * nInputModes) / nThreads);
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }

  greenC.transposeInPlace();
  for (uint im = 0; im < nOutputModes; im++)
    for (uint om = 0; om < nOutputModes; om++)
      greenC.block(om * _nFreqs, im * _nFreqs, _nFreqs, _nFreqs) = fftshift2(greenC.block(om * _nFreqs, im * _nFreqs, _nFreqs, _nFreqs));

  greenS.transposeInPlace();
  for (uint im = 0; im < nOutputModes; im++)
    for (uint om = 0; om < nOutputModes; om++)
      greenS.block(om * _nFreqs, im * _nFreqs, _nFreqs, _nFreqs) = fftshift2(greenS.block(om * _nFreqs, im * _nFreqs, _nFreqs, _nFreqs));

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

  // run n-1 separate threads, run part on this process
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

  uint inputChannels = inputProf.size() / _nFreqs;
  if (T::_nSignalModes <= 1) inputMode = 0; // compiler guarantee

  if (inTimeDomain)
    for (uint m = 0; m < T::_nSignalModes; m++) {
      if (m == inputMode)
        FFTtimes(signalFreq[m].row(0), inputProf.segment(0, _nFreqs), ((0.5_I * _dz) * _dispersionSign[m]).exp())
      else if (inputMode < 1 && m < inputChannels)
        FFTtimes(signalFreq[m].row(0), inputProf.segment(m*_nFreqs, _nFreqs), ((0.5_I * _dz) * _dispersionSign[m]).exp())
      else
        signalFreq[m].row(0) = 0;
    }
  else
    for (uint m = 0; m < T::_nSignalModes; m++) {
      if (m == inputMode)
        signalFreq[m].row(0) = inputProf.segment(0, _nFreqs) * ((0.5_I * _dz) * _dispersionSign[m]).exp();
      else if (inputMode < 1 && m < inputChannels)
        signalFreq[m].row(0) = inputProf.segment(m*_nFreqs, _nFreqs) * ((0.5_I * _dz) * _dispersionSign[m]).exp();
      else
        signalFreq[m].row(0) = 0;
    }
  for (uint m = 0; m < T::_nSignalModes; m++) {
    if (m == inputMode)
      IFFT(signalTime[m].row(0), signalFreq[m].row(0))
    else if (inputMode < 1 && m < inputChannels)
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


Chi3::Chi3(double relativeLength, double nlLength, double dispLength,
           double beta2, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
           double beta3, double chirp, double rayleighLength, double tMax, uint tPrecision, uint zPrecision) :
    _NonlinearMedium(_nSignalModes, relativeLength, {nlLength}, dispLength, beta2, {beta2}, customPump, pulseType,
                     0, {0}, beta3, {beta3}, {0}, chirp, rayleighLength, tMax, tPrecision, zPrecision)
{}


void Chi3::runPumpSimulation() {
  RowVectorcd fftTemp(_nFreqs);

  FFTtimes(pumpFreq.row(0), _env, ((0.5_I * _dz) * _dispersionPump).exp())
  IFFT(pumpTime.row(0), pumpFreq.row(0))

  Arraycd temp(_nFreqs);
  for (uint i = 1; i < _nZSteps; i++) {
    temp = pumpTime.row(i-1) * (_nlStep[0] * pumpTime.row(i-1).abs2()).exp();
    FFTtimes(pumpFreq.row(i), temp, _dispStepPump)
    IFFT(pumpTime.row(i), pumpFreq.row(i))
  }

  pumpFreq.row(_nZSteps-1) *= ((-0.5_I * _dz) * _dispersionPump).exp();
  IFFT(pumpTime.row(_nZSteps-1), pumpFreq.row(_nZSteps-1))

  if (_rayleighLength != std::numeric_limits<double>::infinity()) {
    Eigen::VectorXd relativeStrength = 1 / (1 + (Arrayd::LinSpaced(_nZSteps, -0.5 * _z, 0.5 * _z) / _rayleighLength).square()).sqrt();
    pumpFreq.colwise() *= relativeStrength.array();
    pumpTime.colwise() *= relativeStrength.array();
  }
}


void Chi3::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, std::vector<Arraycd>& k4,
                  const Arraycd& prevP, const Arraycd& currP, const Arraycd& interpP, const std::vector<Array2Dcd>& signal) {
  const auto& prev = signal[0].row(i-1);
  k1[0] = _nlStep[0] * (2 * prevP.abs2()   *  prev                + prevP.square()   *  prev.conjugate());
  k2[0] = _nlStep[0] * (2 * interpP.abs2() * (prev + 0.5 * k1[0]) + interpP.square() * (prev + 0.5 * k1[0]).conjugate());
  k3[0] = _nlStep[0] * (2 * interpP.abs2() * (prev + 0.5 * k2[0]) + interpP.square() * (prev + 0.5 * k2[0]).conjugate());
  k4[0] = _nlStep[0] * (2 * currP.abs2()   * (prev + k3[0])       + currP.square()   * (prev + k3[0]).conjugate());
}


_Chi2::_Chi2(uint nSignalmodes, double relativeLength, std::initializer_list<double> nlLength, double dispLength,
             double beta2, std::initializer_list<double> beta2s, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
             double beta1, std::initializer_list<double> beta1s, double beta3, std::initializer_list<double> beta3s, std::initializer_list<double> diffBeta0,
             double chirp, double rayleighLength, double tMax, uint tPrecision, uint zPrecision,
             const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium::_NonlinearMedium(nSignalmodes, relativeLength, nlLength, dispLength, beta2, beta2s, customPump, pulseType,
                                     beta1, beta1s, beta3, beta3s, diffBeta0, chirp, rayleighLength, tMax, tPrecision, zPrecision)
{
  setPoling(poling);
}


void _Chi2::setPoling(const Eigen::Ref<const Arrayd>& poling) {
  if (poling.cols() <= 1)
    _poling.setOnes(_nZSteps);
  else {
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


Chi2PDC::Chi2PDC(double relativeLength, double nlLength, double dispLength, double beta2, double beta2s,
                 const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                 double beta1, double beta1s, double beta3, double beta3s, double diffBeta0,
                 double chirp, double rayleighLength, double tMax, uint tPrecision, uint zPrecision,
                 const Eigen::Ref<const Arrayd>& poling) :
    _Chi2(_nSignalModes, relativeLength, {nlLength}, dispLength, beta2, {beta2s}, customPump, pulseType, beta1, {beta1s}, beta3,
          {beta3s}, {diffBeta0}, chirp, rayleighLength, tMax, tPrecision, zPrecision, poling) {}


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
Chi2SHG::Chi2SHG(double relativeLength, double nlLength, double nlLengthP, double dispLength, double beta2, double beta2s,
#else
Chi2SHG::Chi2SHG(double relativeLength, double nlLength, double dispLength, double beta2, double beta2s,
#endif
                 const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                 double beta1, double beta1s, double beta3, double beta3s, double diffBeta0,
                 double chirp, double rayleighLength, double tMax, uint tPrecision, uint zPrecision,
                 const Eigen::Ref<const Arrayd>& poling) :

#ifdef DEPLETESHG
    _Chi2::_Chi2(_nSignalModes, relativeLength, {nlLength, nlLengthP}, dispLength, beta2, {beta2s}, customPump, pulseType,
#else
    _Chi2::_Chi2(_nSignalModes, relativeLength, {nlLength}, dispLength, beta2, {beta2s}, customPump, pulseType,
#endif
                 beta1, {beta1s}, beta3, {beta3s}, {diffBeta0}, chirp, rayleighLength, tMax, tPrecision, zPrecision, poling)
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


Chi2SFG::Chi2SFG(double relativeLength, double nlLength, double nlLengthOrig, double dispLength,
                 double beta2, double beta2s, double beta2o, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                 double beta1, double beta1s, double beta1o, double beta3, double beta3s, double beta3o,
                 double diffBeta0, double diffBeta0o, double chirp, double rayleighLength,
                 double tMax, uint tPrecision, uint zPrecision, const Eigen::Ref<const Arrayd>& poling) :
  _Chi2::_Chi2(_nSignalModes, relativeLength, {nlLength, nlLengthOrig}, dispLength, beta2, {beta2s, beta2o},
               customPump, pulseType, beta1, {beta1s, beta1o}, beta3, {beta3s, beta3o}, {diffBeta0, diffBeta0o},
               chirp, rayleighLength, tMax, tPrecision, zPrecision, poling) {}


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
  const std::complex<double> prevMismatcho = std::exp(1._I * _diffBeta0[1] * ((i- 1) * _dz));
  const std::complex<double> intmMismatcho = std::exp(1._I * _diffBeta0[1] * ((i-.5) * _dz));
  const std::complex<double> currMismatcho = std::exp(1._I * _diffBeta0[1] * ( i     * _dz));

  k1[1] = (prevPolDir * _nlStep[1]) * (prevInvMsmch  * prevP.conjugate()   *  prevS                + prevMismatcho * prevP   *  prevO.conjugate());
  k1[0] = (prevPolDir * _nlStep[0]  *  prevMismatch) * prevP               *  prevO;
  k2[1] = (intmPolDir * _nlStep[1]) * (intmInvMsmch  * interpP.conjugate() * (prevS + 0.5 * k1[0]) + intmMismatcho * interpP * (prevO + 0.5 * k1[1]).conjugate());
  k2[0] = (intmPolDir * _nlStep[0]  *  intmMismatch) * interpP             * (prevO + 0.5 * k1[1]);
  k3[1] = (intmPolDir * _nlStep[1]) * (intmInvMsmch  * interpP.conjugate() * (prevS + 0.5 * k2[0]) + intmMismatcho * interpP * (prevO + 0.5 * k2[1]).conjugate());
  k3[0] = (intmPolDir * _nlStep[0]  *  intmMismatch) * interpP             * (prevO + 0.5 * k2[1]);
  k4[1] = (currPolDir * _nlStep[1]) * (currInvMsmch  * currP.conjugate()   * (prevS + k3[0])       + currMismatcho * currP   * (prevO + k3[1]).conjugate());
  k4[0] = (currPolDir * _nlStep[0]  *  currMismatch) * currP               * (prevO + k3[1]);
}


Chi2PDCII::Chi2PDCII(double relativeLength, double nlLength, double nlLengthOrig, double nlLengthI, double dispLength,
                     double beta2, double beta2s, double beta2o, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                     double beta1, double beta1s, double beta1o, double beta3, double beta3s, double beta3o,
                     double diffBeta0, double diffBeta0o, double chirp, double rayleighLength,
                     double tMax, uint tPrecision, uint zPrecision, const Eigen::Ref<const Arrayd>& poling) :
  _Chi2::_Chi2(_nSignalModes, relativeLength, {nlLength, nlLengthOrig, nlLengthI}, dispLength, beta2, {beta2s, beta2o},
               customPump, pulseType, beta1, {beta1s, beta1o}, beta3, {beta3s, beta3o}, {diffBeta0, diffBeta0o},
               chirp, rayleighLength, tMax, tPrecision, zPrecision, poling) {}


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

  k1[1] =  prevPolDir * ((_nlStep[1] * prevMismatch) * prevP   *  prevS.conjugate()                + (_nlStep[2] * prevMismatcho) * prevP   *  prevO.conjugate());
  k1[0] = (prevPolDir *   _nlStep[0] * prevMismatch) * prevP   *  prevO.conjugate();
  k2[1] =  intmPolDir * ((_nlStep[1] * intmMismatch) * interpP * (prevS + 0.5 * k1[0]).conjugate() + (_nlStep[2] * intmMismatcho) * interpP * (prevO + 0.5 * k1[1]).conjugate());
  k2[0] = (intmPolDir *   _nlStep[0] * intmMismatch) * interpP * (prevO + 0.5 * k1[1]).conjugate();
  k3[1] =  intmPolDir * ((_nlStep[1] * intmMismatch) * interpP * (prevS + 0.5 * k2[0]).conjugate() + (_nlStep[2] * intmMismatcho) * interpP * (prevO + 0.5 * k2[1]).conjugate());
  k3[0] = (intmPolDir *   _nlStep[0] * intmMismatch) * interpP * (prevO + 0.5 * k2[1]).conjugate();
  k4[1] =  currPolDir * ((_nlStep[1] * currMismatch) * currP   * (prevS + k3[0]).conjugate()       + (_nlStep[2] * currMismatcho) * currP   * (prevO + k3[1]).conjugate());
  k4[0] = (currPolDir *   _nlStep[0] * currMismatch) * currP   * (prevO + k3[1]).conjugate();
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


void Cascade::setPump(int pulseType, double chirpLength) {
  if (sharedPump)
    media[0].get().setPump(pulseType, chirpLength);
  else {
    for (auto& medium : media)
      medium.get().setPump(pulseType, chirpLength);
  }
}


void Cascade::setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength) {
  if (sharedPump)
    media[0].get().setPump(customPump, chirpLength);
  else {
    for (auto& medium : media)
      medium.get().setPump(customPump, chirpLength);
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


std::pair<Array2Dcd, Array2Dcd> Cascade::computeGreensFunction(bool inTimeDomain, bool runPump, uint nThreads,
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
    auto CandS = medium.get().computeGreensFunction(inTimeDomain, false, nThreads);
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
