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


_NonlinearMedium::_NonlinearMedium(double relativeLength, double nlLength, double dispLength,
                                   double beta2, double beta2s, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                                   double beta1, double beta1s, double beta3, double beta3s, double diffBeta0,
                                   double chirp, double tMax, uint tPrecision, uint zPrecision) {
  setLengths(relativeLength, nlLength, dispLength, zPrecision);
  resetGrids(tPrecision, tMax);
  setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s, diffBeta0);
  if (customPump.size() != 0)
    setPump(customPump, chirp);
  else
    setPump(pulseType, chirp);
}


void _NonlinearMedium::setLengths(double relativeLength, double nlLength, double dispLength, uint zPrecision) {
  // Equation defined in terms of dispersion and nonlinear lengh ratio N^2 = L_ds / L_nl
  // The length z is given in units of dispersion length (of pump)
  // The time is given in units of initial width (of pump)

  // Total length (in units of pump dispersion length or, if infinite, nonlinear length)
  _z = relativeLength;

  // Nonlinear and dispersion length scales of the pump
  // DS should keep fixed at 1 to not mess up z unless no dispersion
  _DS = dispLength;

  // if no dispersion keep NL fixed at 1 to not mess up z, otherwise change relative to DS
  _NL = nlLength;

  _noDispersion = _DS == std::numeric_limits<double>::infinity() ? true : false;
  _noNonlinear =  _NL == std::numeric_limits<double>::infinity() ? true : false;

  if (_noDispersion) {
    if (_NL != 1) throw std::invalid_argument("Non unit NL");
  }
  else
    if (_DS != 1) throw std::invalid_argument("Non unit DS");

  // Soliton order
  double _Nsquared = _DS / _NL;
  if (_noDispersion) _Nsquared = 1;
  if (_noNonlinear)  _Nsquared = 0;

  // space resolution
  _nZSteps = static_cast<uint>(zPrecision * _z / std::min({1., _DS, _NL}));
  _dz = _z / _nZSteps;

  // helper values
  _nlStep = 1._I * _Nsquared * _dz;
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
  signalFreq.resize(_nZSteps, _nFreqs);
  signalTime.resize(_nZSteps, _nFreqs);
}


void _NonlinearMedium::setDispersion(double beta2, double beta2s, double beta1, double beta1s,
                                     double beta3, double beta3s, double diffBeta0) {

  // positive or negative dispersion for pump (ie should be +/- 1), relative dispersion for signal
  _beta2  = beta2;
  _beta2s = beta2s;

  if (std::abs(beta2) != 1 && !_noDispersion)
    throw std::invalid_argument("Non unit beta2");

  // group velocity difference (relative to beta2 and pulse width)
  _beta1  = beta1;
  _beta1s = beta1s;
  _beta3  = beta3;
  _beta3s = beta3s;

  // signal's phase mis-match
  _diffBeta0 = diffBeta0;

  // dispersion profile
  if (_noDispersion) {
    _dispersionPump.setZero(_nFreqs);
    _dispersionSign.setZero(_nFreqs);
  }
  else {
    _dispersionPump = _omega * (beta1  + _omega * (0.5 * beta2  + _omega * beta3  / 6));
    _dispersionSign = _omega * (beta1s + _omega * (0.5 * beta2s + _omega * beta3s / 6));
  }

  // helper values
  _dispStepPump = (1._I * _dispersionPump * _dz).exp();
  _dispStepSign = (1._I * _dispersionSign * _dz).exp();
}


void _NonlinearMedium::setPump(int pulseType, double chirp) {
  // initial time domain envelopes (pick Gaussian, Hyperbolic Secant, Sinc)
  if (pulseType == 1)
    _env = 1 / _tau.cosh() * (-0.5_I * chirp * _tau.square()).exp();
  else if (pulseType == 2) {
    _env = _tau.sin() / _tau * (-0.5_I * chirp * _tau.square()).exp();
    _env(0) = 1;
  }
  else
    _env = (-0.5 * _tau.square() * (1. + 1._I * chirp)).exp();
}

void _NonlinearMedium::setPump(const Eigen::Ref<const Arraycd>& customPump, double chirp) {
  // custom initial time domain envelope
  if (customPump.size() != _nFreqs)
    throw std::invalid_argument("Custom pump array length does not match number of frequency/time bins");
  _env = customPump * (-0.5_I * chirp * _tau.square()).exp();
}


void _NonlinearMedium::runSignalSimulation(Eigen::Ref<const Arraycd> inputProf, bool inTimeDomain) {
  if (inputProf.size() != _nFreqs)
    throw std::invalid_argument("inputProf array size does not match number of frequency/time bins");
  runSignalSimulation(inputProf, inTimeDomain, signalFreq, signalTime);
}


std::pair<Array2Dcd, Array2Dcd> _NonlinearMedium::computeGreensFunction(bool inTimeDomain, bool runPump, uint nThreads) {

  if (nThreads > _nFreqs)
    throw std::invalid_argument("Too many threads requested!");

  if (runPump) runPumpSimulation();

  // Green function matrices -- Note: hopefully large enough to avoid dirtying cache?
  Array2Dcd greenC;
  Array2Dcd greenS;
  greenC.setZero(_nFreqs, _nFreqs);
  greenS.setZero(_nFreqs, _nFreqs);

  // run n-1 separate threads, run part on this process
  std::vector<std::thread> threads;
  threads.reserve(nThreads - 1);

  // Each thread needs a separate computation grid to avoid interfering with other threads
  std::vector<Array2Dcd> grids(2 * (nThreads - 1));

  // Calculate Green's functions with real and imaginary impulse response
  auto calcGreensPart = [&, inTimeDomain](Array2Dcd& gridFreq, Array2Dcd& gridTime, uint start, uint stop) {
    if (gridFreq.size() == 0) gridFreq.resize(_nZSteps, _nFreqs);
    if (gridTime.size() == 0) gridTime.resize(_nZSteps, _nFreqs);
    auto& grid = inTimeDomain ? gridTime : gridFreq;

    for (uint i = start; i < stop; i++) {

      grid.row(0) = 0;
      grid(0, i) = 1;
      runSignalSimulation(grid.row(0), inTimeDomain, gridFreq, gridTime);

      greenC.row(i) += 0.5 * grid.bottomRows<1>();
      greenS.row(i) += 0.5 * grid.bottomRows<1>();

      grid.row(0) = 0;
      grid(0, i) = 1._I;
      runSignalSimulation(grid.row(0), inTimeDomain, gridFreq, gridTime);

      greenC.row(i) -= 0.5_I * grid.bottomRows<1>();
      greenS.row(i) += 0.5_I * grid.bottomRows<1>();
    }
  };

  for (uint i = 1; i < nThreads; i++) {
    threads.emplace_back(calcGreensPart, std::ref(grids[2*i-2]), std::ref(grids[2*i-1]),
                         (i * _nFreqs) / nThreads, ((i + 1) * _nFreqs) / nThreads);
  }
  calcGreensPart(signalFreq, signalTime, 0, _nFreqs / nThreads);
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }

  greenC.transposeInPlace();
  greenS.transposeInPlace();

  greenC = fftshift2(greenC);
  greenS = fftshift2(greenS);

  return std::make_pair(std::move(greenC), std::move(greenS));
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


Chi3::Chi3(double relativeLength, double nlLength, double dispLength,
           double beta2, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
           double beta3, double chirp, double tMax, uint tPrecision, uint zPrecision) :
    _NonlinearMedium(relativeLength, nlLength, dispLength, beta2, beta2, customPump, pulseType,
                     0, 0, beta3, beta3, 0, chirp, tMax, tPrecision, zPrecision)
{}


void Chi3::runPumpSimulation() {
  RowVectorcd fftTemp(_nFreqs);

  FFTtimes(pumpFreq.row(0), _env, ((0.5_I * _dz) * _dispersionPump).exp())
  IFFT(pumpTime.row(0), pumpFreq.row(0))

  Arraycd temp(_nFreqs);
  for (uint i = 1; i < _nZSteps; i++) {
    temp = pumpTime.row(i-1) * (_nlStep * pumpTime.row(i-1).abs2()).exp();
    FFTtimes(pumpFreq.row(i), temp, _dispStepPump)
    IFFT(pumpTime.row(i), pumpFreq.row(i))
  }

  pumpFreq.row(_nZSteps-1) *= ((-0.5_I * _dz) * _dispersionPump).exp();
  IFFT(pumpTime.row(_nZSteps-1), pumpFreq.row(_nZSteps-1))
}


void Chi3::runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain,
                               Array2Dcd& signalFreq, Array2Dcd& signalTime) {
  RowVectorcd fftTemp(_nFreqs);

  if (inTimeDomain)
    FFTtimes(signalFreq.row(0), inputProf, ((0.5_I * _dz) * _dispersionSign).exp())
  else
    signalFreq.row(0) = inputProf * ((0.5_I * _dz) * _dispersionSign).exp();
  IFFT(signalTime.row(0), signalFreq.row(0))

  Arraycd interpP(_nFreqs), k1(_nFreqs), k2(_nFreqs), k3(_nFreqs), k4(_nFreqs), temp(_nFreqs);
  for (uint i = 1; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the non-linear propagation
    const auto& prev  = signalTime.row(i-1);
    const auto& prevP = pumpTime.row(i-1);
    const auto& currP = pumpTime.row(i);

    interpP = 0.5 * (prevP + currP);

    k1 = _nlStep * (2 * prevP.abs2()   *  prev             + prevP.square()   *  prev.conjugate());
    k2 = _nlStep * (2 * interpP.abs2() * (prev + 0.5 * k1) + interpP.square() * (prev + 0.5 * k1).conjugate());
    k3 = _nlStep * (2 * interpP.abs2() * (prev + 0.5 * k2) + interpP.square() * (prev + 0.5 * k2).conjugate());
    k4 = _nlStep * (2 * currP.abs2()   * (prev + k3)       + currP.square()   * (prev + k3).conjugate());

    temp = signalTime.row(i-1) + (k1 + 2 * k2 + 2 * k3 + k4) / 6;

    // Dispersion step
    FFTtimes(signalFreq.row(i), temp, _dispStepSign)
    IFFT(signalTime.row(i), signalFreq.row(i))
  }

  signalFreq.row(_nZSteps-1) *= ((-0.5_I * _dz) * _dispersionSign).exp();
  IFFT(signalTime.row(_nZSteps-1), signalFreq.row(_nZSteps-1))
}


_Chi2::_Chi2(double relativeLength, double nlLength, double dispLength, double beta2, double beta2s,
             const Eigen::Ref<const Arraycd>& customPump, int pulseType,
             double beta1, double beta1s, double beta3, double beta3s, double diffBeta0,
             double chirp, double tMax, uint tPrecision, uint zPrecision,
             const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium::_NonlinearMedium(relativeLength, nlLength, dispLength, beta2, beta2s, customPump, pulseType,
                                     beta1, beta1s, beta3, beta3s, diffBeta0, chirp, tMax, tPrecision, zPrecision)
{
  setPoling(poling);
}


void _Chi2::runPumpSimulation() {
  RowVectorcd fftTemp(_nFreqs);

  FFT(pumpFreq.row(0), _env)
  pumpTime.row(0) = _env;

  for (uint i = 1; i < _nZSteps; i++) {
    pumpFreq.row(i) = pumpFreq.row(0) * (1._I * (i * _dz) * _dispersionPump).exp();
    IFFT(pumpTime.row(i), pumpFreq.row(i))
  }
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


void Chi2PDC::runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain,
                                  Array2Dcd& signalFreq, Array2Dcd& signalTime) {
  RowVectorcd fftTemp(_nFreqs);

  if (inTimeDomain)
    FFTtimes(signalFreq.row(0), inputProf, ((0.5_I * _dz) * _dispersionSign).exp())
  else
    signalFreq.row(0) = inputProf * ((0.5_I * _dz) * _dispersionSign).exp();
  IFFT(signalTime.row(0), signalFreq.row(0))

  Arraycd interpP(_nFreqs), k1(_nFreqs), k2(_nFreqs), k3(_nFreqs), k4(_nFreqs), temp(_nFreqs);
  for (uint i = 1; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the non-linear propagation
    const auto& prev =  signalTime.row(i-1);
    const auto& prevP = pumpTime.row(i-1);
    const auto& currP = pumpTime.row(i);
    interpP = 0.5 * (prevP + currP);

    const double prevPolDir = _poling(i-1);
    const double currPolDir = _poling(i);
    const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

    const std::complex<double> mismatch = std::exp(1._I * _diffBeta0 * (i * _dz));

    k1 = (prevPolDir * _nlStep * mismatch) * prevP   *  prev.conjugate();
    k2 = (intmPolDir * _nlStep * mismatch) * interpP * (prev + 0.5 * k1).conjugate();
    k3 = (intmPolDir * _nlStep * mismatch) * interpP * (prev + 0.5 * k2).conjugate();
    k4 = (currPolDir * _nlStep * mismatch) * currP   * (prev + k3).conjugate();

    temp = signalTime.row(i-1) + (k1 + 2 * k2 + 2 * k3 + k4) / 6;

    // Dispersion step
    FFTtimes(signalFreq.row(i), temp, _dispStepSign)
    IFFT(signalTime.row(i), signalFreq.row(i))
  }

  signalFreq.row(_nZSteps-1) *= ((-0.5_I * _dz) * _dispersionSign).exp();
  IFFT(signalTime.row(_nZSteps-1), signalFreq.row(_nZSteps-1))
}


Chi2SFG::Chi2SFG(double relativeLength, double nlLength, double nlLengthOrig, double dispLength,
                 double beta2, double beta2s, double beta2o, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                 double beta1, double beta1s, double beta1o, double beta3, double beta3s, double beta3o,
                 double diffBeta0, double diffBeta0o, double chirp, double tMax, uint tPrecision, uint zPrecision,
                 const Eigen::Ref<const Arrayd>& poling)
{
  setLengths(relativeLength, nlLength, nlLengthOrig, dispLength, zPrecision);
  resetGrids(tPrecision, tMax);
  setDispersion(beta2, beta2s, beta2o, beta1, beta1s, beta1o, beta3, beta3s, beta3o, diffBeta0, diffBeta0o);
  if (customPump.size() != 0)
    setPump(customPump, chirp);
  else
    setPump(pulseType, chirp);
  setPoling(poling);
}


void Chi2SFG::setLengths(double relativeLength, double nlLength, double nlLengthOrig,
                         double dispLength, uint zPrecision) {
  _NonlinearMedium::setLengths(relativeLength, nlLength, dispLength, zPrecision);

  _NLo = nlLengthOrig;

  if (_noDispersion)
    _nlStepO = 1._I * _NL / nlLengthOrig * _dz;
  else if (_noNonlinear)
    _nlStepO = 0;
  else
    _nlStepO = 1._I * _DS / nlLengthOrig * _dz;
}


void Chi2SFG::resetGrids(uint nFreqs, double tMax) {
  _NonlinearMedium::resetGrids(nFreqs, tMax);
  originalFreq.resize(_nZSteps, _nFreqs);
  originalTime.resize(_nZSteps, _nFreqs);
}


void Chi2SFG::setDispersion(double beta2, double beta2s, double beta2o, double beta1, double beta1s, double beta1o,
                            double beta3, double beta3s, double beta3o, double diffBeta0, double diffBeta0o) {
  _NonlinearMedium::setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s, diffBeta0);
  _beta2o = beta2o;
  _beta1o = beta1o;
  _beta3o = beta3o;
  _diffBeta0o = diffBeta0o;

  if (_noDispersion)
    _dispersionOrig.setZero(_nFreqs);
  else
    _dispersionOrig = _omega * (beta1o + _omega * (0.5 * beta2o + _omega * beta3o / 6));

  _dispStepOrig = (1._I * _dispersionOrig * _dz).exp();
}


void Chi2SFG::runSignalSimulation(Eigen::Ref<const Arraycd> inputProf, bool inTimeDomain) {
  if (inputProf.size() != _nFreqs && inputProf.size() != 2 * _nFreqs)
    throw std::invalid_argument("inputProf array size does not match number of frequency/time bins");
  runSignalSimulation(inputProf, inTimeDomain, signalFreq, signalTime);
}


void Chi2SFG::runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain,
                                  Array2Dcd& signalFreq, Array2Dcd& signalTime) {
  RowVectorcd fftTemp(_nFreqs);

  // Hack: If we are using grids that are the member variables of the class, then proceed normally.
  // However if called from computeGreensFunction we need a workaround to use only one grid.
  const bool usingMemberGrids = (&signalFreq == &this->signalFreq);
  const uint O = usingMemberGrids? 0 : _nZSteps; // offset
  Array2Dcd& originalFreq = usingMemberGrids? this->originalFreq : signalFreq;
  Array2Dcd& originalTime = usingMemberGrids? this->originalTime : signalTime;
  if (!usingMemberGrids) {
    signalFreq.resize(2 * _nZSteps, _nFreqs);
    signalTime.resize(2 * _nZSteps, _nFreqs);
  }

  // Takes as input the signal in the first frequency and outputs in the second frequency
  if (inputProf.size() == _nFreqs) {
    if (inTimeDomain)
      FFTtimes(originalFreq.row(0), inputProf, ((0.5_I * _dz) * _dispersionOrig).exp())
    else
      originalFreq.row(0) = inputProf * ((0.5_I * _dz) * _dispersionOrig).exp();
    IFFT(originalTime.row(0), originalFreq.row(0))

    signalFreq.row(O) = 0;
    signalTime.row(O) = 0;
  }
  // input array spanning both frequencies, ordered as signal then original
  else if (inputProf.size() == 2 * _nFreqs) {
    if (inTimeDomain) {
      const Arraycd& inputSignal = inputProf.head(_nFreqs);
      FFTtimes(signalFreq.row(O), inputSignal, ((0.5_I * _dz) * _dispersionSign).exp())
      const Arraycd& inputOriginal = inputProf.tail(_nFreqs);
      FFTtimes(originalFreq.row(0), inputOriginal, ((0.5_I * _dz) * _dispersionOrig).exp())
    }
    else {
      signalFreq.row(O)   = inputProf.head(_nFreqs) * ((0.5_I * _dz) * _dispersionSign).exp();
      originalFreq.row(0) = inputProf.tail(_nFreqs) * ((0.5_I * _dz) * _dispersionOrig).exp();
    }
    IFFT(signalTime.row(O),   signalFreq.row(O))
    IFFT(originalTime.row(0), originalFreq.row(0))
  }
  else {
    throw std::invalid_argument("inputProf array size does not match number of frequency/time bins");
  }

  Arraycd interpP(_nFreqs);
  Arraycd k1(_nFreqs), k2(_nFreqs), k3(_nFreqs), k4(_nFreqs), tempOrig(_nFreqs);
  Arraycd l1(_nFreqs), l2(_nFreqs), l3(_nFreqs), l4(_nFreqs), tempSign(_nFreqs);
  for (uint i = 1; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the non-linear propagation
    const auto& prevS = signalTime.row(O+i-1);
    const auto& prevO = originalTime.row(i-1);
    const auto& prevP = pumpTime.row(i-1);
    const auto& currP = pumpTime.row(i);

    interpP = 0.5 * (prevP + currP);

    const double prevPolDir = _poling(i-1);
    const double currPolDir = _poling(i);
    const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

    const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0 * ((i- 1) * _dz));
    const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0 * ((i-.5) * _dz));
    const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0 * ( i     * _dz));
    const std::complex<double> prevInvMsmch = 1. / prevMismatch;
    const std::complex<double> intmInvMsmch = 1. / intmMismatch;
    const std::complex<double> currInvMsmch = 1. / currMismatch;
    const std::complex<double> prevMismatcho = std::exp(1._I * _diffBeta0o * ((i- 1) * _dz));
    const std::complex<double> intmMismatcho = std::exp(1._I * _diffBeta0o * ((i-.5) * _dz));
    const std::complex<double> currMismatcho = std::exp(1._I * _diffBeta0o * ( i     * _dz));

    k1 = (prevPolDir * _nlStepO) * (prevInvMsmch  * prevP.conjugate()   *  prevS             + prevMismatcho * prevP   *  prevO.conjugate());
    l1 = (prevPolDir * _nlStep   *  prevMismatch) * prevP               *  prevO;
    k2 = (intmPolDir * _nlStepO) * (intmInvMsmch  * interpP.conjugate() * (prevS + 0.5 * l1) + intmMismatcho * interpP * (prevO + 0.5 * k1).conjugate());
    l2 = (intmPolDir * _nlStep   *  intmMismatch) * interpP             * (prevO + 0.5 * k1);
    k3 = (intmPolDir * _nlStepO) * (intmInvMsmch  * interpP.conjugate() * (prevS + 0.5 * l2) + intmMismatcho * interpP * (prevO + 0.5 * k2).conjugate());
    l3 = (intmPolDir * _nlStep   *  intmMismatch) * interpP             * (prevO + 0.5 * k2);
    k4 = (currPolDir * _nlStepO) * (currInvMsmch  * currP.conjugate()   * (prevS + l3)       + currMismatcho * currP   * (prevO + k3).conjugate());
    l4 = (currPolDir * _nlStep   *  currMismatch) * currP               * (prevO + k3);

    tempOrig = originalTime.row(i-1) + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    tempSign = signalTime.row(O+i-1) + (l1 + 2 * l2 + 2 * l3 + l4) / 6;

    // Dispersion step
    FFTtimes(signalFreq.row(O+i), tempSign, _dispStepSign)
    IFFT(signalTime.row(O+i), signalFreq.row(O+i))

    FFTtimes(originalFreq.row(i), tempOrig, _dispStepOrig)
    IFFT(originalTime.row(i), originalFreq.row(i))
  }

  signalFreq.row(O+_nZSteps-1) *= ((-0.5_I * _dz) * _dispersionSign).exp();
  IFFT(signalTime.row(O+_nZSteps-1), signalFreq.row(O+_nZSteps-1))

  originalFreq.row(_nZSteps-1) *= ((-0.5_I * _dz) * _dispersionOrig).exp();
  IFFT(originalTime.row(_nZSteps-1), originalFreq.row(_nZSteps-1))
}


std::pair<Array2Dcd, Array2Dcd> Chi2SFG::computeTotalGreen(bool inTimeDomain, bool runPump, uint nThreads) {

  if (nThreads > 2 * _nFreqs)
    throw std::invalid_argument("Too many threads requested!");

  if (runPump) runPumpSimulation();

  // Green function matrices -- Note: hopefully large enough to avoid dirtying cache?
  Array2Dcd greenC;
  Array2Dcd greenS;
  greenC.setZero(2 * _nFreqs, 2 * _nFreqs);
  greenS.setZero(2 * _nFreqs, 2 * _nFreqs);

  // run n-1 separate threads, run part on this process
  std::vector<std::thread> threads;
  threads.reserve(nThreads - 1);
  // Each thread needs a separate computation grid to avoid interfering with other threads
  std::vector<Array2Dcd> grids(2 * (nThreads - 1));

  // Calculate Green's functions with real and imaginary impulse response
  // Signal frequency comes first in the matrix, original frequency second
  auto calcGreensPart = [&, inTimeDomain](Array2Dcd& gridFreq, Array2Dcd& gridTime, uint start, uint stop) {

    const bool usingMemberGrids = (start == 0);
    if (!usingMemberGrids) {
      gridTime.resize(2 * _nZSteps, _nFreqs);
      gridFreq.resize(2 * _nZSteps, _nFreqs);
    }
    const auto& outputOriginal = usingMemberGrids? (inTimeDomain? originalTime.bottomRows<1>() : originalFreq.bottomRows<1>()) :
                                                   (inTimeDomain? gridTime.row(_nZSteps-1)     : gridFreq.row(_nZSteps-1));
    const auto& outputSignal   = usingMemberGrids? (inTimeDomain? signalTime.bottomRows<1>() : signalFreq.bottomRows<1>()) :
                                                   (inTimeDomain? gridTime.bottomRows<1>()   : gridFreq.bottomRows<1>());

    Arraycd impulse;
    impulse.setZero(2 * _nFreqs);

    for (uint i = start; i < stop; i++) {
      impulse(i) = 1;
      runSignalSimulation(impulse, inTimeDomain, gridFreq, gridTime);

      greenC.row(i).head(_nFreqs) += 0.5 * outputSignal;
      greenC.row(i).tail(_nFreqs) += 0.5 * outputOriginal;
      greenS.row(i).head(_nFreqs) += 0.5 * outputSignal;
      greenS.row(i).tail(_nFreqs) += 0.5 * outputOriginal;

      impulse(i) = 1._I;
      runSignalSimulation(impulse, inTimeDomain, gridFreq, gridTime);

      greenC.row(i).head(_nFreqs) -= (0.5_I) * outputSignal.bottomRows<1>();
      greenC.row(i).tail(_nFreqs) -= (0.5_I) * outputOriginal.bottomRows<1>();
      greenS.row(i).head(_nFreqs) += (0.5_I) * outputSignal.bottomRows<1>();
      greenS.row(i).tail(_nFreqs) += (0.5_I) * outputOriginal.bottomRows<1>();

      impulse(i) = 0;
    }
  };

  for (uint i = 1; i < nThreads; i++) {
    threads.emplace_back(calcGreensPart, std::ref(grids[2*i-2]), std::ref(grids[2*i-1]),
                         (i * 2 * _nFreqs) / nThreads, ((i + 1) * 2 * _nFreqs) / nThreads);
  }
  calcGreensPart(signalFreq, signalTime, 0, (2 * _nFreqs) / nThreads);
  for (auto& thread : threads)
    if (thread.joinable()) thread.join();

  greenC.transposeInPlace();
  greenS.transposeInPlace();

  // Need to fftshift each frequency block
  greenC.topLeftCorner(_nFreqs, _nFreqs) = fftshift2(greenC.topLeftCorner(_nFreqs, _nFreqs));
  greenS.topLeftCorner(_nFreqs, _nFreqs) = fftshift2(greenS.topLeftCorner(_nFreqs, _nFreqs));

  greenC.topRightCorner(_nFreqs, _nFreqs) = fftshift2(greenC.topRightCorner(_nFreqs, _nFreqs));
  greenS.topRightCorner(_nFreqs, _nFreqs) = fftshift2(greenS.topRightCorner(_nFreqs, _nFreqs));

  greenC.bottomLeftCorner(_nFreqs, _nFreqs) = fftshift2(greenC.bottomLeftCorner(_nFreqs, _nFreqs));
  greenS.bottomLeftCorner(_nFreqs, _nFreqs) = fftshift2(greenS.bottomLeftCorner(_nFreqs, _nFreqs));

  greenC.bottomRightCorner(_nFreqs, _nFreqs) = fftshift2(greenC.bottomRightCorner(_nFreqs, _nFreqs));
  greenS.bottomRightCorner(_nFreqs, _nFreqs) = fftshift2(greenS.bottomRightCorner(_nFreqs, _nFreqs));

  return std::make_pair(std::move(greenC), std::move(greenS));
}


Cascade::Cascade(bool sharePump, const std::vector<std::reference_wrapper<_NonlinearMedium>>& inputMedia) {

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

  sharedPump = sharePump;
}


void Cascade::addMedium(_NonlinearMedium& medium) {
  if (medium._nFreqs != _nFreqs or medium._tMax != _tMax)
    throw std::invalid_argument("Medium does not have same time and frequency axes as the first");

  media.emplace_back(medium);
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


void Cascade::runSignalSimulation(Eigen::Ref<const Arraycd> inputProf, bool inTimeDomain) {
  if (inputProf.size() != _nFreqs)
    throw std::invalid_argument("inputProf array size does not match number of frequency/time bins");

  media[0].get().runSignalSimulation(inputProf, inTimeDomain);
  for (uint i = 1; i < media.size(); i++) {
    media[i].get().runSignalSimulation(media[i-1].get().signalFreq.bottomRows<1>(), false);
  }
}


std::pair<Array2Dcd, Array2Dcd> Cascade::computeGreensFunction(bool inTimeDomain, bool runPump, uint nThreads) {
  // Green function matrices
  Array2Dcd greenC;
  Array2Dcd greenS;
  greenC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(_nFreqs, _nFreqs);
  greenS.setZero(_nFreqs, _nFreqs);

  Array2Dcd tempC, tempS;
  for (auto& medium : media) {
    auto CandS = medium.get().computeGreensFunction(inTimeDomain, runPump, nThreads);
    tempC = std::get<0>(CandS).matrix() * greenC.matrix() + std::get<1>(CandS).matrix() * greenS.conjugate().matrix();
    tempS = std::get<0>(CandS).matrix() * greenS.matrix() + std::get<1>(CandS).matrix() * greenC.conjugate().matrix();
    greenC.swap(tempC);
    greenS.swap(tempS);
  }

  return std::make_pair(std::move(greenC), std::move(greenS));
}
