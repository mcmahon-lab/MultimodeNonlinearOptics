#include "NonlinearMedium.hpp"
#include <stdexcept>
#include <limits>


_NonlinearMedium::_NonlinearMedium(double relativeLength, double nlLength, double dispLength,
                                   double beta2, double beta2s, int pulseType,
                                   double beta1, double beta1s, double beta3, double beta3s,
                                   double chirp, double tMax, uint tPrecision, uint zPrecision) {
  setLengths(relativeLength, nlLength, dispLength, zPrecision);
  resetGrids(tPrecision, tMax);
  setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s);
  setPump(pulseType, chirp);
}

_NonlinearMedium::_NonlinearMedium(double relativeLength, double nlLength, double dispLength,
                                   double beta2, double beta2s, const Eigen::Ref<const Arraycd>& customPump,
                                   int pulseType, double beta1, double beta1s, double beta3, double beta3s,
                                   double chirp, double tMax, uint tPrecision, uint zPrecision) {
  setLengths(relativeLength, nlLength, dispLength, zPrecision);
  resetGrids(tPrecision, tMax);
  setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s);
  setPump(customPump, chirp);
}


void _NonlinearMedium::setLengths(double relativeLength, double nlLength, double dispLength, uint zPrecision) {
  // Equation defined in terms of dispersion and nonlinear lengh ratio N^2 = Lds / Lnl
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
  _Nsquared = _DS / _NL;
  if (_noDispersion) _Nsquared = 1;
  if (_noNonlinear)  _Nsquared = 0;

  // space resolution
  _nZSteps = static_cast<uint>(zPrecision * _z / std::min({1., _DS, _NL}));
  _dz = _z / _nZSteps;

  // helper values
  _nlStep = 1i * _Nsquared * _dz;

  // Reset grids -- skip during construction
//  try:
//    resetGrids();
//  except AttributeError:
//    pass;
}


void _NonlinearMedium::resetGrids(uint nFreqs, double tMax) {

  // time windowing and resolution
  if (nFreqs != 0)
    _nFreqs = nFreqs;
  if (tMax != 0)
    _tMax = tMax;

  if (nFreqs != 0 || tMax != 0) {
    auto Nt = _nFreqs;

    // time and frequency axes
    _tau = Eigen::VectorXd::LinSpaced(Nt, -tMax, tMax);
    _tau = fftshift(_tau);
    _omega = Eigen::VectorXd::LinSpaced(_nFreqs, -M_PI / _tMax * _nFreqs / 2, M_PI / _tMax * _nFreqs / 2);
    _omega = fftshift(_omega);

    // Reset dispersion and pulse
    //  try:
    //    setDispersion(_beta2, _beta2s, _beta1, _beta1, _beta3, _beta3s);
    //  except AttributeError:
    //    pass;
  }

  if (_nFreqs % 2 != 0 || _nFreqs == 0 || _nZSteps == 0)
    throw std::invalid_argument("Invalid PDE grid size");

  // Grids for PDE propagation
  pumpFreq.resize(_nZSteps, _nFreqs);
  pumpTime.resize(_nZSteps, _nFreqs);
  signalFreq.resize(_nZSteps, _nFreqs);
  signalTime.resize(_nZSteps, _nFreqs);

  fftTemp.resize(_nFreqs);
}


void _NonlinearMedium::setDispersion(double beta2, double beta2s, double beta1, double beta1s, double beta3, double beta3s) {

  // positive or negative dispersion for pump (ie should be +/- 1), relative dispersion for signal
  _beta2  = beta2;
  _beta2s = beta2s;

  if (std::abs(_beta2) != 1 && !_noDispersion)
    throw std::invalid_argument("Non unit beta2");

  // group velocity difference (relative to beta2 and pulse width)
  _beta1  = beta1;
  _beta1s = beta1s;
  _beta3  = beta3;
  _beta3s = beta3s;

  // dispersion profile
  if (_noDispersion) {
    _dispersionPump = 0;
    _dispersionSign = 0;
  }
  else {
    _dispersionPump = _omega * (beta1  + _omega * (0.5 * beta2  + _omega * _beta3  / 6));
    _dispersionSign = _omega * (beta1s + _omega * (0.5 * beta2s + _omega * _beta3s / 6));
  }

  // helper values
  _dispStepPump = (1i * _dispersionPump * _dz).exp();
  _dispStepSign = (1i * _dispersionSign * _dz).exp();
}

void _NonlinearMedium::setPump(int pulseType, double chirp) {
  // initial time domain envelopes (pick Gaussian or Soliton Hyperbolic Secant)
  if (pulseType)
    _env = 1 / _tau.cosh() * (-0.5i * chirp * _tau.square()).exp();
  else
    _env = (-0.5 * _tau.square() * (1 + 1i * chirp)).exp();
}

void _NonlinearMedium::setPump(const Eigen::Ref<const Arraycd>& customPump, double chirp) {
  // custom initial time domain envelope
  if (customPump.size() != _nFreqs)
    throw std::invalid_argument("Custom pump array length does not match number of frequency/time bins");
  _env = customPump * (-0.5i * chirp * _tau.square()).exp();
}


std::pair<Array2Dcd, Array2Dcd> _NonlinearMedium::computeGreensFunction(bool inTimeDomain, bool runPump) {
  // Green function matrices
  Array2Dcd greenC;
  Array2Dcd greenS;
  greenC.setZero(_nFreqs, _nFreqs);
  greenS.setZero(_nFreqs, _nFreqs);

  if (runPump) runPumpSimulation();

  auto& grid = inTimeDomain ? signalTime : signalFreq;

  // Calculate Green's functions with real and imaginary impulse response
  for (uint i = 0; i < _nFreqs; i++) {
    grid.row(0) = 0;
    grid(0, i) = 1;
    runSignalSimulation(grid.row(0), inTimeDomain);

    greenC.row(i) += grid.row(_nZSteps - 1) * 0.5;
    greenS.row(i) += grid.row(_nZSteps - 1) * 0.5;

    grid.row(0) = 0;
    grid(0, i) = 1i;
    runSignalSimulation(grid.row(0), inTimeDomain);

    greenC.row(i) -= grid.row(_nZSteps - 1) * 0.5i;
    greenS.row(i) += grid.row(_nZSteps - 1) * 0.5i;
  }

  greenC.transposeInPlace();
  greenS.transposeInPlace();

  greenC = fftshift(greenC);
  greenS = fftshift(greenS);

  return std::make_pair(std::move(greenC), std::move(greenS));
}


inline const RowVectorcd& _NonlinearMedium::fft(const RowVectorcd& input) {
  fftObj.fwd(fftTemp, input);
  return fftTemp;
}

inline const RowVectorcd& _NonlinearMedium::ifft(const RowVectorcd& input) {
  fftObj.inv(fftTemp, input);
  return fftTemp;
}

inline Arrayf _NonlinearMedium::fftshift(const Arrayf& input) {
  Arrayf out(input.rows(), input.cols());
  auto half = input.cols() / 2;
  out.head(half) = input.tail(half);
  out.tail(half) = input.head(half);
  return out;
}


inline Array2Dcd _NonlinearMedium::fftshift(const Array2Dcd& input) {
  Array2Dcd out(input.rows(), input.cols());

  auto halfCols = input.cols() / 2;
  auto halfRows = input.rows() / 2;

  out.topLeftCorner(halfRows, halfCols) = input.bottomRightCorner(halfRows, halfCols);
  out.topRightCorner(halfRows, halfCols) = input.bottomLeftCorner(halfRows, halfCols);
  out.bottomLeftCorner(halfRows, halfCols) = input.topRightCorner(halfRows, halfCols);
  out.bottomRightCorner(halfRows, halfCols) = input.topLeftCorner(halfRows, halfCols);
  return out;
}


void Chi3::runPumpSimulation() {

  pumpFreq.row(0) = fft(_env).array() * (0.5i * _dispersionPump * _dz).exp();
  pumpTime.row(0) = ifft(pumpFreq.row(0));

  Arraycd temp(_nFreqs);
  for (uint i = 1; i < _nZSteps; i++) {
    temp = pumpTime.row(i-1) * (_nlStep * pumpTime.row(i-1).abs2()).exp();
    pumpFreq.row(i) = fft(temp).array() * _dispStepPump;
    pumpTime.row(i) = ifft(pumpFreq.row(i));
  }

  pumpFreq.row(_nZSteps-1) *= (-0.5i * _dispersionPump * _dz).exp();
  pumpTime.row(_nZSteps-1)  = ifft(pumpFreq.row(_nZSteps-1));
}


void Chi3::runSignalSimulation(const Arraycd& inputProf, bool timeSignal) {
  if (timeSignal)
    signalFreq.row(0) = fft(inputProf).array() * (0.5i * _dispersionSign * _dz).exp();
  else
    signalFreq.row(0) = inputProf * (0.5i * _dispersionSign * _dz).exp();
  signalTime.row(0) = ifft(signalFreq.row(0));

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
    signalFreq.row(i) = fft(temp).array() * _dispStepSign;
    signalTime.row(i) = ifft(signalFreq.row(i));
  }

  signalFreq.row(_nZSteps-1) *= (-0.5i * _dispersionSign * _dz).exp();
  signalTime.row(_nZSteps-1)  = ifft(signalFreq.row(_nZSteps-1));
}


void Chi2::runPumpSimulation() {

  pumpFreq.row(0) = fft(_env);
  pumpTime.row(0) = _env;

  for (uint i = 1; i < _nZSteps; i++) {
    pumpFreq.row(i) = pumpFreq.row(0) * (1i * i * _dispersionPump * _dz).exp();
    pumpTime.row(i) = ifft(pumpFreq.row(i));
  }
}


void Chi2::runSignalSimulation(const Arraycd& inputProf, bool timeSignal) {
  if (timeSignal)
    signalFreq.row(0) = fft(inputProf).array() * (0.5i * _dispersionSign * _dz).exp();
  else
    signalFreq.row(0) = inputProf * (0.5i * _dispersionSign * _dz).exp();
  signalTime.row(0) = ifft(signalFreq.row(0));

  Arraycd interpP(_nFreqs), k1(_nFreqs), k2(_nFreqs), k3(_nFreqs), k4(_nFreqs), temp(_nFreqs);
  for (uint i = 1; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the non-linear propagation
    const auto& prev =  signalTime.row(i-1);
    const auto& prevP = pumpTime.row(i-1);
    const auto& currP = pumpTime.row(i);

    interpP = 0.5 * (prevP + currP);

    k1 = _nlStep * prevP   *  prev.conjugate();
    k2 = _nlStep * interpP * (prev + 0.5 * k1).conjugate();
    k3 = _nlStep * interpP * (prev + 0.5 * k2).conjugate();
    k4 = _nlStep * currP   * (prev + k3).conjugate();

    temp = signalTime.row(i-1) + (k1 + 2 * k2 + 2 * k3 + k4) / 6;

    // Dispersion step
    signalFreq.row(i) = fft(temp).array() * _dispStepSign;
    signalTime.row(i) = ifft(signalFreq.row(i));
  }

  signalFreq.row(_nZSteps-1) *= (-0.5i * _dispersionSign * _dz).exp();
  signalTime.row(_nZSteps-1)  = ifft(signalFreq.row(_nZSteps-1));
}

