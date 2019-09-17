#include "NonlinearMedium.hpp"
#include <stdexcept>
#include <limits>

_NonlinearMedium::_NonlinearMedium(float relativeLength, float nlLength, float dispLength,
                                   float beta2, float beta2s, int pulseType,
                                   float beta1, float beta1s, float beta3, float beta3s,
                                   float chirp, float tMax, uint tPrecision, uint zPrecision) {
  setLengths(relativeLength, nlLength, dispLength, zPrecision);
  resetGrids(tPrecision, tMax);
  setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s);
  setPump(pulseType, chirp);
}


void _NonlinearMedium::setLengths(float relativeLength, float nlLength, float dispLength, uint zPrecision) {
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

  _noDispersion = _DS == std::numeric_limits<float>::infinity() ? true : false;
  _noNonlinear =  _NL == std::numeric_limits<float>::infinity() ? true : false;

  if (_noDispersion)
    if (_NL != 1) throw std::invalid_argument("Non unit NL");
  else
    if (_DS != 1) throw std::invalid_argument("Non unit DS");

  // Soliton order
  _Nsquared = _DS / _NL;
  if (_noDispersion) _Nsquared = 1;
  if (_noNonlinear)  _Nsquared = 0;

  // space resolution
  _nZSteps = static_cast<uint>(zPrecision * _z / std::min({1.f, _DS, _NL}));
  _dz = _z / _nZSteps;

  // helper values
  _nlStep = 1i * _Nsquared * _dz;

  // Reset grids -- skip during construction
//  try:
//    resetGrids();
//  except AttributeError:
//    pass;
}


void _NonlinearMedium::resetGrids(uint nFreqs, float tMax) {

  // time windowing and resolution
  if (nFreqs != 0)
    _nFreqs = nFreqs;
  if (tMax != 0)
    _tMax = tMax;

  if (nFreqs != 0 || tMax != 0) {
    auto Nt = _nFreqs;
    float dt = 2 * _tMax / Nt;

    // time and frequency axes
    _tau = Eigen::VectorXf::LinSpaced(Nt, -tMax, tMax);
    _tau = fftshift(_tau);
    _omega = Eigen::VectorXf::LinSpaced(_nFreqs, -M_PI / _tMax * _nFreqs / 2, M_PI / _tMax * _nFreqs / 2);
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
  pumpGridFreq.resize(_nZSteps, _nFreqs);
  pumpGridTime.resize(_nZSteps, _nFreqs);
  signalGridFreq.resize(_nZSteps, _nFreqs);
  signalGridTime.resize(_nZSteps, _nFreqs);
}


void _NonlinearMedium::setDispersion(float beta2, float beta2s, float beta1, float beta1s, float beta3, float beta3s) {

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

void _NonlinearMedium::setPump(float pulseType, float chirp) {
  // initial time domain envelopes (pick Gaussian or Soliton Hyperbolic Secant)
  if (pulseType)
    _env = 1 / _tau.cosh() * (-0.5i * chirp * _tau.square()).exp();
  else
    _env = (-0.5 * _tau.square() * (1 + 1i * chirp)).exp();
  // TODO allow custom envelopes
}



std::pair<Array2Dcf, Array2Dcf> _NonlinearMedium::computeGreensFunction(bool inTimeDomain, bool runPump) {
  // Green function matrices
  Array2Dcf greenC;
  Array2Dcf greenS;
  greenC.setZero(_nFreqs, _nFreqs);
  greenS.setZero(_nFreqs, _nFreqs);

  if (runPump) runPumpSimulation();

  auto& grid = inTimeDomain ? signalGridTime : signalGridFreq;

  // Calculate Green's functions with real and imaginary impulse response
  for (int i = 0; i < _nFreqs; i++) {
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


inline Arraycf _NonlinearMedium::fft(const Eigen::VectorXcf& input) {
  Eigen::VectorXcf output(input.cols());
  fftObj.fwd(output, input);
  return output;
}

inline Arraycf _NonlinearMedium::ifft(const Eigen::VectorXcf& input) {
  Eigen::VectorXcf output(input.cols());
  fftObj.inv(output, input);
  return output;
}

/*
inline void _NonlinearMedium::fft(const Eigen::VectorXcf& input, Eigen::VectorXcf& output) {
  fftObj.fwd(output, input);
}

inline void _NonlinearMedium::ifft(const Eigen::VectorXcf& input, Eigen::VectorXcf& output) {
  fftObj.inv(output, input);
}
*/

inline Arrayf _NonlinearMedium::fftshift(const Arrayf& input) {
  Arrayf out(input.rows(), input.cols());
  auto half = input.cols() / 2;
  out.head(half) = input.tail(half);
  out.tail(half) = input.head(half);
  return out;
}


inline Array2Dcf _NonlinearMedium::fftshift(const Array2Dcf& input) {
  Array2Dcf out(input.rows(), input.cols());

  auto halfCols = input.cols() / 2;
  auto halfRows = input.rows() / 2;

  out.topLeftCorner(halfRows, halfCols) = input.bottomRightCorner(halfRows, halfCols);
  out.topRightCorner(halfRows, halfCols) = input.bottomLeftCorner(halfRows, halfCols);
  out.bottomLeftCorner(halfRows, halfCols) = input.topRightCorner(halfRows, halfCols);
  out.bottomRightCorner(halfRows, halfCols) = input.topLeftCorner(halfRows, halfCols);
  return out;
}


void Chi3::runPumpSimulation() {

  pumpGridFreq.row(0) = fft(_env) * (0.5i * _dispersionPump * _dz).exp();
  pumpGridTime.row(0) = ifft(pumpGridFreq.row(0));

  Arraycf temp;
  for (int i = 1; i < _nZSteps; i++) {
    temp = pumpGridTime.row(i-1) * (_nlStep * pumpGridTime.row(i-1).abs2()).exp();
    pumpGridFreq.row(i) = fft(temp) * _dispStepPump;
    pumpGridTime.row(i) = ifft(pumpGridFreq.row(i));
  }

  pumpGridFreq.row(_nZSteps-1) *= (-0.5i * _dispersionPump * _dz).exp();
  pumpGridTime.row(_nZSteps-1)  = ifft(pumpGridFreq.row(_nZSteps-1));
}


void Chi3::runSignalSimulation(const Arraycf& inputProf, bool timeSignal) {
  if (timeSignal)
    signalGridFreq.row(0) = fft(inputProf) * (0.5i * _dispersionSign * _dz).exp();
  else
    signalGridFreq.row(0) = inputProf * (0.5i * _dispersionSign * _dz).exp();
  signalGridTime.row(0) = ifft(signalGridFreq.row(0));

  Arraycf interpP, k1, k2, k3, k4, temp;
  for (int i = 1; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the non-linear propagation
    const auto& prev  = signalGridTime.row(i-1);
    const auto& prevP = pumpGridTime.row(i-1);
    const auto& currP = pumpGridTime.row(i);

    interpP = 0.5 * (prevP + currP);

    k1 = _nlStep * (2 * prevP.abs2()   *  prev             + prevP.square()   *  prev.conjugate());
    k2 = _nlStep * (2 * interpP.abs2() * (prev + 0.5 * k1) + interpP.square() * (prev + 0.5 * k1).conjugate());
    k3 = _nlStep * (2 * interpP.abs2() * (prev + 0.5 * k2) + interpP.square() * (prev + 0.5 * k2).conjugate());
    k4 = _nlStep * (2 * currP.abs2()   * (prev + k3)       + currP.square()   * (prev + k3).conjugate());

    temp = signalGridTime.row(i-1) + (k1 + 2 * k2 + 2 * k3 + k4) / 6;

    // Dispersion step
    signalGridFreq.row(i) = fft(temp) * _dispStepSign;
    signalGridTime.row(i) = ifft(signalGridFreq.row(i));
  }

  signalGridFreq.row(_nZSteps-1) *= (-0.5i * _dispersionSign * _dz).exp();
  signalGridTime.row(_nZSteps-1)  = ifft(signalGridFreq.row(_nZSteps-1));
}


void Chi2::runPumpSimulation() {

  pumpGridFreq.row(0) = fft(_env);
  pumpGridTime.row(0) = _env;

  for (int i = 1; i < _nZSteps; i++) {
    pumpGridFreq.row(i) = pumpGridFreq.row(0) * (1i * i * _dispersionPump * _dz).exp();
    pumpGridTime.row(i) = ifft(pumpGridFreq.row(i));
  }
}


void Chi2::runSignalSimulation(const Arraycf& inputProf, bool timeSignal) {
  if (timeSignal)
    signalGridFreq.row(0) = fft(inputProf) * (0.5i * _dispersionSign * _dz).exp();
  else
    signalGridFreq.row(0) = inputProf * (0.5i * _dispersionSign * _dz).exp();
  signalGridTime.row(0) = ifft(signalGridFreq.row(0));

  Arraycf interpP, k1, k2, k3, k4, temp;
  for (int i = 1; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the non-linear propagation
    const auto& prev =  signalGridTime.row(i-1);
    const auto& prevP = pumpGridTime.row(i-1);
    const auto& currP = pumpGridTime.row(i);

    interpP = 0.5 * (prevP + currP);

    k1 = _nlStep * prevP   *  prev.conjugate();
    k2 = _nlStep * interpP * (prev + 0.5 * k1).conjugate();
    k3 = _nlStep * interpP * (prev + 0.5 * k2).conjugate();
    k4 = _nlStep * currP   * (prev + k3).conjugate();

    temp = signalGridTime.row(i-1) + (k1 + 2 * k2 + 2 * k3 + k4) / 6;

    // Dispersion step
    signalGridFreq.row(i) = fft(temp) * _dispStepSign;
    signalGridTime.row(i) = ifft(signalGridFreq.row(i));
  }

  signalGridFreq.row(_nZSteps-1) *= (-0.5i * _dispersionSign * _dz).exp();
  signalGridTime.row(_nZSteps-1)  = ifft(signalGridFreq.row(_nZSteps-1));
}

