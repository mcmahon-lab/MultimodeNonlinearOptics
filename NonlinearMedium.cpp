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
                                   double beta1, double beta1s, double beta3, double beta3s,
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
    int Nt = _nFreqs;

    // time and frequency axes
    _tau = 2 * tMax / Nt * Arrayf::LinSpaced(Nt, -Nt / 2, Nt / 2 - 1);
    _tau = fftshift(_tau);
    _omega = M_PI / _tMax * Arrayf::LinSpaced(Nt, -Nt / 2, Nt / 2 - 1);
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

  if (std::abs(beta2) != 1 && !_noDispersion)
    throw std::invalid_argument("Non unit beta2");

  // group velocity difference (relative to beta2 and pulse width)
  _beta1  = beta1;
  _beta1s = beta1s;
  _beta3  = beta3;
  _beta3s = beta3s;

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

    greenC.row(i) += grid.bottomRows<1>() * 0.5;
    greenS.row(i) += grid.bottomRows<1>() * 0.5;

    grid.row(0) = 0;
    grid(0, i) = 1i;
    runSignalSimulation(grid.row(0), inTimeDomain);

    greenC.row(i) -= grid.bottomRows<1>() * 0.5i;
    greenS.row(i) += grid.bottomRows<1>() * 0.5i;
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


void Chi3::runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain) {
  if (inTimeDomain)
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


void _Chi2::runPumpSimulation() {

  pumpFreq.row(0) = fft(_env);
  pumpTime.row(0) = _env;

  for (uint i = 1; i < _nZSteps; i++) {
    pumpFreq.row(i) = pumpFreq.row(0) * (1i * i * _dispersionPump * _dz).exp();
    pumpTime.row(i) = ifft(pumpFreq.row(i));
  }
}


void Chi2PDC::runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain) {
  if (inTimeDomain)
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


Chi2SFG::Chi2SFG(double relativeLength, double nlLength, double nlLengthOrig, double dispLength,
                 double beta2, double beta2s, double beta2o, int pulseType,
                 double beta1, double beta1s, double beta1o, double beta3, double beta3s, double beta3o,
                 double chirp, double tMax, uint tPrecision, uint zPrecision) :
  _Chi2(relativeLength, nlLength, dispLength, beta2, beta2s, pulseType,
        beta1, beta1s, beta3, beta3s, chirp, tMax, tPrecision, zPrecision)
{
  setLengths(relativeLength, nlLength, nlLengthOrig, dispLength, zPrecision);
  resetGrids(tPrecision, tMax);
  setDispersion(beta2, beta2s, beta2o, beta1, beta1s, beta1o, beta3, beta3s, beta3o);
  setPump(pulseType, chirp);
}

Chi2SFG::Chi2SFG(double relativeLength, double nlLength, double nlLengthOrig, double dispLength,
                 double beta2, double beta2s, double beta2o, const Eigen::Ref<const Arraycd>& customPump,
                 double beta1, double beta1s, double beta1o, double beta3, double beta3s, double beta3o,
                 double chirp, double tMax, uint tPrecision, uint zPrecision) :
    _Chi2(relativeLength, nlLength, dispLength,beta2, beta2s, customPump,
          beta1, beta1s, beta3, beta3s,chirp, tMax, tPrecision, zPrecision)
{
  setLengths(relativeLength, nlLength, nlLengthOrig, dispLength, zPrecision);
  resetGrids(tPrecision, tMax);
  setDispersion(beta2, beta2s, beta2o, beta1, beta1s, beta1o, beta3, beta3s, beta3o);
  setPump(customPump, chirp);
}


void Chi2SFG::setLengths(double relativeLength, double nlLength, double nlLengthOrig,
                         double dispLength, uint zPrecision) {
  _NonlinearMedium::setLengths(relativeLength, nlLength, dispLength, zPrecision);

  _NLo = nlLengthOrig;

  if (_noDispersion)
    _nlStepO = 1i * _NL / nlLengthOrig * _dz;
  else if (_noNonlinear)
    _nlStepO = 0;
  else
    _nlStepO = 1i * _DS / nlLengthOrig * _dz;
}


void Chi2SFG::resetGrids(uint nFreqs, double tMax) {
  _NonlinearMedium::resetGrids(nFreqs, tMax);
  originalFreq.resize(_nZSteps, _nFreqs);
  originalTime.resize(_nZSteps, _nFreqs);
}


void Chi2SFG::setDispersion(double beta2, double beta2s, double beta2o, double beta1, double beta1s, double beta1o,
                   double beta3, double beta3s, double beta3o) {
  _NonlinearMedium::setDispersion(beta2, beta2s, beta1, beta1s, beta3, beta3s);
  _beta2o = beta2o;
  _beta1o = beta1o;
  _beta3o = beta3o;

  if (_noDispersion)
    _dispersionOrig.setZero(_nFreqs);
  else
    _dispersionOrig = _omega * (beta1o + _omega * (0.5 * beta2o + _omega * beta3o / 6));

  _dispStepOrig = (1i * _dispersionOrig * _dz).exp();
}


void Chi2SFG::runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain) {
  if (inTimeDomain)
    originalFreq.row(0) = fft(inputProf).array() * (0.5i * _dispersionOrig * _dz).exp();
  else
    originalFreq.row(0) = inputProf * (0.5i * _dispersionOrig * _dz).exp();
  originalTime.row(0) = ifft(originalFreq.row(0));

  signalFreq.row(0) = 0;
  signalTime.row(0) = 0;

  Arraycd interpP(_nFreqs), conjInterpP(_nFreqs);
  Arraycd k1(_nFreqs), k2(_nFreqs), k3(_nFreqs), k4(_nFreqs), tempOrig(_nFreqs);
  Arraycd l1(_nFreqs), l2(_nFreqs), l3(_nFreqs), l4(_nFreqs), tempSign(_nFreqs);
  for (uint i = 1; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the non-linear propagation
    const auto& prevS = signalTime.row(i-1);
    const auto& prevO = originalTime.row(i-1);
    const auto& prevP = pumpTime.row(i-1);
    const auto& currP = pumpTime.row(i);

    interpP = 0.5 * (prevP + currP);
    conjInterpP = interpP.conjugate();

    k1 = _nlStepO * prevP.conjugate() *  prevS;
    l1 = _nlStep  * prevP             *  prevO;
    k2 = _nlStepO * conjInterpP       * (prevS + 0.5 * l1);
    l2 = _nlStep  * interpP           * (prevO + 0.5 * k1);
    k3 = _nlStepO * conjInterpP       * (prevS + 0.5 * l2);
    l3 = _nlStep  * interpP           * (prevO + 0.5 * k2);
    k4 = _nlStepO * currP.conjugate() * (prevS + l3);
    l4 = _nlStep  * currP             * (prevO + k3);

    tempOrig = originalTime.row(i-1) + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    tempSign = signalTime.row(i-1)   + (l1 + 2 * l2 + 2 * l3 + l4) / 6;

    // Dispersion step
    signalFreq.row(i) = fft(tempSign).array() * _dispStepSign;
    signalTime.row(i) = ifft(signalFreq.row(i));

    originalFreq.row(i) = fft(tempOrig).array() * _dispStepOrig;
    originalTime.row(i) = ifft(originalFreq.row(i));
  }

  signalFreq.row(_nZSteps-1) *= (-0.5i * _dispersionSign * _dz).exp();
  signalTime.row(_nZSteps-1)  = ifft(signalFreq.row(_nZSteps-1));

  originalFreq.row(_nZSteps-1) *= (-0.5i * _dispersionOrig * _dz).exp();
  originalTime.row(_nZSteps-1)  = ifft(originalFreq.row(_nZSteps-1));
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

  nMedia = inputMedia.size();
  sharedPump = sharePump;
}


void Cascade::addMedium(_NonlinearMedium& medium) {
  if (medium._nFreqs != _nFreqs or medium._tMax != _tMax)
    throw std::invalid_argument("Medium does not have same time and frequency axes as the first");

  nMedia += 1;
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


void Cascade::runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain) {
  media[0].get().runSignalSimulation(inputProf, inTimeDomain);
  for (uint i = 1; i < media.size(); i++) {
    media[i].get().runSignalSimulation(media[i-1].get().signalFreq.bottomRows<1>(), false);
  }
}


std::pair<Array2Dcd, Array2Dcd> Cascade::computeGreensFunction(bool inTimeDomain, bool runPump) {
  // Green function matrices
  Array2Dcd greenC;
  Array2Dcd greenS;
  greenC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(_nFreqs, _nFreqs);
  greenS.setZero(_nFreqs, _nFreqs);

  if (runPump) runPumpSimulation();

  Array2Dcd tempC, tempS;
  for (auto& medium : media) {
    auto CandS = medium.get().computeGreensFunction(inTimeDomain, false);
    tempC = std::get<0>(CandS).matrix() * greenC.matrix() + std::get<1>(CandS).matrix() * greenS.conjugate().matrix();
    tempS = std::get<0>(CandS).matrix() * greenS.matrix() + std::get<1>(CandS).matrix() * greenC.conjugate().matrix();
    greenC.swap(tempC);
    greenS.swap(tempS);
  }

  return std::make_pair(std::move(greenC), std::move(greenS));
};
