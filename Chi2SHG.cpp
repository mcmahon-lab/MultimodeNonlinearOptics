#ifndef CHI2SHG
#define CHI2SHG

#include "_NonlinearMedium.hpp"

class Chi2SHG : public _NonlinearMedium {
public:
  using _NonlinearMedium::runSignalSimulation;
#ifdef DEPLETESHG
  Chi2SHG(double relativeLength, double nlLength, double nlLengthP, double beta2, double beta2s,
#else
  Chi2SHG(double relativeLength, double nlLength, double beta2, double beta2s,
#endif
          const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
          double beta1=0, double beta1s=0, double beta3=0, double beta3s=0, double diffBeta0=0,
          double rayleighLength=std::numeric_limits<double>::infinity(), double tMax=10, uint tPrecision=512, uint zPrecision=100,
          double chirp=0, double delay=0, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));

protected:
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                           std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime) override;
};


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

#endif //CHI2SHG