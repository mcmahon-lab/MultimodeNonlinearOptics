#ifndef CHI2PDC
#define CHI2PDC

#include "NonlinearMedium.hpp"

class Chi2PDC : public _NonlinearMedium {
  NLM(Chi2PDC, 1)
public:
  Chi2PDC(double relativeLength, double nlLength, double beta2, double beta2s,
          const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
          double beta1=0, double beta1s=0, double beta3=0, double beta3s=0, double diffBeta0=0,
          double rayleighLength=std::numeric_limits<double>::infinity(),
          double tMax=10, uint tPrecision=512, uint zPrecision=100, double chirp=0, double delay=0,
          const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


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

#endif //CHI2PDC