#ifndef CHI2SFGOPA
#define CHI2SFGOPA

#include "_NonlinearMedium.hpp"

class Chi2SFGOPA : public _NonlinearMedium {
  NLM(Chi2SFGOPA, 2)
public:
  Chi2SFGOPA(double relativeLength, double nlLengthFh, double nlLengthHh, double nlLengthFf, double nlLengthHf,
             double beta2F, double beta2H, double beta2h, double beta2f,
             const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
             double beta1F=0, double beta1H=0, double beta1h=0, double beta1f=0, double beta3F=0, double beta3H=0,
             double beta3h=0, double beta3f=0, double diffBeta0SFG=0, double diffBeta0OPA=0, double diffBeta0DOPA=0,
             double rayleighLength=std::numeric_limits<double>::infinity(), double tMax=10, uint tPrecision=512,
             uint zPrecision=100, double chirp=0, double delay=0, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


Chi2SFGOPA::Chi2SFGOPA(double relativeLength, double nlLengthFh, double nlLengthHh, double nlLengthFf, double nlLengthHf,
                       double beta2F, double beta2H, double beta2h, double beta2f, const Eigen::Ref<const Arraycd>& customPump,
                       int pulseType, double beta1F, double beta1H, double beta1h, double beta1f, double beta3F, double beta3H,
                       double beta3h, double beta3f, double diffBeta0SFG, double diffBeta0OPA, double diffBeta0DOPA,
                       double rayleighLength, double tMax, uint tPrecision, uint zPrecision, double chirp, double delay,
                       const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(_nSignalModes, 2, true, relativeLength, {nlLengthFh, nlLengthHh, nlLengthFf, nlLengthHf}, {beta2F, beta2H},
                   {beta2h, beta2f}, customPump, pulseType, {beta1F, beta1H}, {beta1h, beta1f}, {beta3F, beta3H}, {beta3h, beta3f},
                   {diffBeta0SFG, diffBeta0OPA, diffBeta0DOPA}, rayleighLength, tMax, tPrecision, zPrecision, chirp, delay, poling) {}


void Chi2SFGOPA::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                        std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prevH = signal[0].row(i-1);
  const auto& prevF = signal[1].row(i-1);

  const auto& prevP0 = pumpTime[0].row(2*i-2);
  const auto& intrP0 = pumpTime[0].row(2*i-1);
  const auto& currP0 = pumpTime[0].row(2*i);

  const auto& prevP1 = pumpTime[1].row(2*i-2);
  const auto& intrP1 = pumpTime[1].row(2*i-1);
  const auto& currP1 = pumpTime[1].row(2*i);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const std::complex<double> prevMismatchSFG = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchSFG = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatchSFG = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevInvMsmchSFG = 1. / prevMismatchSFG;
  const std::complex<double> intmInvMsmchSFG = 1. / intmMismatchSFG;
  const std::complex<double> currInvMsmchSFG = 1. / currMismatchSFG;
  const std::complex<double> prevMismatchOPA = std::exp(1._I * _diffBeta0[1] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchOPA = std::exp(1._I * _diffBeta0[1] * ((i-.5) * _dz));
  const std::complex<double> currMismatchOPA = std::exp(1._I * _diffBeta0[1] * ( i     * _dz));
  const std::complex<double> prevMismatchDOPA = std::exp(1._I * _diffBeta0[2] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchDOPA = std::exp(1._I * _diffBeta0[2] * ((i-.5) * _dz));
  const std::complex<double> currMismatchDOPA = std::exp(1._I * _diffBeta0[2] * ( i     * _dz));

  k1[0] = prevPolDir * (_nlStep[1] * prevMismatchOPA * prevP1 * prevF.conjugate() + _nlStep[0] *  prevMismatchSFG * prevP0 * prevF);
  k1[1] = prevPolDir * (_nlStep[3] * prevMismatchOPA * prevP1 * prevH.conjugate() + _nlStep[2] * (prevInvMsmchSFG * prevP0.conjugate() * prevH + prevMismatchDOPA * prevP0 * prevF.conjugate()));

  k2[0] = intmPolDir * (_nlStep[1] * intmMismatchOPA * intrP1 * (prevF + 0.5 * k1[1]).conjugate() + _nlStep[0] *  intmMismatchSFG * intrP0 * (prevF + 0.5 * k1[1]));
  k2[1] = intmPolDir * (_nlStep[3] * intmMismatchOPA * intrP1 * (prevH + 0.5 * k1[0]).conjugate() + _nlStep[2] * (intmInvMsmchSFG * intrP0.conjugate() * (prevH + 0.5 * k1[0]) + intmMismatchDOPA * intrP0 * (prevF + 0.5 * k1[1]).conjugate()));

  k3[0] = intmPolDir * (_nlStep[1] * intmMismatchOPA * intrP1 * (prevF + 0.5 * k2[1]).conjugate() + _nlStep[0] *  intmMismatchSFG * intrP0 * (prevF + 0.5 * k2[1]));
  k3[1] = intmPolDir * (_nlStep[3] * intmMismatchOPA * intrP1 * (prevH + 0.5 * k2[0]).conjugate() + _nlStep[2] * (intmInvMsmchSFG * intrP0.conjugate() * (prevH + 0.5 * k2[0]) + intmMismatchDOPA * intrP0 * (prevF + 0.5 * k2[1]).conjugate()));

  k4[0] = currPolDir * (_nlStep[1] * currMismatchOPA * currP1 * (prevF + k3[1]).conjugate() + _nlStep[0] *  currMismatchSFG * currP0 * (prevF + k3[1]));
  k4[1] = currPolDir * (_nlStep[3] * currMismatchOPA * currP1 * (prevH + k3[0]).conjugate() + _nlStep[2] * (currInvMsmchSFG * currP0.conjugate() * (prevH + k3[0]) + currMismatchDOPA * currP0 * (prevF + k3[1]).conjugate()));
}

#endif //CHI2SFGOPA