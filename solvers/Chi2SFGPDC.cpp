#ifndef CHI2SFGPDC
#define CHI2SFGPDC

#include "_NonlinearMedium.hpp"

class Chi2SFGPDC : public _NonlinearMedium {
  NLM(Chi2SFGPDC, 2)
public:
  Chi2SFGPDC(double relativeLength, double nlLength, double nlLengthOrig, double beta2, double beta2s, double beta2o,
             const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
             double beta1=0, double beta1s=0, double beta1o=0, double beta3=0, double beta3s=0, double beta3o=0,
             double diffBeta0=0, double diffBeta0o=0, double rayleighLength=std::numeric_limits<double>::infinity(),
             double tMax=10, uint tPrecision=512, uint zPrecision=100, double chirp=0, double delay=0,
             const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


Chi2SFGPDC::Chi2SFGPDC(double relativeLength, double nlLength, double nlLengthOrig, double beta2, double beta2s, double beta2o,
                       const Eigen::Ref<const Arraycd>& customPump, int pulseType, double beta1, double beta1s, double beta1o,
                       double beta3, double beta3s, double beta3o, double diffBeta0, double diffBeta0o, double rayleighLength,
                       double tMax, uint tPrecision, uint zPrecision, double chirp, double delay, const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(_nSignalModes, 1, true, relativeLength, {nlLength, nlLengthOrig}, {beta2}, {beta2s, beta2o},
                   customPump, pulseType, {beta1}, {beta1s, beta1o}, {beta3}, {beta3s, beta3o}, {diffBeta0, diffBeta0o},
                   rayleighLength, tMax, tPrecision, zPrecision, chirp, delay, poling) {}


void Chi2SFGPDC::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                        std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prevS = signal[0].row(i-1);
  const auto& prevO = signal[1].row(i-1);

  const auto& prevP = pumpTime[0].row(2*i-2);
  const auto& intrP = pumpTime[0].row(2*i-1);
  const auto& currP = pumpTime[0].row(2*i);

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

  k1[0] = (prevPolDir * _nlStep[0]  *  prevMismatch) * prevP             *  prevO;
  k1[1] = (prevPolDir * _nlStep[1]) * (prevInvMsmch  * prevP.conjugate() *  prevS                + prevMismatcho * prevP *  prevO.conjugate());
  k2[0] = (intmPolDir * _nlStep[0]  *  intmMismatch) * intrP             * (prevO + 0.5 * k1[1]);
  k2[1] = (intmPolDir * _nlStep[1]) * (intmInvMsmch  * intrP.conjugate() * (prevS + 0.5 * k1[0]) + intmMismatcho * intrP * (prevO + 0.5 * k1[1]).conjugate());
  k3[0] = (intmPolDir * _nlStep[0]  *  intmMismatch) * intrP             * (prevO + 0.5 * k2[1]);
  k3[1] = (intmPolDir * _nlStep[1]) * (intmInvMsmch  * intrP.conjugate() * (prevS + 0.5 * k2[0]) + intmMismatcho * intrP * (prevO + 0.5 * k2[1]).conjugate());
  k4[0] = (currPolDir * _nlStep[0]  *  currMismatch) * currP             * (prevO + k3[1]);
  k4[1] = (currPolDir * _nlStep[1]) * (currInvMsmch  * currP.conjugate() * (prevS + k3[0])       + currMismatcho * currP * (prevO + k3[1]).conjugate());
}

#endif //CHI2SFGPDC