#ifndef CHI2SHGOPA
#define CHI2SHGOPA

#include "_FullyNonlinearMedium.hpp"

class Chi2SHGOPA : public _FullyNonlinearMedium {
  NLM(Chi2SHGOPA, 4)
public:
  Chi2SHGOPA(double relativeLength, double nlLengthP, double nlLengthSH, double nlLengthPA1, double nlLengthPA2,
             double beta2p, double beta2sh, double beta2pa1, double beta2pa2,
             double beta1p=0, double beta1sh=0, double beta1pa1=0, double beta1pa2=0,
             double beta3p=0, double beta3sh=0, double beta3pa1=0, double beta3pa2=0,
             double diffBeta0shg=0, double diffBeta0opa=0, double rayleighLength=std::numeric_limits<double>::infinity(),
             double tMax=10, uint tPrecision=512, uint zPrecision=100,
             const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


Chi2SHGOPA::Chi2SHGOPA(double relativeLength, double nlLengthP, double nlLengthSH, double nlLengthPA1, double nlLengthPA2,
                       double beta2p, double beta2sh, double beta2pa1, double beta2pa2,
                       double beta1p, double beta1sh, double beta1pa1, double beta1pa2,
                       double beta3p, double beta3sh, double beta3pa1, double beta3pa2,
                       double diffBeta0shg, double diffBeta0opa, double rayleighLength,
                       double tMax, uint tPrecision, uint zPrecision, const Eigen::Ref<const Arrayd>& poling) :
  _FullyNonlinearMedium(_nSignalModes, true, relativeLength, {nlLengthP, nlLengthSH, nlLengthPA1, nlLengthPA2},
                        {beta2p, beta2sh, beta2pa1, beta2pa2}, {beta1p, beta1sh, beta1pa1, beta1pa2},
                        {beta3p, beta3sh, beta3pa1, beta3pa2}, {diffBeta0shg, diffBeta0opa},
                        rayleighLength, tMax, tPrecision, zPrecision, poling) {}


void Chi2SHGOPA::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                        std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prvPp = signal[0].row(i-1);
  const auto& prvSH = signal[1].row(i-1);
  const auto& prvA1 = signal[2].row(i-1);
  const auto& prvA2 = signal[3].row(i-1);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const double relIntPrv = sqrt(1 / (1 + std::pow(((i- 1) * _dz - 0.5 * _z), 2) / _rayleighLength));
  const double relIntInt = sqrt(1 / (1 + std::pow(((i-.5) * _dz - 0.5 * _z), 2) / _rayleighLength));
  const double relIntCur = sqrt(1 / (1 + std::pow(( i     * _dz - 0.5 * _z), 2) / _rayleighLength));

  const double prevRelNL = prevPolDir * relIntPrv;
  const double currRelNL = currPolDir * relIntInt;
  const double intmRelNL = intmPolDir * relIntCur;

  const std::complex<double> prevMismatchSHG = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchSHG = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatchSHG = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevInvMsmchSHG = 1. / prevMismatchSHG;
  const std::complex<double> intmInvMsmchSHG = 1. / intmMismatchSHG;
  const std::complex<double> currInvMsmchSHG = 1. / currMismatchSHG;

  const std::complex<double> prevMismatchOPA = std::exp(1._I * _diffBeta0[1] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchOPA = std::exp(1._I * _diffBeta0[1] * ((i-.5) * _dz));
  const std::complex<double> currMismatchOPA = std::exp(1._I * _diffBeta0[1] * ( i     * _dz));
  const std::complex<double> prevInvMsmchOPA = 1. / prevMismatchOPA;
  const std::complex<double> intmInvMsmchOPA = 1. / intmMismatchOPA;
  const std::complex<double> currInvMsmchOPA = 1. / currMismatchOPA;

  k1[0] = (prevRelNL * _nlStep[0]  *  prevMismatchSHG) * prvPp.conjugate() * prvSH;
  k1[1] = (prevRelNL * _nlStep[1]) * (prevInvMsmchSHG  * prvPp.square() + prevMismatchOPA * prvA2 * prvA1);
  k1[2] = (prevRelNL * _nlStep[2]  *  prevInvMsmchOPA) * prvSH * prvA2.conjugate();
  k1[3] = (prevRelNL * _nlStep[3]  *  prevInvMsmchOPA) * prvSH * prvA1.conjugate();

  k2[0] = (intmRelNL * _nlStep[0]  *  intmMismatchSHG) * (prvPp + .5 * k1[0]).conjugate() * (prvSH + .5 * k1[1]);
  k2[1] = (intmRelNL * _nlStep[1]) * (intmInvMsmchSHG  * (prvPp + .5 * k1[0]).square() + intmMismatchOPA * (prvA1 + .5 * k1[2]) * (prvA2 + .5 * k1[3]));
  k2[2] = (intmRelNL * _nlStep[2]  *  intmInvMsmchOPA) * (prvSH + .5 * k1[1]) * (prvA2 + .5 * k1[3]).conjugate();
  k2[3] = (intmRelNL * _nlStep[3]  *  intmInvMsmchOPA) * (prvSH + .5 * k1[1]) * (prvA1 + .5 * k1[2]).conjugate();

  k3[0] = (intmRelNL * _nlStep[0]  *  intmMismatchSHG) * (prvPp + .5 * k2[0]).conjugate() * (prvSH + .5 * k2[1]);
  k3[1] = (intmRelNL * _nlStep[1]) * (intmInvMsmchSHG  * (prvPp + .5 * k2[0]).square() + intmMismatchOPA * (prvA1 + .5 * k2[2]) * (prvA2 + .5 * k2[3]));
  k3[2] = (intmRelNL * _nlStep[2]  *  intmInvMsmchOPA) * (prvSH + .5 * k2[1]) * (prvA2 + .5 * k2[3]).conjugate();
  k3[3] = (intmRelNL * _nlStep[3]  *  intmInvMsmchOPA) * (prvSH + .5 * k2[1]) * (prvA1 + .5 * k2[2]).conjugate();

  k4[0] = (currRelNL * _nlStep[0]  *  currMismatchSHG) * (prvPp + k3[0]).conjugate() * (prvSH + k3[1]);
  k4[1] = (currRelNL * _nlStep[1]) * (currInvMsmchSHG  * (prvPp + k3[0]).square() + currMismatchOPA * (prvA1 + k3[2]) * (prvA2 + k3[3]));
  k4[2] = (currRelNL * _nlStep[2]  *  currInvMsmchOPA) * (prvSH + k3[1]) * (prvA2 + k3[3]).conjugate();
  k4[3] = (currRelNL * _nlStep[3]  *  currInvMsmchOPA) * (prvSH + k3[1]) * (prvA1 + k3[2]).conjugate();
}

#endif //CHI2SHGOPA