#ifndef CHI2ASHG
#define CHI2ASHG

#include "_NonlinearMedium.hpp"

class Chi2ASHG : public _FullyNonlinearMedium {
  NLM(Chi2ASHG, 2)
public:
  Chi2ASHG(double relativeLength, double nlLengthH, double nlLengthP, double beta2h, double beta2p,
           double beta1h=0, double beta1p=0, double beta3h=0, double beta3p=0, double diffBeta0Start=0,
           double diffBeta0End=0, double rayleighLength=std::numeric_limits<double>::infinity(),
           double tMax=10, uint tPrecision=512, uint zPrecision=100, IntensityProfile intensityProfile=IntensityProfile{});
};


Chi2ASHG::Chi2ASHG(double relativeLength, double nlLengthH, double nlLengthP, double beta2h, double beta2p,
                   double beta1h, double beta1p, double beta3h, double beta3p, double diffBeta0Start, double diffBeta0End,
                   double rayleighLength, double tMax, uint tPrecision, uint zPrecision, IntensityProfile intensityProfile) :
    _FullyNonlinearMedium(_nSignalModes, false, relativeLength, {0.5 * M_PI * nlLengthP, 0.5 * M_PI * nlLengthH}, {beta2p,  beta2h}, {beta1p, beta1h},
                          {beta3p, beta3h}, {diffBeta0Start, diffBeta0End}, rayleighLength, tMax, tPrecision, zPrecision, intensityProfile)
{}


void Chi2ASHG::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                      std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prvPp = signal[0].row(iPrevSig);
  const auto& prvSH = signal[1].row(iPrevSig);

  const double relIntPrv = sqrt(1 / (1 + std::pow(((i- 1) * _dz - 0.5 * _z), 2) / _rayleighLength));
  const double relIntInt = sqrt(1 / (1 + std::pow(((i-.5) * _dz - 0.5 * _z), 2) / _rayleighLength));
  const double relIntCur = sqrt(1 / (1 + std::pow(( i     * _dz - 0.5 * _z), 2) / _rayleighLength));

  const double arg = 0.5 * (_diffBeta0[1] - _diffBeta0[0]) / _z;
  const std::complex<double> prevMismatch = std::exp(0.5_I * ((_diffBeta0[0] + (i- 1) * _dz * arg) * (i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(0.5_I * ((_diffBeta0[0] + (i-.5) * _dz * arg) * (i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(0.5_I * ((_diffBeta0[0] +  i     * _dz * arg) *  i     * _dz));
  const std::complex<double> prevInvMsmch = 1. / prevMismatch;
  const std::complex<double> intmInvMsmch = 1. / intmMismatch;
  const std::complex<double> currInvMsmch = 1. / currMismatch;

  const double prevRelNL = relIntPrv;
  const double intmRelNL = relIntInt;
  const double currRelNL = relIntCur;

  k1[0] = (prevRelNL * _nlStep[0] * prevMismatch) * prvPp.conjugate() * prvSH;
  k1[1] = (prevRelNL * _nlStep[1] * prevInvMsmch  * .5) * prvPp.square();

  k2[0] = (intmRelNL * _nlStep[0] * intmMismatch) * (prvPp + .5 * k1[0]).conjugate() * (prvSH + .5 * k1[1]);
  k2[1] = (intmRelNL * _nlStep[1] * intmInvMsmch  * .5) * (prvPp + .5 * k1[0]).square();

  k3[0] = (intmRelNL * _nlStep[0] * intmMismatch) * (prvPp + .5 * k2[0]).conjugate() * (prvSH + .5 * k2[1]);
  k3[1] = (intmRelNL * _nlStep[1] * intmInvMsmch  * .5) * (prvPp + .5 * k2[0]).square();

  k4[0] = (currRelNL * _nlStep[0] * currMismatch) * (prvPp + k3[0]).conjugate() * (prvSH + k3[1]);
  k4[1] = (currRelNL * _nlStep[1] * currInvMsmch  * .5) * (prvPp + k3[0]).square();
}

#endif //CHI2ASHG