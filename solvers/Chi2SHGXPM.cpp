#ifndef CHI2SHGXPM
#define CHI2SHGXPM

#include "_NonlinearMedium.hpp"

class Chi2SHGXPM : public _FullyNonlinearMedium {
  NLM(Chi2SHGXPM, 2)
public:
  Chi2SHGXPM(double relativeLength, double nlLength, double nlLengthP, double nlLengthChi3, double beta2, double beta2s,
             double beta1=0, double beta1s=0, double beta3=0, double beta3s=0, double diffBeta0=0,
             double rayleighLength=std::numeric_limits<double>::infinity(), double tMax=10, uint tPrecision=512, uint zPrecision=100,
             IntensityProfile intensityProfile=IntensityProfile{}, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


Chi2SHGXPM::Chi2SHGXPM(double relativeLength, double nlLengthH, double nlLengthP, double nlLengthChi3, double beta2h, double beta2p,
               double beta1h, double beta1p, double beta3h, double beta3p, double diffBeta0,
               double rayleighLength, double tMax, uint tPrecision, uint zPrecision, IntensityProfile intensityProfile,
               const Eigen::Ref<const Arrayd>& poling) :
  _FullyNonlinearMedium(_nSignalModes, true, relativeLength, {nlLengthP, nlLengthH, nlLengthChi3}, {beta2p, beta2h}, {beta1p, beta1h},
                        {beta3p, beta3h}, {diffBeta0}, rayleighLength, tMax, tPrecision, zPrecision, intensityProfile, poling)
{}


void Chi2SHGXPM::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                    std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prvPp = signal[0].row(iPrevSig);
  const auto& prvSH = signal[1].row(iPrevSig);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const double relIntPrv = relativeAmplitude(i- 1);
  const double relIntInt = relativeAmplitude(i-.5);
  const double relIntCur = relativeAmplitude(i);

  const double prevRelNL = prevPolDir * relIntPrv;
  const double intmRelNL = intmPolDir * relIntInt;
  const double currRelNL = currPolDir * relIntCur;

  const std::complex<double> prevRelChi3NL = std::pow(relIntPrv, 2) * _nlStep[2];
  const std::complex<double> intmRelChi3NL = std::pow(relIntInt, 2) * _nlStep[2];
  const std::complex<double> currRelChi3NL = std::pow(relIntCur, 2) * _nlStep[2];

  const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevInvMsmch = 1. / prevMismatch;
  const std::complex<double> intmInvMsmch = 1. / intmMismatch;
  const std::complex<double> currInvMsmch = 1. / currMismatch;

  // TODO multiple repetitions of signal values at intm and curr steps can probably be optimized
  k1[0] = (prevRelNL * _nlStep[0] * prevMismatch) * prvPp.conjugate() * prvSH + prevRelChi3NL * (prvPp.abs2() + 2 * prvSH.abs2()) * prvPp;
  k1[1] = (prevRelNL * _nlStep[1] * prevInvMsmch  * .5) * prvPp.square() + 2. * prevRelChi3NL * (prvSH.abs2() + 2 * prvPp.abs2()) * prvSH;

  k2[0] = (intmRelNL * _nlStep[0] * intmMismatch) * (prvPp + .5 * k1[0]).conjugate() * (prvSH + .5 * k1[1]) + intmRelChi3NL * ((prvPp + .5 * k1[0]).abs2() + 2 * (prvSH + .5 * k1[1]).abs2()) * (prvPp + .5 * k1[0]);
  k2[1] = (intmRelNL * _nlStep[1] * intmInvMsmch  * .5) * (prvPp + .5 * k1[0]).square()                + 2. * intmRelChi3NL * ((prvSH + .5 * k1[1]).abs2() + 2 * (prvPp + .5 * k1[0]).abs2()) * (prvSH + .5 * k1[1]);

  k3[0] = (intmRelNL * _nlStep[0] * intmMismatch) * (prvPp + .5 * k2[0]).conjugate() * (prvSH + .5 * k2[1]) + intmRelChi3NL * ((prvPp + .5 * k2[0]).abs2() + 2 * (prvSH + .5 * k2[1]).abs2()) * (prvPp + .5 * k2[0]);
  k3[1] = (intmRelNL * _nlStep[1] * intmInvMsmch  * .5) * (prvPp + .5 * k2[0]).square()                + 2. * intmRelChi3NL * ((prvSH + .5 * k2[1]).abs2() + 2 * (prvPp + .5 * k2[0]).abs2()) * (prvSH + .5 * k2[1]);

  k4[0] = (currRelNL * _nlStep[0] * currMismatch) * (prvPp + k3[0]).conjugate() * (prvSH + k3[1]) + currRelChi3NL * ((prvPp + k3[0]).abs2() + 2 * (prvSH + k3[1]).abs2()) * (prvPp + k3[0]);
  k4[1] = (currRelNL * _nlStep[1] * currInvMsmch  * .5) * (prvPp + k3[0]).square()           + 2. * currRelChi3NL * ((prvSH + k3[1]).abs2() + 2 * (prvPp + k3[0]).abs2()) * (prvSH + k3[1]);
}

#endif //CHI2SHGXPM