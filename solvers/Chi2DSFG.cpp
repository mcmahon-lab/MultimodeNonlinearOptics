#ifndef CHI2DSFG
#define CHI2DSFG

#include "_FullyNonlinearMedium.hpp"

class Chi2DSFG : public _FullyNonlinearMedium {
  NLM(Chi2DSFG, 3)
public:
  Chi2DSFG(double relativeLength, double nlLengthP, double nlLengthS, double nlLengthD,
           double beta2p, double beta2s, double beta2d, double beta1p=0, double beta1s=0, double beta1d=0,
           double beta3p=0, double beta3s=0, double beta3d=0, double diffBeta0=0,
           double rayleighLength=std::numeric_limits<double>::infinity(), double tMax=10, uint tPrecision=512,
           uint zPrecision=100, IntensityProfile intensityProfile=IntensityProfile{},
           const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


Chi2DSFG::Chi2DSFG(double relativeLength, double nlLengthP, double nlLengthS, double nlLengthD,
                   double beta2p, double beta2s, double beta2d, double beta1p, double beta1s, double beta1d,
                   double beta3p, double beta3s, double beta3d, double diffBeta0, double rayleighLength, double tMax,
                   uint tPrecision, uint zPrecision, IntensityProfile intensityProfile, const Eigen::Ref<const Arrayd>& poling) :
  _FullyNonlinearMedium(_nSignalModes, true, relativeLength, {nlLengthP, nlLengthS, nlLengthD}, {beta2p, beta2s, beta2d},
                        {beta1p, beta1s, beta1d}, {beta3p, beta3s, beta3d}, {diffBeta0}, rayleighLength, tMax,
                        tPrecision, zPrecision, intensityProfile, poling) {}


void Chi2DSFG::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                      std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prvP = signal[0].row(iPrevSig);
  const auto& prvS = signal[1].row(iPrevSig);
  const auto& prvD = signal[2].row(iPrevSig);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const double relIntPrv = sqrt(1 / (1 + std::pow(((i- 1) * _dz - 0.5 * _z), 2) / _rayleighLength));
  const double relIntInt = sqrt(1 / (1 + std::pow(((i-.5) * _dz - 0.5 * _z), 2) / _rayleighLength));
  const double relIntCur = sqrt(1 / (1 + std::pow(( i     * _dz - 0.5 * _z), 2) / _rayleighLength));

  const double prevRelNL = prevPolDir * relIntPrv;
  const double intmRelNL = intmPolDir * relIntInt;
  const double currRelNL = currPolDir * relIntCur;

  const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevInvMsmch = 1. / prevMismatch;
  const std::complex<double> intmInvMsmch = 1. / intmMismatch;
  const std::complex<double> currInvMsmch = 1. / currMismatch;

  k1[0] = (prevRelNL * _nlStep[0] * prevMismatch) * prvS * prvD.conjugate();
  k1[1] = (prevRelNL * _nlStep[1] * prevInvMsmch) * prvP * prvD;
  k1[2] = (prevRelNL * _nlStep[2] * prevMismatch) * prvS * prvP.conjugate();

  k2[0] = (intmRelNL * _nlStep[0] * intmMismatch) * (prvS + .5 * k1[1]) * (prvD + .5 * k1[2]).conjugate();
  k2[1] = (intmRelNL * _nlStep[1] * intmInvMsmch) * (prvP + .5 * k1[0]) * (prvD + .5 * k1[2]);
  k2[2] = (intmRelNL * _nlStep[2] * intmMismatch) * (prvS + .5 * k1[1]) * (prvP + .5 * k1[0]).conjugate();

  k3[0] = (intmRelNL * _nlStep[0] * intmMismatch) * (prvS + .5 * k2[1]) * (prvD + .5 * k2[2]).conjugate();
  k3[1] = (intmRelNL * _nlStep[1] * intmInvMsmch) * (prvP + .5 * k2[0]) * (prvD + .5 * k2[2]);
  k3[2] = (intmRelNL * _nlStep[2] * intmMismatch) * (prvS + .5 * k2[1]) * (prvP + .5 * k2[0]).conjugate();

  k4[0] = (currRelNL * _nlStep[0] * currMismatch) * (prvS + k3[1]) * (prvD + k3[2]).conjugate();
  k4[1] = (currRelNL * _nlStep[1] * currInvMsmch) * (prvP + k3[0]) * (prvD + k3[2]);
  k4[2] = (currRelNL * _nlStep[2] * currMismatch) * (prvS + k3[1]) * (prvP + k3[0]).conjugate();
}

#endif //CHI2DSFG