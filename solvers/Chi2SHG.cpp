#ifndef CHI2SHG
#define CHI2SHG

#include "_FullyNonlinearMedium.hpp"

class Chi2SHG : public _FullyNonlinearMedium {
  NLM(Chi2SHG, 2)
public:
  Chi2SHG(double relativeLength, double nlLengthH, double nlLengthP, double beta2h, double beta2p,
          double beta1h=0, double beta1p=0, double beta3h=0, double beta3p=0, double diffBeta0=0,
          double rayleighLength=std::numeric_limits<double>::infinity(), double tMax=10, uint tPrecision=512, uint zPrecision=100,
          IntensityProfile intensityProfile=IntensityProfile{}, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


Chi2SHG::Chi2SHG(double relativeLength, double nlLengthH, double nlLengthP, double beta2h, double beta2p,
                 double beta1h, double beta1p, double beta3h, double beta3p, double diffBeta0,
                 double rayleighLength, double tMax, uint tPrecision, uint zPrecision,
                 IntensityProfile intensityProfile, const Eigen::Ref<const Arrayd>& poling) :
  _FullyNonlinearMedium(_nSignalModes, true, 0, relativeLength, {nlLengthP, nlLengthH}, {beta2p,  beta2h}, {beta1p, beta1h},
                        {beta3p, beta3h}, {diffBeta0}, rayleighLength, tMax, tPrecision, zPrecision, intensityProfile, poling)
{}


void Chi2SHG::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
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

  const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevInvMsmch = 1. / prevMismatch;
  const std::complex<double> intmInvMsmch = 1. / intmMismatch;
  const std::complex<double> currInvMsmch = 1. / currMismatch;

  k1[0] = (prevRelNL * _nlStep[0] * prevMismatch) * prvPp.conjugate() * prvSH;
  k1[1] = (prevRelNL * _nlStep[1] * prevInvMsmch  * .5) * prvPp.square();

  k2[0] = (intmRelNL * _nlStep[0] * intmMismatch) * (prvPp + .5 * k1[0]).conjugate() * (prvSH + .5 * k1[1]);
  k2[1] = (intmRelNL * _nlStep[1] * intmInvMsmch  * .5) * (prvPp + .5 * k1[0]).square();

  k3[0] = (intmRelNL * _nlStep[0] * intmMismatch) * (prvPp + .5 * k2[0]).conjugate() * (prvSH + .5 * k2[1]);
  k3[1] = (intmRelNL * _nlStep[1] * intmInvMsmch  * .5) * (prvPp + .5 * k2[0]).square();

  k4[0] = (currRelNL * _nlStep[0] * currMismatch) * (prvPp + k3[0]).conjugate() * (prvSH + k3[1]);
  k4[1] = (currRelNL * _nlStep[1] * currInvMsmch  * .5) * (prvPp + k3[0]).square();
}

#endif //CHI2SHG

#ifdef NLMMODULE
py::class_<Chi2SHG, _FullyNonlinearMedium> Chi2SHG(m, "Chi2SHG", "Fully nonlinear second harmonic generation");
Chi2SHG.def(
    py::init<double, double, double, double, double, double, double, double, double, double,
             double, double, uint, uint, _NonlinearMedium::IntensityProfile, const Eigen::Ref<const Arrayd>&>(),
    "relativeLength"_a, "nlLengthH"_a, "nlLengthP"_a, "beta2h"_a, "beta2p"_a, "beta1h"_a = 0, "beta1p"_a = 0,
    "beta3h"_a = 0, "beta3p"_a = 0, "diffBeta0"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512,
    "zPrecision"_a = 100, "intensityProfile"_a = _NonlinearMedium::IntensityProfile{}, "poling"_a = defArrayf);
#endif