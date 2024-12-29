#ifndef CHI3THG
#define CHI3THG

#include "_FullyNonlinearMedium.hpp"

class Chi3THG : public _FullyNonlinearMedium {
  NLM(Chi3THG, 2)
public:
  Chi3THG(double relativeLength, double nlLengthH, double nlLengthF, double beta2h, double beta2f,
          double beta1h=0, double beta1f=0, double beta3h=0, double beta3f=0, double diffBeta0=0,
          double rayleighLength=std::numeric_limits<double>::infinity(), double tMax=10,
          uint tPrecision=512, uint zPrecision=100, IntensityProfile intensityProfile=IntensityProfile{});
};


Chi3THG::Chi3THG(double relativeLength, double nlLengthH, double nlLengthF, double beta2h, double beta2f,
                 double beta1h, double beta1f, double beta3h, double beta3f, double diffBeta0,
                 double rayleighLength, double tMax, uint tPrecision, uint zPrecision, IntensityProfile intensityProfile) :
  _FullyNonlinearMedium(_nSignalModes, false, 0, relativeLength, {nlLengthF, nlLengthH}, {beta2f,  beta2h}, {beta1f, beta1h},
                        {beta3f, beta3h}, {diffBeta0}, rayleighLength, tMax, tPrecision, zPrecision, intensityProfile)
{
  _beta2 = {beta2f, beta2h};
  for (uint m = 0; m < _nSignalModes; m++) {
    if (_beta2[m] != 0) {
      _dispersionSign[m] = 1 / (_beta2[m] * _beta2[m]) - _omega.square();
      _dispersionSign[m] = (_dispersionSign[m] > 0).select(_dispersionSign[m], Arrayd::Zero(_nFreqs));
      _dispersionSign[m] = _dispersionSign[m].sqrt();
      _dispStepSign[m] = (_dispersionSign[m] > 0).select(((1._I * _dz) * (_dispersionSign[m] - _dispersionSign[m](0))).exp(), Arraycd::Zero(_nFreqs));
    }
  }
}


void Chi3THG::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                     std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {

  // Need a temporary, thread safe storage array. Hack (Caution) to avoid allocating each time:
  if (i == 1) {k1.emplace_back(_nFreqs); k1.emplace_back(_nFreqs);}
  Arraycd& tempF = k1[2];
  Arraycd& tempT = k1[3];

  const auto& prvPp = signal[0].row(iPrevSig);
  const auto& prvTH = signal[1].row(iPrevSig);

  const double relIntPrv = relativeIntensity(i- 1);
  const double relIntInt = relativeIntensity(i-.5);
  const double relIntCur = relativeIntensity(i);
  constexpr double third = 1. / 3;

  const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevInvMsmch = 1. / prevMismatch;
  const std::complex<double> intmInvMsmch = 1. / intmMismatch;
  const std::complex<double> currInvMsmch = 1. / currMismatch;

  k1[0] = (relIntPrv * _nlStep[0]) * (prevInvMsmch * prvPp.conjugate().square() * prvTH
                                    + (prvPp.abs2() + 2 * prvTH.abs2()) * prvPp);
  k1[1] = (relIntPrv * _nlStep[1]) * (prevMismatch * third * prvPp.cube()
                                    + (2 * prvPp.abs2() + prvTH.abs2()) * prvTH);
  tempF = prvPp + .5 * k1[0];
  tempT = prvTH + .5 * k1[1];
  k2[0] = (relIntInt * _nlStep[0]) * (intmInvMsmch * tempF.conjugate().square() * tempT
                                    + (tempF.abs2() + 2 * tempT.abs2()) * tempF);
  k2[1] = (relIntInt * _nlStep[1]) * (intmMismatch * third * tempF.cube()
                                    + (2 * tempF.abs2() + tempT.abs2()) * tempT);
  tempF = prvPp + .5 * k2[0];
  tempT = prvTH + .5 * k2[1];
  k3[0] = (relIntInt * _nlStep[0]) * (intmInvMsmch * tempF.conjugate().square() * tempT
                                    + (tempF.abs2() + 2 * tempT.abs2()) * tempF);
  k3[1] = (relIntInt * _nlStep[1]) * (intmMismatch * third * tempF.cube()
                                    + (2 * tempF.abs2() + tempT.abs2()) * tempT);

  tempF = prvPp + k3[0];
  tempT = prvTH + k3[1];
  k4[0] = (relIntCur * _nlStep[0]) * (currInvMsmch * tempF.conjugate().square() * tempT
                                    + (tempF.abs2() + 2 * tempT.abs2()) * tempF);
  k4[1] = (relIntCur * _nlStep[1]) * (currMismatch * third * tempF.cube()
                                    + (2 * tempF.abs2() + tempT.abs2()) * tempT);
}

#endif //CHI3THG

#ifdef NLMMODULE
py::class_<Chi3THG, _FullyNonlinearMedium> Chi3THG(m, "Chi3THG", "Fully nonlinear third harmonic generation.");
Chi3THG.def(
    py::init<double, double, double, double, double, double, double, double, double, double, double, double, uint, uint,
             _NonlinearMedium::IntensityProfile>(),
    "relativeLength"_a, "nlLengthH"_a, "nlLengthF"_a,  "beta2h"_a, "beta2f"_a, "beta1h"_a = 0, "beta1f"_a = 0,
    "beta3h"_a = 0, "beta3f"_a = 0, "diffBeta0"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10,
    "tPrecision"_a = 512, "zPrecision"_a = 100, "intensityProfile"_a = _NonlinearMedium::IntensityProfile{});
#endif