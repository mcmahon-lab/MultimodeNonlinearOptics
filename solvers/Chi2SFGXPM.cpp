#ifndef CHI2SFGXPM
#define CHI2SFGXPM

#include "_NonlinearMedium.hpp"

class Chi2SFGXPM : public _NonlinearMedium {
  NLM(Chi2SFGXPM, 2)
public:
  Chi2SFGXPM(double relativeLength, double nlLength, double nlLengthOrig, double nlLengthChi3, double nlLengthChi3Orig,
             double beta2, double beta2s, double beta2o,
             const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), PulseType pulseType=PulseType{},
             double beta1=0, double beta1s=0, double beta1o=0, double beta3=0, double beta3s=0, double beta3o=0,
             double diffBeta0=0, double rayleighLength=std::numeric_limits<double>::infinity(),
             double tMax=10, uint tPrecision=512, uint zPrecision=100, IntensityProfile intensityProfile=IntensityProfile{},
             double chirp=0, double delay=0, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


Chi2SFGXPM::Chi2SFGXPM(double relativeLength, double nlLength, double nlLengthOrig, double nlLengthChi3, double nlLengthChi3Orig,
                       double beta2, double beta2s, double beta2o, const Eigen::Ref<const Arraycd>& customPump, PulseType pulseType,
                       double beta1, double beta1s, double beta1o, double beta3, double beta3s, double beta3o, double diffBeta0,
                       double rayleighLength, double tMax, uint tPrecision, uint zPrecision, IntensityProfile intensityProfile, double chirp,
                       double delay, const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(_nSignalModes, 1, true, relativeLength, {nlLength, nlLengthOrig, nlLengthChi3 / 2, nlLengthChi3Orig / 2},
                   {beta2}, {beta2s, beta2o}, customPump, pulseType, {beta1}, {beta1s, beta1o}, {beta3}, {beta3s, beta3o},
                   {diffBeta0}, rayleighLength, tMax, tPrecision, zPrecision, intensityProfile, chirp, delay, poling) {}


void Chi2SFGXPM::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                        std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prevS = signal[0].row(iPrevSig);
  const auto& prevO = signal[1].row(iPrevSig);

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

  // note factor of 2 in XPM accounted for in redefinition of nlLengthChi3 and nlLengthChi3Orig above
  k1[0] = ((prevPolDir * _nlStep[0] * prevMismatch) * prevP             + _nlStep[2] * prevP.abs2()) *  prevO;
  k1[1] = ((prevPolDir * _nlStep[1] * prevInvMsmch) * prevP.conjugate() + _nlStep[3] * prevP.abs2()) *  prevS;
  k2[0] = ((intmPolDir * _nlStep[0] * intmMismatch) * intrP             + _nlStep[2] * intrP.abs2()) * (prevO + 0.5 * k1[1]);
  k2[1] = ((intmPolDir * _nlStep[1] * intmInvMsmch) * intrP.conjugate() + _nlStep[3] * intrP.abs2()) * (prevS + 0.5 * k1[0]);
  k3[0] = ((intmPolDir * _nlStep[0] * intmMismatch) * intrP             + _nlStep[2] * intrP.abs2()) * (prevO + 0.5 * k2[1]);
  k3[1] = ((intmPolDir * _nlStep[1] * intmInvMsmch) * intrP.conjugate() + _nlStep[3] * intrP.abs2()) * (prevS + 0.5 * k2[0]);
  k4[0] = ((currPolDir * _nlStep[0] * currMismatch) * currP             + _nlStep[2] * currP.abs2()) * (prevO + k3[1]);
  k4[1] = ((currPolDir * _nlStep[1] * currInvMsmch) * currP.conjugate() + _nlStep[3] * currP.abs2()) * (prevS + k3[0]);
}

#endif //CHI2SFGXPM

#ifdef NLMMODULE
py::class_<Chi2SFGXPM, _NonlinearMedium> Chi2SFGXPM(m, "Chi2SFGXPM", "Sum (or difference) frequency generation with cross phase modulation");
Chi2SFGXPM.def(
    py::init<double, double, double, double, double, double, double, double, Eigen::Ref<const Arraycd>&, _NonlinearMedium::PulseType,
             double, double, double, double, double, double, double, double, double, uint, uint, _NonlinearMedium::IntensityProfile,
             double, double, Eigen::Ref<const Arrayd>&>(),
    "relativeLength"_a, "nlLength"_a, "nlLengthOrig"_a, "nlLengthChi3"_a, "nlLengthChi3Orig"_a, "beta2"_a, "beta2s"_a, "beta2o"_a,
    "customPump"_a = defArraycd, "pulseType"_a = _NonlinearMedium::PulseType{}, "beta1"_a = 0, "beta1s"_a = 0, "beta1o"_a = 0,
    "beta3"_a = 0, "beta3s"_a = 0, "beta3o"_a = 0, "diffBeta0"_a = 0, "rayleighLength"_a = infinity,
    "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "intensityProfile"_a = _NonlinearMedium::IntensityProfile{},
    "chirp"_a = 0, "delay"_a = 0, "poling"_a = defArrayf);
#endif