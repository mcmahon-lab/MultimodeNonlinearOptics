#ifndef CHI2AFC
#define CHI2AFC

#include "_NonlinearMedium.hpp"

class Chi2AFC : public _NonlinearMedium {
  NLM(Chi2AFC, 2)
public:
  Chi2AFC(double relativeLength, double nlLength, double nlLengthOrig, double beta2, double beta2s, double beta2o,
          const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), PulseType pulseType=PulseType{},
          double beta1=0, double beta1s=0, double beta1o=0, double beta3=0, double beta3s=0, double beta3o=0,
          double diffBeta0Start=0, double diffBeta0End=0, double rayleighLength=std::numeric_limits<double>::infinity(),
          double tMax=10, uint tPrecision=512, uint zPrecision=100, IntensityProfile intensityProfile=IntensityProfile{}, double chirp=0, double delay=0);
};


Chi2AFC::Chi2AFC(double relativeLength, double nlLength, double nlLengthOrig, double beta2, double beta2s, double beta2o,
                 const Eigen::Ref<const Arraycd>& customPump, PulseType pulseType, double beta1, double beta1s, double beta1o,
                 double beta3, double beta3s, double beta3o, double diffBeta0Start, double diffBeta0End, double rayleighLength,
                 double tMax, uint tPrecision, uint zPrecision, IntensityProfile intensityProfile, double chirp, double delay) :
  _NonlinearMedium(_nSignalModes, 1, false, relativeLength, {0.5 * M_PI * nlLength, 0.5 * M_PI * nlLengthOrig}, {beta2}, {beta2s, beta2o},
                   customPump, pulseType, {beta1}, {beta1s, beta1o}, {beta3}, {beta3s, beta3o}, {diffBeta0Start, diffBeta0End},
                   rayleighLength, tMax, tPrecision, zPrecision, intensityProfile, chirp, delay) {}


void Chi2AFC::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                     std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prevS = signal[0].row(iPrevSig);
  const auto& prevO = signal[1].row(iPrevSig);

  const auto& prevP = pumpTime[0].row(2*i-2);
  const auto& intrP = pumpTime[0].row(2*i-1);
  const auto& currP = pumpTime[0].row(2*i);

  const double arg = 0.5 * (_diffBeta0[1] - _diffBeta0[0]) / _z;
  const std::complex<double> prevMismatch = std::exp(0.5_I * ((_diffBeta0[0] + (i- 1) * _dz * arg) * (i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(0.5_I * ((_diffBeta0[0] + (i-.5) * _dz * arg) * (i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(0.5_I * ((_diffBeta0[0] +  i     * _dz * arg) *  i     * _dz));
  const std::complex<double> prevInvMsmch = 1. / prevMismatch;
  const std::complex<double> intmInvMsmch = 1. / intmMismatch;
  const std::complex<double> currInvMsmch = 1. / currMismatch;

  k1[0] = (_nlStep[0] * prevMismatch) * prevP             *  prevO;
  k1[1] = (_nlStep[1] * prevInvMsmch) * prevP.conjugate() *  prevS;
  k2[0] = (_nlStep[0] * intmMismatch) * intrP             * (prevO + 0.5 * k1[1]);
  k2[1] = (_nlStep[1] * intmInvMsmch) * intrP.conjugate() * (prevS + 0.5 * k1[0]);
  k3[0] = (_nlStep[0] * intmMismatch) * intrP             * (prevO + 0.5 * k2[1]);
  k3[1] = (_nlStep[1] * intmInvMsmch) * intrP.conjugate() * (prevS + 0.5 * k2[0]);
  k4[0] = (_nlStep[0] * currMismatch) * currP             * (prevO + k3[1]);
  k4[1] = (_nlStep[1] * currInvMsmch) * currP.conjugate() * (prevS + k3[0]);
}

#endif //CHI2AFC

#ifdef NLMMODULE
py::class_<Chi2AFC, _NonlinearMedium> Chi2AFC(m, "Chi2AFC", "Adiabatic sum (or difference) frequency generation, in a rotating frame with linearly varying poling frequency built into the solver");
Chi2AFC.def(
    py::init<double, double, double, double, double, double, Eigen::Ref<const Arraycd>&, _NonlinearMedium::PulseType,
             double, double, double, double, double, double, double, double, double, double, uint, uint,
             _NonlinearMedium::IntensityProfile, double, double>(),
    "relativeLength"_a, "nlLength"_a, "nlLengthOrig"_a, "beta2"_a, "beta2s"_a, "beta2o"_a,
    "customPump"_a = defArraycd, "pulseType"_a = _NonlinearMedium::PulseType{}, "beta1"_a = 0, "beta1s"_a = 0, "beta1o"_a = 0,
    "beta3"_a = 0, "beta3s"_a = 0, "beta3o"_a = 0, "diffBeta0Start"_a = 0, "diffBeta0End"_a = 0, "rayleighLength"_a = infinity,
    "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "intensityProfile"_a = _NonlinearMedium::IntensityProfile{},
    "chirp"_a = 0, "delay"_a = 0);
#endif