#ifndef CHI2SFGII
#define CHI2SFGII

#include "_NonlinearMedium.hpp"

class Chi2SFGII : public _NonlinearMedium {
  NLM(Chi2SFGII, 4)
public:
  Chi2SFGII(double relativeLength, double nlLengthZY, double nlLengthZZ,
            double beta2, double beta2sz, double beta2sy, double beta2oz, double beta2oy,
            const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), PulseType pulseType=PulseType{},
            double beta1=0, double beta1sz=0, double beta1sy=0, double beta1oz=0, double beta1oy=0, double beta3=0,
            double beta3sz=0, double beta3sy=0, double beta3oz=0, double beta3oy=0, double diffBeta0z=0,
            double diffBeta0y=0, double diffBeta0=0, double rayleighLength=std::numeric_limits<double>::infinity(),
            double tMax=10, uint tPrecision=512, uint zPrecision=100, IntensityProfile intensityProfile=IntensityProfile{},
            double chirp=0, double delay=0, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


Chi2SFGII::Chi2SFGII(double relativeLength, double nlLengthZY, double nlLengthZZ,
                     double beta2, double beta2sz, double beta2sy, double beta2oz, double beta2oy,
                     const Eigen::Ref<const Arraycd>& customPump, PulseType pulseType,
                     double beta1, double beta1sz, double beta1sy, double beta1oz, double beta1oy,
                     double beta3, double beta3sz, double beta3sy, double beta3oz, double beta3oy,
                     double diffBeta0z, double diffBeta0y, double diffBeta0, double rayleighLength,
                     double tMax, uint tPrecision, uint zPrecision, IntensityProfile intensityProfile,
                     double chirp, double delay, const Eigen::Ref<const Arrayd>& poling) :
    _NonlinearMedium(_nSignalModes, 1, true, 0, relativeLength, {nlLengthZY, nlLengthZZ},
                     {beta2}, {beta2sz, beta2sy, beta2oz, beta2oy}, customPump, pulseType, {beta1}, {beta1sz, beta1sy, beta1oz, beta1oy},
                     {beta3}, {beta3sz, beta3sy, beta3oz, beta3oy}, {diffBeta0z, diffBeta0y, diffBeta0},
                     rayleighLength, tMax, tPrecision, zPrecision, intensityProfile, chirp, delay, poling) {}


void Chi2SFGII::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                       std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {

  const auto& prevSz = signal[0].row(iPrevSig);
  const auto& prevSy = signal[1].row(iPrevSig);
  const auto& prevOz = signal[2].row(iPrevSig);
  const auto& prevOy = signal[3].row(iPrevSig);

  const auto& prevP = pumpTime[0].row(2*i-2);
  const auto& intrP = pumpTime[0].row(2*i-1);
  const auto& currP = pumpTime[0].row(2*i);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const std::complex<double> prevMismatchCz = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchCz = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatchCz = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevInvMsmchCz = 1. / prevMismatchCz;
  const std::complex<double> intmInvMsmchCz = 1. / intmMismatchCz;
  const std::complex<double> currInvMsmchCz = 1. / currMismatchCz;

  const std::complex<double> prevMismatchCy = std::exp(1._I * _diffBeta0[1] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchCy = std::exp(1._I * _diffBeta0[1] * ((i-.5) * _dz));
  const std::complex<double> currMismatchCy = std::exp(1._I * _diffBeta0[1] * ( i     * _dz));
  const std::complex<double> prevInvMsmchCy = 1. / prevMismatchCy;
  const std::complex<double> intmInvMsmchCy = 1. / intmMismatchCy;
  const std::complex<double> currInvMsmchCy = 1. / currMismatchCy;

  const std::complex<double> prevMismatchSz = std::exp(1._I * _diffBeta0[2] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchSz = std::exp(1._I * _diffBeta0[2] * ((i-.5) * _dz));
  const std::complex<double> currMismatchSz = std::exp(1._I * _diffBeta0[2] * ( i     * _dz));
  const std::complex<double> prevInvMsmchSz = 1. / prevMismatchSz;
  const std::complex<double> intmInvMsmchSz = 1. / intmMismatchSz;
  const std::complex<double> currInvMsmchSz = 1. / currMismatchSz;

  k1[0] =  prevPolDir * prevP * (_nlStep[0] * prevMismatchCz * prevOy + _nlStep[1] * prevMismatchSz * prevOz);
  k1[1] = (prevPolDir * _nlStep[0]  * prevMismatchCy) * prevP * prevOz;
  k1[2] =  prevPolDir * prevP.conjugate() * (_nlStep[0] * prevInvMsmchCy * prevSy + _nlStep[1] * prevInvMsmchSz * prevSz);
  k1[3] = (prevPolDir * _nlStep[0]  * prevInvMsmchCz) * prevP.conjugate() * prevSz;

  k2[0] =  intmPolDir * intrP * (_nlStep[0] * intmMismatchCz * (prevOy + 0.5 * k1[3]) + _nlStep[1] * intmMismatchSz * (prevOz + 0.5 * k1[2]));
  k2[1] = (intmPolDir * _nlStep[0]  * intmMismatchCy) * intrP * (prevOz + 0.5 * k1[2]);
  k2[2] =  intmPolDir * intrP.conjugate() * (_nlStep[0] * intmInvMsmchCy * (prevSy + 0.5 * k1[1]) + _nlStep[1] * intmInvMsmchSz * (prevSz + 0.5 * k1[0]));
  k2[3] = (intmPolDir * _nlStep[0]  * intmInvMsmchCz) * intrP.conjugate() * (prevSz + 0.5 * k1[0]);

  k3[0] =  intmPolDir * intrP * (_nlStep[0] * intmMismatchCz * (prevOy + 0.5 * k2[3]) + _nlStep[1] * intmMismatchSz * (prevOz + 0.5 * k2[2]));
  k3[1] = (intmPolDir * _nlStep[0]  * intmMismatchCy) * intrP * (prevOz + 0.5 * k2[2]);
  k3[2] =  intmPolDir * intrP.conjugate() * (_nlStep[0] * intmInvMsmchCy * (prevSy + 0.5 * k2[1]) + _nlStep[1] * intmInvMsmchSz * (prevSz + 0.5 * k2[0]));
  k3[3] = (intmPolDir * _nlStep[0]  * intmInvMsmchCz) * intrP.conjugate() * (prevSz + 0.5 * k2[0]);

  k4[0] =  prevPolDir * currP * (_nlStep[0] * currMismatchCz * (prevOy + k3[3]) + _nlStep[1] * currMismatchSz * (prevOz + k3[2]));
  k4[1] = (prevPolDir * _nlStep[0]  * currMismatchCy) * currP * (prevOz + k3[2]);
  k4[2] =  prevPolDir * currP.conjugate() * (_nlStep[0] * currInvMsmchCy * (prevSy + k3[1]) + _nlStep[1] * currInvMsmchSz * (prevSz + k3[0]));
  k4[3] = (prevPolDir * _nlStep[0]  * currInvMsmchCz) * currP.conjugate() * (prevSz + k3[0]);
}

#endif //CHI2SFGII

#ifdef NLMMODULE
py::class_<Chi2SFGII, _NonlinearMedium> Chi2SFGII(m, "Chi2SFGII", "Type II or simultaneous 2-mode sum frequency generation with parametric amplification");
Chi2SFGII.def(
    py::init<double, double, /*double, double,*/ double, double, double, double, double, double, Eigen::Ref<const Arraycd>&,
             _NonlinearMedium::PulseType, double, double, double, double, double, double, double, double, double, double,
             double, double, double, double, double, uint, uint, _NonlinearMedium::IntensityProfile, double, double,
             Eigen::Ref<const Arrayd>&>(),
    "relativeLength"_a, "nlLengthZY"_a, "nlLengthZZ"_a, //"nlLengthSignZ"_a, "nlLengthSignY"_a, "nlLengthOrigZ"_a, "nlLengthOrigY"_a,
    "beta2"_a, "beta2sz"_a, "beta2sy"_a, "beta2oz"_a, "beta2oy"_a, "customPump"_a = defArraycd, "pulseType"_a = _NonlinearMedium::PulseType{},
    "beta1"_a = 0, "beta1sz"_a = 0, "beta1sy"_a = 0, "beta1oz"_a = 0, "beta1oy"_a = 0, "beta3"_a = 0, "beta3sz"_a = 0,
    "beta3sy"_a = 0, "beta3oz"_a = 0, "beta3oy"_a = 0, "diffBeta0z"_a = 0, "diffBeta0y"_a = 0, "diffBeta0"_a = 0,
    "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "intensityProfile"_a = _NonlinearMedium::IntensityProfile{},
    "zPrecision"_a = 100, "chirp"_a = 0, "delay"_a = 0, "poling"_a = defArrayf);
#endif