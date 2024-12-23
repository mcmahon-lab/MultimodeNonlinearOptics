#ifndef CHI2SFGII
#define CHI2SFGII

#include "_NonlinearMedium.hpp"

class Chi2SFGII : public _NonlinearMedium {
  NLM(Chi2SFGII, 4)
public:
  Chi2SFGII(double relativeLength, double nlLengthSignZ, double nlLengthSignY, double nlLengthOrigZ, double nlLengthOrigY,
            double beta2, double beta2sz, double beta2sy, double beta2oz, double beta2oy,
            const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), PulseType pulseType=PulseType{},
            double beta1=0, double beta1sz=0, double beta1sy=0, double beta1oz=0, double beta1oy=0, double beta3=0,
            double beta3sz=0, double beta3sy=0, double beta3oz=0, double beta3oy=0, double diffBeta0z=0,
            double diffBeta0y=0, double diffBeta0s=0, double rayleighLength=std::numeric_limits<double>::infinity(),
            double tMax=10, uint tPrecision=512, uint zPrecision=100, IntensityProfile intensityProfile=IntensityProfile{},
            double chirp=0, double delay=0, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


Chi2SFGII::Chi2SFGII(double relativeLength, double nlLengthSignZ, double nlLengthSignY, double nlLengthOrigZ, double nlLengthOrigY,
                     double beta2, double beta2sz, double beta2sy, double beta2oz, double beta2oy,
                     const Eigen::Ref<const Arraycd>& customPump, PulseType pulseType,
                     double beta1, double beta1sz, double beta1sy, double beta1oz, double beta1oy,
                     double beta3, double beta3sz, double beta3sy, double beta3oz, double beta3oy,
                     double diffBeta0z, double diffBeta0y, double diffBeta0s, double rayleighLength,
                     double tMax, uint tPrecision, uint zPrecision, IntensityProfile intensityProfile,
                     double chirp, double delay, const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(_nSignalModes, 1, true, relativeLength, {nlLengthSignZ, nlLengthSignY, nlLengthOrigZ, nlLengthOrigY},
                   {beta2}, {beta2sz, beta2sy, beta2oz, beta2oy}, customPump, pulseType, {beta1}, {beta1sz, beta1sy, beta1oz, beta1oy},
                   {beta3}, {beta3sz, beta3sy, beta3oz, beta3oy}, {diffBeta0z, diffBeta0y, diffBeta0s},
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

  const std::complex<double> prevMismatchSq = std::exp(1._I * _diffBeta0[2] * ((i- 1) * _dz));
  const std::complex<double> intmMismatchSq = std::exp(1._I * _diffBeta0[2] * ((i-.5) * _dz));
  const std::complex<double> currMismatchSq = std::exp(1._I * _diffBeta0[2] * ( i     * _dz));

  k1[0] = (prevPolDir * _nlStep[0]  * prevMismatchCz) * prevP * prevOy;
  k1[1] = (prevPolDir * _nlStep[1]  * prevMismatchCy) * prevP * prevOz;
  k1[2] = (prevPolDir * _nlStep[2]) * (prevInvMsmchCy * prevP.conjugate() * prevSy + prevMismatchSq * prevP * prevOy.conjugate());
  k1[3] = (prevPolDir * _nlStep[3]) * (prevInvMsmchCz * prevP.conjugate() * prevSz + prevMismatchSq * prevP * prevOz.conjugate());

  k2[0] = (intmPolDir * _nlStep[0]  * intmMismatchCz) * intrP * (prevOy + 0.5 * k1[3]);
  k2[1] = (intmPolDir * _nlStep[1]  * intmMismatchCy) * intrP * (prevOz + 0.5 * k1[2]);
  k2[2] = (intmPolDir * _nlStep[2]) * (intmInvMsmchCy * intrP.conjugate() * (prevSy + 0.5 * k1[1]) + intmMismatchSq * intrP * (prevOy + 0.5 * k1[3]).conjugate());
  k2[3] = (intmPolDir * _nlStep[3]) * (intmInvMsmchCz * intrP.conjugate() * (prevSz + 0.5 * k1[0]) + intmMismatchSq * intrP * (prevOz + 0.5 * k1[2]).conjugate());

  k3[0] = (intmPolDir * _nlStep[0]  * intmMismatchCz) * intrP * (prevOy + 0.5 * k2[3]);
  k3[1] = (intmPolDir * _nlStep[1]  * intmMismatchCy) * intrP * (prevOz + 0.5 * k2[2]);
  k3[2] = (intmPolDir * _nlStep[2]) * (intmInvMsmchCy * intrP.conjugate() * (prevSy + 0.5 * k2[1]) + intmMismatchSq * intrP * (prevOy + 0.5 * k2[3]).conjugate());
  k3[3] = (intmPolDir * _nlStep[3]) * (intmInvMsmchCz * intrP.conjugate() * (prevSz + 0.5 * k2[0]) + intmMismatchSq * intrP * (prevOz + 0.5 * k2[2]).conjugate());

  k4[0] = (prevPolDir * _nlStep[0]  * currMismatchCz) * currP * (prevOy + k3[3]);
  k4[1] = (prevPolDir * _nlStep[1]  * currMismatchCy) * currP * (prevOz + k3[2]);
  k4[2] = (prevPolDir * _nlStep[2]) * (currInvMsmchCy * currP.conjugate() * (prevSy + k3[1]) + currMismatchSq * currP * (prevOy + k3[3]).conjugate());
  k4[3] = (prevPolDir * _nlStep[3]) * (currInvMsmchCz * currP.conjugate() * (prevSz + k3[0]) + currMismatchSq * currP * (prevOz + k3[2]).conjugate());
}

#endif //CHI2SFGII