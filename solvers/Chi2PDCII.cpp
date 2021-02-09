#ifndef CHI2PDCII
#define CHI2PDCII

#include "_NonlinearMedium.hpp"

class Chi2PDCII : public _NonlinearMedium {
  NLM(Chi2PDCII, 2)
public:
  Chi2PDCII(double relativeLength, double nlLength, double nlLengthOrig, double nlLengthI,
            double beta2, double beta2s, double beta2o,
            const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
            double beta1=0, double beta1s=0, double beta1o=0, double beta3=0, double beta3s=0, double beta3o=0,
            double diffBeta0=0, double diffBeta0o=0, double rayleighLength=std::numeric_limits<double>::infinity(),
            double tMax=10, uint tPrecision=512, uint zPrecision=100, double chirp=0, double delay=0,
            const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


Chi2PDCII::Chi2PDCII(double relativeLength, double nlLength, double nlLengthOrig, double nlLengthI,
                     double beta2, double beta2s, double beta2o, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                     double beta1, double beta1s, double beta1o, double beta3, double beta3s, double beta3o,
                     double diffBeta0, double diffBeta0o, double rayleighLength, double tMax, uint tPrecision, uint zPrecision,
                     double chirp, double delay, const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(_nSignalModes, true, relativeLength, {nlLength, nlLengthOrig, nlLengthI}, beta2, {beta2s, beta2o},
                   customPump, pulseType, beta1, {beta1s, beta1o}, beta3, {beta3s, beta3o}, {diffBeta0, diffBeta0o},
                   rayleighLength, tMax, tPrecision, zPrecision, chirp, delay, poling) {}


void Chi2PDCII::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                       std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {

  const auto& prevS = signal[0].row(i-1);
  const auto& prevO = signal[1].row(i-1);

  const auto& prevP = pumpTime.row(2*i-2);
  const auto& intrP = pumpTime.row(2*i-1);
  const auto& currP = pumpTime.row(2*i);

  const double prevPolDir = _poling(i-1);
  const double currPolDir = _poling(i);
  const double intmPolDir = 0.5 * (prevPolDir + currPolDir);

  const std::complex<double> prevMismatch = std::exp(1._I * _diffBeta0[0] * ((i- 1) * _dz));
  const std::complex<double> intmMismatch = std::exp(1._I * _diffBeta0[0] * ((i-.5) * _dz));
  const std::complex<double> currMismatch = std::exp(1._I * _diffBeta0[0] * ( i     * _dz));
  const std::complex<double> prevMismatcho = std::exp(1._I * _diffBeta0[1] * ((i- 1) * _dz));
  const std::complex<double> intmMismatcho = std::exp(1._I * _diffBeta0[1] * ((i-.5) * _dz));
  const std::complex<double> currMismatcho = std::exp(1._I * _diffBeta0[1] * ( i     * _dz));

  k1[0] = (prevPolDir *   _nlStep[0] * prevMismatch) * prevP *  prevO.conjugate();
  k1[1] =  prevPolDir * ((_nlStep[1] * prevMismatch) * prevP *  prevS.conjugate()                + (_nlStep[2] * prevMismatcho) * prevP *  prevO.conjugate());
  k2[0] = (intmPolDir *   _nlStep[0] * intmMismatch) * intrP * (prevO + 0.5 * k1[1]).conjugate();
  k2[1] =  intmPolDir * ((_nlStep[1] * intmMismatch) * intrP * (prevS + 0.5 * k1[0]).conjugate() + (_nlStep[2] * intmMismatcho) * intrP * (prevO + 0.5 * k1[1]).conjugate());
  k3[0] = (intmPolDir *   _nlStep[0] * intmMismatch) * intrP * (prevO + 0.5 * k2[1]).conjugate();
  k3[1] =  intmPolDir * ((_nlStep[1] * intmMismatch) * intrP * (prevS + 0.5 * k2[0]).conjugate() + (_nlStep[2] * intmMismatcho) * intrP * (prevO + 0.5 * k2[1]).conjugate());
  k4[0] = (currPolDir *   _nlStep[0] * currMismatch) * currP * (prevO + k3[1]).conjugate();
  k4[1] =  currPolDir * ((_nlStep[1] * currMismatch) * currP * (prevS + k3[0]).conjugate()       + (_nlStep[2] * currMismatcho) * currP * (prevO + k3[1]).conjugate());
}

#endif //CHI2PDCII