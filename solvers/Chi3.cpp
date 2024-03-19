#ifndef CHI3
#define CHI3

#include "_NonlinearMedium.hpp"

class Chi3 : public _NonlinearMedium {
  NLM(Chi3, 1)
public:
  Chi3(double relativeLength, double nlLength, double beta2,
       const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
       double beta3=0, double rayleighLength=std::numeric_limits<double>::infinity(),
       double tMax=10, uint tPrecision=512, uint zPrecision=100, double chirp=0);

  void runPumpSimulation() override;
};


Chi3::Chi3(double relativeLength, double nlLength, double beta2, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
           double beta3, double rayleighLength, double tMax, uint tPrecision, uint zPrecision, double chirp) :
  _NonlinearMedium(_nSignalModes, 1, false, relativeLength, {nlLength}, {beta2}, {beta2}, customPump, pulseType,
                   {0}, {0}, {beta3}, {beta3}, {}, rayleighLength, tMax, tPrecision, zPrecision, chirp, 0)
{}


void Chi3::runPumpSimulation() {
  FFTi(pumpFreq[0], _envelope[0], 0, 0)
  pumpFreq[0].row(0) *= ((0.5_I * _dzp) * _dispersionPump[0]).exp();
  IFFTi(pumpTime[0], pumpFreq[0], 0, 0)

  Eigen::VectorXcd relativeIntensity = (_dzp / _dz * _nlStep[0]) /
      (1 + (Arrayd::LinSpaced(_nZStepsP, -0.5 * _z, 0.5 * _z) / _rayleighLength).square());

  for (uint i = 1; i < _nZStepsP; i++) {
    pumpTime[0].row(i) = pumpTime[0].row(i-1) * (relativeIntensity(i-1) * pumpTime[0].row(i-1).abs2()).exp();
    FFTi(pumpFreq[0], pumpTime[0], i, i)
    pumpFreq[0].row(i) *= _dispStepPump[0];
    IFFTi(pumpTime[0], pumpFreq[0], i, i)
  }

  pumpFreq[0].row(_nZStepsP-1) *= ((-0.5_I * _dzp) * _dispersionPump[0]).exp();
  IFFTi(pumpTime[0], pumpFreq[0], _nZStepsP-1, _nZStepsP-1)
}


void Chi3::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                  std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prev = signal[0].row(iPrevSig);

  const auto& prevP = pumpTime[0].row(2*i-2);
  const auto& intrP = pumpTime[0].row(2*i-1);
  const auto& currP = pumpTime[0].row(2*i);

  k1[0] = _nlStep[0] * (2 * prevP.abs2() *  prev                + prevP.square() *  prev.conjugate());
  k2[0] = _nlStep[0] * (2 * intrP.abs2() * (prev + 0.5 * k1[0]) + intrP.square() * (prev + 0.5 * k1[0]).conjugate());
  k3[0] = _nlStep[0] * (2 * intrP.abs2() * (prev + 0.5 * k2[0]) + intrP.square() * (prev + 0.5 * k2[0]).conjugate());
  k4[0] = _nlStep[0] * (2 * currP.abs2() * (prev + k3[0])       + currP.square() * (prev + k3[0]).conjugate());
}

#endif //CHI3