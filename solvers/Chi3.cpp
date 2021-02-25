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
  _NonlinearMedium(_nSignalModes, false, relativeLength, {nlLength}, beta2, {beta2}, customPump, pulseType,
                   0, {0}, beta3, {beta3}, {}, rayleighLength, tMax, tPrecision, zPrecision, chirp, 0)
{}


void Chi3::runPumpSimulation() {
  RowVectorcd fftTemp(_nFreqs);

  FFTtimes(pumpFreq.row(0), _env, ((0.5_I * _dzp) * _dispersionPump).exp())
  IFFT(pumpTime.row(0), pumpFreq.row(0))

  Eigen::VectorXcd relativeStrength = (_dzp / _dz * _nlStep[0]) /
      (1 + (Arrayd::LinSpaced(_nZStepsP, -0.5 * _z, 0.5 * _z) / _rayleighLength).square()).sqrt();

  Arraycd temp(_nFreqs);
  for (uint i = 1; i < _nZStepsP; i++) {
    temp = pumpTime.row(i-1) * (relativeStrength(i-1) * pumpTime.row(i-1).abs2()).exp();
    FFTtimes(pumpFreq.row(i), temp, _dispStepPump)
    IFFT(pumpTime.row(i), pumpFreq.row(i))
  }

  pumpFreq.row(_nZStepsP-1) *= ((-0.5_I * _dzp) * _dispersionPump).exp();
  IFFT(pumpTime.row(_nZStepsP-1), pumpFreq.row(_nZStepsP-1))
}


void Chi3::DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                  std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prev = signal[0].row(i-1);

  const auto& prevP = pumpTime.row(2*i-2);
  const auto& intrP = pumpTime.row(2*i-1);
  const auto& currP = pumpTime.row(2*i);

  k1[0] = _nlStep[0] * (2 * prevP.abs2() *  prev                + prevP.square() *  prev.conjugate());
  k2[0] = _nlStep[0] * (2 * intrP.abs2() * (prev + 0.5 * k1[0]) + intrP.square() * (prev + 0.5 * k1[0]).conjugate());
  k3[0] = _nlStep[0] * (2 * intrP.abs2() * (prev + 0.5 * k2[0]) + intrP.square() * (prev + 0.5 * k2[0]).conjugate());
  k4[0] = _nlStep[0] * (2 * currP.abs2() * (prev + k3[0])       + currP.square() * (prev + k3[0]).conjugate());
}

#endif //CHI3