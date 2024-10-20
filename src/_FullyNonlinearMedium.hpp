#ifndef FULLYNONLINEARMEDIUM
#define FULLYNONLINEARMEDIUM

#include "_NonlinearMedium.hpp"

class _FullyNonlinearMedium : public _NonlinearMedium {
public:
  inline Array2Dcd batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs, bool inTimeDomain=false,
                                         uint nThreads=1, uint inputMode=0, const std::vector<char>& useOutput={}) {
    return _NonlinearMedium::batchSignalSimulation(inputProfs, inTimeDomain, false, nThreads, inputMode, useOutput);
  }

  void setPump(PulseType pulseType, double chirpLength, double delayLength, uint pumpIndex) override {
    throw std::runtime_error("Object does not have this method.");
  }
  void setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength, double delayLength, uint pumpIndex) override {
    throw std::runtime_error("Object does not have this method.");
  }
  void setPump(const _NonlinearMedium& other, uint signalIndex, double delayLength, uint pumpIndex) override {
    throw std::runtime_error("Object does not have this method.");
  }
  void runPumpSimulation() override {
    throw std::runtime_error("Object does not have this method.");
  }
  std::pair<Array2Dcd, Array2Dcd>
      computeGreensFunction(bool inTimeDomain, bool runPump, uint nThreads, bool normalize,
                            const std::vector<char>& useInput, const std::vector<char>& useOutput) override {
    throw std::runtime_error("Object does not have this method.");
  }

protected:
  _FullyNonlinearMedium(uint nSignalmodes, bool canBePoled, double relativeLength, std::initializer_list<double> nlLength,
                        std::initializer_list<double> beta2s, std::initializer_list<double> beta1s, std::initializer_list<double> beta3s,
                        std::initializer_list<double> diffBeta0, double rayleighLength, double tMax, uint tPrecision, uint zPrecision,
                        IntensityProfile intensityProfile, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));

  // Same as _NonlinearMedium except that no pump variables are set
  inline void setDispersion(const std::vector<double>& beta2s, const std::vector<double>& beta1s,
                            const std::vector<double>& beta3s, std::initializer_list<double> diffBeta0);

private:
  using _NonlinearMedium::getPumpFreq;
  using _NonlinearMedium::getPumpTime;
  using _NonlinearMedium::batchSignalSimulation;
};


#endif //FULLYNONLINEARMEDIUM