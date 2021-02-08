#ifndef CASCADE
#define CASCADE

#include "NonlinearMedium.hpp"

class Cascade : public _NonlinearMedium {
public:
  Cascade(bool sharePump, const std::vector<std::reference_wrapper<_NonlinearMedium>>& inputMedia,
          const std::vector<std::map<uint, uint>>& connections);
  void addMedium(_NonlinearMedium& medium, const std::map<uint, uint>& connection);
  void setPump(int pulseType, double chirpLength=0, double delayLength=0) override;
  void setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength=0, double delayLength=0) override;
  void runPumpSimulation() override;
  void runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain=true, uint inputMode=0) override;
  std::pair<Array2Dcd, Array2Dcd> computeGreensFunction(bool inTimeDomain=false, bool runPump=true, uint nThreads=1, bool normalize=false,
                                                        const std::vector<char>& useInput={}, const std::vector<char>& useOutput={}) override;
  Array2Dcd batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs, bool inTimeDomain=false, bool runPump=true,
                                  uint nThreads=1, uint inputMode=0, const std::vector<char>& useOutput={}) override;

  _NonlinearMedium& getMedium(uint i) {return media.at(i).get();}
  const std::vector<std::reference_wrapper<_NonlinearMedium>>& getMedia() {return media;}
  uint getNMedia() {return media.size();}

  const Arrayd& getTime()      {return media.at(0).get().getTime();};
  const Arrayd& getFrequency() {return media.at(0).get().getFrequency();};

private: // Disable functions (note: still accessible from base class)
  using _NonlinearMedium::setLengths;
  using _NonlinearMedium::resetGrids;
  using _NonlinearMedium::setDispersion;
  using _NonlinearMedium::getPumpFreq;
  using _NonlinearMedium::getPumpTime;
  using _NonlinearMedium::getSignalFreq;
  using _NonlinearMedium::getSignalTime;
  using _NonlinearMedium::getPoling;
  void runSignalSimulation(const Arraycd&, bool, uint, std::vector<Array2Dcd>&, std::vector<Array2Dcd>&) override {};

protected:
  std::vector<std::reference_wrapper<_NonlinearMedium>> media; /// collection of nonlinear media objects
  std::vector<std::map<uint, uint>> connections;
  bool sharedPump; /// is the pump shared across media or are they independently pumped
};


#endif //CASCADE
