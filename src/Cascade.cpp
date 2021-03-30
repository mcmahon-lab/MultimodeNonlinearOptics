#include "Cascade.hpp"

Cascade::Cascade(bool sharePump, const std::vector<std::reference_wrapper<_NonlinearMedium>>& inputMedia,
                 const std::vector<std::map<uint, uint>>& modeConnections) {

  if (inputMedia.empty())
    throw std::invalid_argument("Cascade must contain at least one medium");

  _nFreqs = inputMedia[0].get()._nFreqs;
  _tMax = inputMedia[0].get()._tMax;

  media.reserve(inputMedia.size());
  for (auto& medium : inputMedia) {
    if (medium.get()._nFreqs != _nFreqs or medium.get()._tMax != _tMax)
      throw std::invalid_argument("Medium does not have same time and frequency axes as the first");
    media.emplace_back(medium);
    _nZSteps += medium.get()._nZSteps;
  }
  _nZStepsP = 0;

  if (modeConnections.size() != media.size() - 1)
    throw std::invalid_argument("Must have one connection per pair of adjacent media");
  uint i = 0;
  for (auto connection = modeConnections.begin(); connection != modeConnections.end(); ++connection, ++i) {
    if (connection->empty())
      throw std::invalid_argument("No connection!");
    for (auto& signalMap : *connection) {
        if (signalMap.first >= media[i].get()._nSignalModes || signalMap.second >= media[i+1].get()._nSignalModes)
          throw std::invalid_argument("Invalid connections, out of range");
    }
    connections.emplace_back(*connection);
  }

  sharedPump = sharePump;
}


void Cascade::addMedium(_NonlinearMedium& medium, const std::map<uint, uint>& connection) {
  if (medium._nFreqs != _nFreqs or medium._tMax != _tMax)
    throw std::invalid_argument("Medium does not have same time and frequency axes as the first");

  if (connection.empty())
    throw std::invalid_argument("No connection!");
  for (auto& signalMap : connection) {
    // todo check uniqueness of values? is this function necessary?
    if (signalMap.first >= medium._nSignalModes || signalMap.second >= medium._nSignalModes)
      throw std::invalid_argument("Invalid connections, out of range");
  }

  media.emplace_back(medium);
  connections.emplace_back(connection);
}


void Cascade::setPump(int pulseType, double chirpLength, double delayLength) {
  if (sharedPump)
    media[0].get().setPump(pulseType, chirpLength, delayLength);
  else {
    for (auto& medium : media)
      medium.get().setPump(pulseType, chirpLength, delayLength);
  }
}


void Cascade::setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength, double delayLength) {
  if (sharedPump)
    media[0].get().setPump(customPump, chirpLength, delayLength);
  else {
    for (auto& medium : media)
      medium.get().setPump(customPump, chirpLength, delayLength);
  }
}


void Cascade::runPumpSimulation() {
  if (not sharedPump) {
    for (auto& medium : media) {
      medium.get().runPumpSimulation();
    }
  }
  else {
    media[0].get().runPumpSimulation();
    for (uint i = 1; i < media.size(); i++) {
      media[i].get()._env = media[i-1].get().pumpTime.bottomRows<1>();
      media[i].get().runPumpSimulation();
    }
  }
}


void Cascade::runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain, uint inputMode) {
  if (inputProf.size() != _nFreqs)
    throw std::invalid_argument("inputProf array size does not match number of frequency/time bins");

  media[0].get().runSignalSimulation(inputProf, inTimeDomain, inputMode);
  for (uint i = 1; i < media.size(); i++) {
    // TODO connect signal channels / specify modes
    media[i].get().runSignalSimulation(media[i-1].get().signalFreq[inputMode].bottomRows<1>(), false);
  }
}


std::pair<Array2Dcd, Array2Dcd>
Cascade::computeGreensFunction(bool inTimeDomain, bool runPump, uint nThreads, bool normalize,
                               const std::vector<char>& useInput, const std::vector<char>& useOutput) {

  if (runPump) runPumpSimulation();

  // Green function matrices
  Array2Dcd greenC;
  Array2Dcd greenS;
  greenC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(_nFreqs, _nFreqs);
  greenS.setZero(_nFreqs, _nFreqs);

  Array2Dcd tempC, tempS;
  for (auto& medium : media) {
    // TODO useInput and useOutput need to be defined based on connections
    auto CandS = medium.get().computeGreensFunction(inTimeDomain, false, nThreads, normalize);
    tempC = std::get<0>(CandS).matrix() * greenC.matrix() + std::get<1>(CandS).matrix() * greenS.conjugate().matrix();
    tempS = std::get<0>(CandS).matrix() * greenS.matrix() + std::get<1>(CandS).matrix() * greenC.conjugate().matrix();
    greenC.swap(tempC);
    greenS.swap(tempS);
  }

  return std::make_pair(std::move(greenC), std::move(greenS));
}


Array2Dcd Cascade::batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs,
                                         bool inTimeDomain, bool runPump, uint nThreads,
                                         uint inputMode, const std::vector<char>& useOutput) {
  if (runPump) runPumpSimulation();

  Array2Dcd outSignals = media[0].get().batchSignalSimulation(inputProfs, inTimeDomain, false, nThreads);

  for (uint i = 1; i < media.size(); i++) {
    // TODO connect signal channels / specify modes
    outSignals = media[i].get().batchSignalSimulation(outSignals, inTimeDomain, false, nThreads);
  }

  return outSignals;
}
