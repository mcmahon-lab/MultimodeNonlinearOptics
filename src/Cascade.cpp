#include "Cascade.hpp"

Cascade::Cascade(const std::vector<std::reference_wrapper<_NonlinearMedium>>& inputMedia,
                 const std::vector<std::map<uint, uint>>& modeConnections, bool sharePump) {

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

  connections.reserve(modeConnections.size());
  for (uint i = 0; i < modeConnections.size(); i++) {
    if (modeConnections[i].empty())
      throw std::invalid_argument("Connections missing between media!");
    for (const auto& signalMap : modeConnections[i]) {
        if (signalMap.second >= media[i].get()._nSignalModes || signalMap.first >= media[i+1].get()._nSignalModes)
          throw std::invalid_argument("Invalid connections, out of range");
    }
    connections.emplace_back(modeConnections[i]);
  }

  sharedPump = sharePump;
}


void Cascade::addMedium(_NonlinearMedium& medium, const std::map<uint, uint>& connection) {
  if (medium._nFreqs != _nFreqs or medium._tMax != _tMax)
    throw std::invalid_argument("Medium does not have same time and frequency axes as the first");

  if (connection.empty())
    throw std::invalid_argument("No connection!");
  for (const auto& signalMap : connection) {
    if (signalMap.second >= media[media.size()-1].get()._nSignalModes || signalMap.first >= medium._nSignalModes)
      throw std::invalid_argument("Invalid connections, out of range");
  }

  media.emplace_back(medium);
  connections.emplace_back(connection);
}


void Cascade::setPump(PulseType pulseType, double chirpLength, double delayLength, uint pumpIndex) {
  if (sharedPump)
    media[0].get().setPump(pulseType, chirpLength, delayLength, pumpIndex);
  else {
    for (auto& medium : media)
      medium.get().setPump(pulseType, chirpLength, delayLength, pumpIndex);
  }
}


void Cascade::setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength, double delayLength, uint pumpIndex) {
  if (sharedPump)
    media[0].get().setPump(customPump, chirpLength, delayLength, pumpIndex);
  else {
    for (auto& medium : media)
      medium.get().setPump(customPump, chirpLength, delayLength, pumpIndex);
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
      media[i].get()._envelope[0] = media[i-1].get().pumpTime[0].bottomRows<1>();
      media[i].get().runPumpSimulation();
    }
  }
}


void Cascade::runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain, uint inputMode) {
  media[0].get().runSignalSimulation(inputProf, inTimeDomain, inputMode);
  for (uint i = 1; i < media.size(); i++) {
    // Determine largest input index
    uint maxInputVal = 0;
    for (const auto& connection : connections[i-1]) {
      if (connection.first > maxInputVal)
        maxInputVal = connection.first;
    }
    // Concatenate inputs based on connection map
    Arraycd inputToNext = Arraycd::Zero(_nFreqs * (maxInputVal + 1));
    for (const auto& connection : connections[i-1]) {
      inputToNext.segment(connection.first * _nFreqs, _nFreqs) = media[i-1].get().signalFreq[connection.second].bottomRows<1>();
    }
    media[i].get().runSignalSimulation(inputToNext, false);
  }
}


std::pair<Array2Dcd, Array2Dcd>
Cascade::computeGreensFunction(bool inTimeDomain, bool runPump, uint nThreads, bool normalize,
                               const std::vector<uint8_t>& useInput, const std::vector<uint8_t>& useOutput) {

  if (runPump) runPumpSimulation();

  // Green function matrices
  Array2Dcd greenC, greenS;
  {
    auto CandS = media[0].get().computeGreensFunction(inTimeDomain, false, nThreads, normalize, useInput,
                                                      (media.size() == 1? useOutput : std::vector<uint8_t>{}));
    greenC.swap(std::get<0>(CandS));
    greenS.swap(std::get<1>(CandS));
  }
  for (uint i = 1; i < media.size(); i++) {
    // TODO matrix multiplication needs to be performed based on connections
    auto CandS = media[i].get().computeGreensFunction(inTimeDomain, false, nThreads, normalize, std::vector<uint8_t>{},
                                                      (i == media.size()-1? useOutput : std::vector<uint8_t>{}));
    Array2Dcd tempC = std::get<0>(CandS).matrix() * greenC.matrix() + std::get<1>(CandS).matrix() * greenS.conjugate().matrix();
    Array2Dcd tempS = std::get<0>(CandS).matrix() * greenS.matrix() + std::get<1>(CandS).matrix() * greenC.conjugate().matrix();
    greenC.swap(tempC);
    greenS.swap(tempS);
  }

  return std::make_pair(std::move(greenC), std::move(greenS));
}


Array2Dcd Cascade::batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs,
                                         bool inTimeDomain, bool runPump, uint nThreads,
                                         uint inputMode, const std::vector<uint8_t>& useOutput) {
  if (runPump) runPumpSimulation();

  Array2Dcd outSignals = media[0].get().batchSignalSimulation(inputProfs, inTimeDomain, false, nThreads, inputMode);

  for (uint i = 1; i < media.size(); i++) {
    // Determine largest input index
    uint maxInputVal = 0;
    for (const auto& connection : connections[i-1]) {
      if (connection.first > maxInputVal)
        maxInputVal = connection.first;
    }

    // Concatenate inputs based on connection map
    Array2Dcd inputToNext = Array2Dcd::Zero(outSignals.rows(), _nFreqs * (maxInputVal + 1));
    for (const auto& connection : connections[i-1]) {
      inputToNext.middleCols(connection.first * _nFreqs, _nFreqs) = media[i-1].get().signalFreq[connection.second].bottomRows<1>();
    }

    if (i != media.size() - 1) {
      outSignals = media[i].get().batchSignalSimulation(outSignals, inTimeDomain, false, nThreads);
    } else {
      outSignals = media[i].get().batchSignalSimulation(outSignals, inTimeDomain, false, nThreads, 0, useOutput);
    }
  }

  return outSignals;
}
