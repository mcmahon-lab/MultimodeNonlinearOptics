#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "_NonlinearMedium.hpp"
#include "_FullyNonlinearMedium.hpp"
#include "Cascade.hpp"
#include "RegisteredSolvers.hpp"

// Pybind11 Python binding
#define NLMMODULE
PYBIND11_MODULE(nonlinearmedium, m) {

  namespace py = pybind11;
  using namespace pybind11::literals;

  m.doc() = "Module for numerical simulation of classical or quantum signals in pumped nonlinear media.";

  py::class_<_NonlinearMedium> _NLMBase(m, "_NonlinearMedium", "Base class for nonlinear medium solvers");
  py::class_<Cascade, _NonlinearMedium> Cascade(m, "Cascade");
  py::class_<_FullyNonlinearMedium, _NonlinearMedium> _FNLMBase(m, "_FullyNonlinearMedium", "Base class for fully nonlinear medium solvers");

  // default arguments for Python, including initialization of empty arrays
  Eigen::Ref<const Arraycd> defArraycd = Eigen::Ref<const Arraycd>(Arraycd{});
  Eigen::Ref<const Arrayd>  defArrayf  = Eigen::Ref<const Arrayd>(Arrayd{});
  const std::vector<uint8_t> defCharVec = {};
  constexpr double infinity = std::numeric_limits<double>::infinity();

/*
 * _NonlinearMedium
 */

  _NLMBase.def("setPump",
               py::overload_cast<_NonlinearMedium::PulseType, double, double, uint>(&_NonlinearMedium::setPump),
               "Set the input shape of the pump\n"
               "pulseType Gaussian, Sech or Sinc profile; 0, 1, 2 respectively.\n"
               "chirp     Initial chirp of the pump, specified in dispersion lengths.\n"
               "delay     Initial time delay of the pump, specified in walk-off lengths.\n"
               "pumpIndex Index of the pump being set, if applicable.",
               "pulseType"_a, "chirp"_a = 0, "delay"_a = 0, "pumpIndex"_a = 0);

  _NLMBase.def("setPump",
               py::overload_cast<const Eigen::Ref<const Arraycd>&, double, double, uint>(&_NonlinearMedium::setPump),
               "Set the input shape of the pump\n"
               "customPump An arbitrary pump shape specified in the time domain, with self.tau as the axis.\n"
               "chirp     Initial chirp of the pump, specified in dispersion lengths.\n"
               "delay     Initial time delay of the pump, specified in walk-off lengths.\n"
               "pumpIndex Index of the pump being set, if applicable.",
               "customPump"_a, "chirp"_a = 0, "delay"_a = 0, "pumpIndex"_a = 0);

  _NLMBase.def("setPump",
               py::overload_cast<const _NonlinearMedium&, uint, double, uint>(&_NonlinearMedium::setPump),
               "Set the pump over the whole propagation length by copying from another simulation, accounting for the frame of reference\n"
               "Note: do not call runPumpSimulation after this function or the pump simulation will be overwritten.\n"
               "other     A NonlinearMedium instance with the same frequency axis, and resolution greater than or equal to the pump's.\n"
               "modeIndex The index of the mode in 'other' to use as pump in this simulation.\n"
               "delay     Initial time delay of the pump, specified in walk-off lengths.\n"
               "pumpIndex Index of the pump being set, if applicable.",
               "other"_a, "modeIndex"_a = 0, "delay"_a = 0, "pumpIndex"_a = 0);

  _NLMBase.def("runPumpSimulation", &_NonlinearMedium::runPumpSimulation,
               "Simulate propagation of the pump field");

  _NLMBase.def("runSignalSimulation",
               py::overload_cast<const Eigen::Ref<const Arraycd>&, bool, uint>(&_NonlinearMedium::runSignalSimulation),
               "Simulate propagation of the signal field(s)\n"
               "inputProf    Profile of input pulse. May be time or frequency domain.\n"
               "             Note: Input is assumed to have self.omega or self.tau as the axis.\n"
               "inTimeDomain Specify if the input is in time or frequency domain.\n"
               "inputMode    Specify which signal mode of the nonlinear medium the input corresponds to.",
               "inputProf"_a, "inTimeDomain"_a = true, "inputMode"_a = 0);

  _NLMBase.def("computeGreensFunction",
               &_NonlinearMedium::computeGreensFunction, py::return_value_policy::move,
               "Solve the Green's function matrices C and S where a(L) = C a(0) + S [a(0)]^t\n"
               "inTimeDomain Compute the Green's function in time or frequency domain.\n"
               "runPump      Whether to run the pump simulation before signal simulations.\n"
               "nThreads     Number of threads used to run simulations in parallel.\n"
               "normalize    Whether to adjust the final matrix so that amplitudes are converted 1-to-1 between different modes.\n"
               "useInput     Specify which inputs modes to include in the transformation. Default is all inputs.\n"
               "useOutput    Specify which output modes to include in the transformation. Default is all outputs.\n"
               "return: Green's function matrices C, S",
               "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1, "normalize"_a = false,
               "useInput"_a = defCharVec, "useOutput"_a = defCharVec);

  _NLMBase.def("batchSignalSimulation",
               &_NonlinearMedium::batchSignalSimulation, py::return_value_policy::move,
               "Run multiple signal simulations.\n"
               "inputProfs   Profiles of input pulses. May be time or frequency domain.\n"
               "             Note: Input is assumed to have self.omega or self.tau as the axis.\n"
               "inTimeDomain Specify if the input is in time or frequency domain.\n"
               "runPump      Whether to run the pump simulation before signal simulations.\n"
               "nThreads     Number of threads used to run simulations in parallel.\n"
               "inputMode    Specify which signal mode of the nonlinear medium the input corresponds to.\n"
               "useOutput    Specify which signal mode outputs to return. Default is all outputs.\n"
               "return: Array of signal profiles at the output of the medium",
               "inputProfs"_a, "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1,
               "inputMode"_a = 0, "useOutput"_a = defCharVec);

  _NLMBase.def_property_readonly("omega", &_NonlinearMedium::getFrequency, py::return_value_policy::reference,
                                 "Read-only frequency axis of the system.");
  _NLMBase.def_property_readonly("tau", &_NonlinearMedium::getTime, py::return_value_policy::reference,
                                 "Read-only time axis of the system");
  _NLMBase.def_property_readonly("signalFreq", [](_NonlinearMedium& nlm){return nlm.getSignalFreq();}, py::return_value_policy::reference,
                                 "Read-only array of the signal frequency profile along the length of the medium.");
  _NLMBase.def_property_readonly("signalTime", [](_NonlinearMedium& nlm){return nlm.getSignalTime();}, py::return_value_policy::reference,
                                 "Read-only array of a signal time profile along the length of the medium.");
  _NLMBase.def_property_readonly("pumpFreq", [](_NonlinearMedium& nlm){return nlm.getPumpFreq();}, py::return_value_policy::reference,
                                 "Read-only array of the pump frequency profile along the length of the medium.");
  _NLMBase.def_property_readonly("pumpTime", [](_NonlinearMedium& nlm){return nlm.getPumpTime();}, py::return_value_policy::reference,
                                 "Read-only array of the pump time profile along the length of the medium.");
  _NLMBase.def("signalFreqs", &_NonlinearMedium::getSignalFreq, py::return_value_policy::reference,
               "Read-only array of a signal frequency profile along the length of the medium.", "i"_a = 0);
  _NLMBase.def("signalTimes", &_NonlinearMedium::getSignalTime, py::return_value_policy::reference,
               "Read-only array of a signal time profile along the length of the medium.", "i"_a = 0);
  _NLMBase.def("pumpFreqs", &_NonlinearMedium::getPumpFreq, py::return_value_policy::reference,
               "Read-only array of a pump frequency profile along the length of the medium.", "i"_a = 0);
  _NLMBase.def("pumpTimes", &_NonlinearMedium::getPumpTime, py::return_value_policy::reference,
               "Read-only array of a pump time profile along the length of the medium.", "i"_a = 0);
  _NLMBase.def_property_readonly("poling", &_NonlinearMedium::getPoling, py::return_value_policy::reference,
                                 "Read-only array of the domain poling along the length of a Chi(2) medium.");

  py::enum_<_NonlinearMedium::PulseType>(m, "PulseType")
      .value("Gaussian", _NonlinearMedium::PulseType::Gaussian)
      .value("Sech", _NonlinearMedium::PulseType::Sech)
      .value("Sinc", _NonlinearMedium::PulseType::Sinc)
      .export_values();
  py::enum_<_NonlinearMedium::IntensityProfile>(m, "IntensityProfile")
      .value("GaussianBeam", _NonlinearMedium::IntensityProfile::GaussianBeam)
      .value("Constant", _NonlinearMedium::IntensityProfile::Constant)
      .value("GaussianApodization", _NonlinearMedium::IntensityProfile::GaussianApodization)
      .export_values();

/*
 * _FullyNonlinearMedium
 */

  _FNLMBase.def("batchSignalSimulation",
                py::overload_cast<const Eigen::Ref<const Array2Dcd>&, bool, uint, uint, const std::vector<uint8_t>&>(
                    &_FullyNonlinearMedium::batchSignalSimulation),
                py::return_value_policy::move,
                "Run multiple signal simulations.\n"
                "inputProfs   Profiles of input pulses. May be time or frequency domain.\n"
                "             Note: Input is assumed to have self.omega or self.tau as the axis.\n"
                "inTimeDomain Specify if the input is in time or frequency domain.\n"
                "nThreads     Number of threads used to run simulations in parallel.\n"
                "inputMode    Specify which signal mode of the nonlinear medium the input corresponds to.\n"
                "useOutput    Specify which signal mode outputs to return. Default is all outputs.\n"
                "return: Array of signal profiles at the output of the medium",
                "inputProfs"_a, "inTimeDomain"_a = false, "nThreads"_a = 1,
                "inputMode"_a = 0, "useOutput"_a = defCharVec);

  _FNLMBase.def("setPump", py::overload_cast<_NonlinearMedium::PulseType, double, double, uint>(&_FullyNonlinearMedium::setPump),
                "pulseType"_a, "chirp"_a = 0, "delay"_a = 0, "pumpIndex"_a = 0);
  _FNLMBase.def("setPump", py::overload_cast<const Eigen::Ref<const Arraycd>&, double, double, uint>(&_FullyNonlinearMedium::setPump),
                "customPump"_a, "chirp"_a = 0, "delay"_a = 0, "pumpIndex"_a = 0);
  _FNLMBase.def("setPump", py::overload_cast<const _NonlinearMedium&, uint, double, uint>(&_FullyNonlinearMedium::setPump),
                "other"_a, "signalIndex"_a = 0, "delayLength"_a = 0, "pumpIndex"_a = 0);
  _FNLMBase.def("runPumpSimulation", &_FullyNonlinearMedium::runPumpSimulation);
  _FNLMBase.def("computeGreensFunction", &_FullyNonlinearMedium::computeGreensFunction,
                "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1, "normalize"_a = false,
                "useInput"_a = defCharVec, "useOutput"_a = defCharVec);

/*
 * Cascade
 */

  Cascade.def(py::init<const std::vector<std::reference_wrapper<_NonlinearMedium>>&, std::vector<std::map<uint, uint>>&, bool>(),
              "inputMedia"_a, "modeConnections"_a, "sharePump"_a,
              py::keep_alive<1, 2>());

  Cascade.def("setPump",
              py::overload_cast<_NonlinearMedium::PulseType, double, double, uint>(&Cascade::setPump),
              "pulseType"_a, "chirp"_a = 0, "delay"_a = 0, "pumpIndex"_a = 0);

  Cascade.def("setPump",
              py::overload_cast<const Eigen::Ref<const Arraycd>&, double, double, uint>(&Cascade::setPump),
               "customPump"_a, "chirp"_a = 0, "delay"_a = 0, "pumpIndex"_a = 0);

  Cascade.def("setPump",
              py::overload_cast<const _NonlinearMedium&, uint, double, uint>(&Cascade::setPump),
              "other"_a, "signalIndex"_a = 0, "delayLength"_a = 0, "pumpIndex"_a = 0);

  Cascade.def("runPumpSimulation", &Cascade::runPumpSimulation);

  Cascade.def("runSignalSimulation",
              py::overload_cast<const Eigen::Ref<const Arraycd>&, bool, uint>(&Cascade::runSignalSimulation),
              "inputProf"_a, "inTimeDomain"_a = true, "inputMode"_a = 0);

  Cascade.def("computeGreensFunction",
              &Cascade::computeGreensFunction, py::return_value_policy::move,
              "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1, "normalize"_a = false,
              "useInput"_a = defCharVec, "useOutput"_a = defCharVec);

  Cascade.def("batchSignalSimulation",
              &Cascade::batchSignalSimulation, py::return_value_policy::move,
              "inputProfs"_a, "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1,
              "inputMode"_a = 0, "useOutput"_a = defCharVec);

  Cascade.def("addMedium", &Cascade::addMedium,
              "medium"_a, "connection"_a,
              py::keep_alive<1, 2>());

  Cascade.def_property_readonly("omega", &Cascade::getFrequency, py::return_value_policy::reference);
  Cascade.def_property_readonly("tau", &Cascade::getTime, py::return_value_policy::reference);

  Cascade.def("__getitem__", &Cascade::getMedium, py::return_value_policy::reference);
  Cascade.def_property_readonly("media", &Cascade::getMedia, py::return_value_policy::reference);
  Cascade.def_property_readonly("nMedia", &Cascade::getNMedia);

/*
 * Solvers
 */
  #include "RegisteredSolvers.hpp"
}
#undef NLMMODULE