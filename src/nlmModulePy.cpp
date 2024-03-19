#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "_NonlinearMedium.hpp"
#include "_FullyNonlinearMedium.hpp"
#include "Cascade.hpp"
#include "Chi3.cpp"
#include "Chi2PDC.cpp"
#include "Chi2SFG.cpp"
#include "Chi2AFC.cpp"
#include "Chi2SFGPDC.cpp"
#include "Chi2SFGII.cpp"
#include "Chi2PDCII.cpp"
#include "Chi2SFGOPA.cpp"
#include "Chi2SHG.cpp"
#include "Chi2SHGOPA.cpp"
#include "Chi2DSFG.cpp"
#include "Chi2SFGXPM.cpp"
#include "Chi2SHGXPM.cpp"
#include "Chi2ASHG.cpp"
#include "Chi3GNLSE.cpp"

// Pybind11 Python binding

PYBIND11_MODULE(nonlinearmedium, m) {

  namespace py = pybind11;
  using namespace pybind11::literals;

  m.doc() = "Module for numerical simulation of classical or quantum signals in pumped nonlinear media.";

  py::class_<_NonlinearMedium> _NLMBase(m, "_NonlinearMedium", "Base class for nonlinear medium solvers");
  py::class_<Chi3, _NonlinearMedium> Chi3(m, "Chi3", "Single mode self phase modulation");
  py::class_<Chi2PDC, _NonlinearMedium> Chi2PDC(m, "Chi2PDC", "Degenerate optical parametric amplification");
  py::class_<Chi2SFGPDC, _NonlinearMedium> Chi2SFGPDC(m, "Chi2SFGPDC", "Simultaneous sum frequency generation and parametric amplification");
  py::class_<Chi2SFG, _NonlinearMedium> Chi2SFG(m, "Chi2SFG", "Sum (or difference) frequency generation");
  py::class_<Chi2AFC, _NonlinearMedium> Chi2AFC(m, "Chi2AFC", "Adiabatic sum (or difference) frequency generation, in a rotating frame with linearly varying poling frequency built-in to the solver");
  py::class_<Chi2PDCII, _NonlinearMedium> Chi2PDCII(m, "Chi2PDCII", "Type II or nondegenerate optical parametric amplification");
  py::class_<Chi2SFGII, _NonlinearMedium> Chi2SFGII(m, "Chi2SFGII", "Type II or simultaneous 2-mode sum frequency generation with parametric amplification");
  py::class_<Chi2SFGXPM, _NonlinearMedium> Chi2SFGXPM(m, "Chi2SFGXPM", "Sum (or difference) frequency generation with cross phase modulation");
  py::class_<Chi2SFGOPA, _NonlinearMedium> Chi2SFGOPA(m, "Chi2SFGOPA", "Simultaneous sum frequency generation and non-degenerate optical parametric amplification with two pumps");

  py::class_<Cascade, _NonlinearMedium> Cascade(m, "Cascade");

  py::class_<_FullyNonlinearMedium, _NonlinearMedium> _FNLMBase(m, "_FullyNonlinearMedium", "Base class for fully nonlinear medium solvers");
  py::class_<Chi2SHG, _FullyNonlinearMedium> Chi2SHG(m, "Chi2SHG", "Fully nonlinear second harmonic generation");
  py::class_<Chi2SHGOPA, _FullyNonlinearMedium> Chi2SHGOPA(m, "Chi2SHGOPA", "Fully nonlinear OPA driven by the second harmonic of the pump");
  py::class_<Chi2DSFG, _FullyNonlinearMedium> Chi2DSFG(m, "Chi2DSFG", "Sum (or difference) frequency generation with pump depletion (fully nonlinear)");
  py::class_<Chi2SHGXPM, _FullyNonlinearMedium> Chi2SHGXPM(m, "Chi2SHGXPM", "Fully nonlinear second harmonic generation with self and cross phase modulation");
  py::class_<Chi2ASHG, _FullyNonlinearMedium> Chi2ASHG(m, "Chi2ASHG", "Fully nonlinear adiabatic second harmonic generation");
  py::class_<Chi3GNLSE, _FullyNonlinearMedium> Chi3GNLSE(m, "Chi3GNLSE", "Fully nonlinear general nonlinear Schrodinger equation");

  // default arguments for Python, including initialization of empty arrays
  Eigen::Ref<const Arraycd> defArraycd = Eigen::Ref<const Arraycd>(Arraycd{});
  Eigen::Ref<const Arrayd>  defArrayf  = Eigen::Ref<const Arrayd>(Arrayd{});
  const std::vector<char> defCharVec = {};
  constexpr double infinity = std::numeric_limits<double>::infinity();

/*
 * _NonlinearMedium
 */

  _NLMBase.def("setPump",
               py::overload_cast<int, double, double, uint>(&_NonlinearMedium::setPump),
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


/*
 * _FullyNonlinearMedium
 */

  _FNLMBase.def("batchSignalSimulation",
                py::overload_cast<const Eigen::Ref<const Array2Dcd>&, bool, uint, uint, const std::vector<char>&>(
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

  _FNLMBase.def("setPump", py::overload_cast<int, double, double, uint>(&_FullyNonlinearMedium::setPump),
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
 * Chi3
 */

  Chi3.def(
      py::init<double, double, double, Eigen::Ref<const Arraycd>&, int, double, double, double, uint, uint, double>(),
      "relativeLength"_a, "nlLength"_a, "beta2"_a, "customPump"_a = defArraycd, "pulseType"_a = 0,
      "beta3"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "chirp"_a = 0);

  Chi3.def("runPumpSimulation", &Chi3::runPumpSimulation,
           "Simulate propagation the of pump field");


/*
 * Chi2PDC
 */

  Chi2PDC.def(
      py::init<double, double, double, double, Eigen::Ref<const Arraycd>&, int,
               double, double, double, double, double, double, double, uint, uint, double, double, Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLength"_a, "beta2"_a, "beta2s"_a, "customPump"_a = defArraycd, "pulseType"_a = 0,
      "beta1"_a = 0, "beta1s"_a = 0, "beta3"_a = 0, "beta3s"_a = 0, "diffBeta0"_a = 0, "rayleighLength"_a = infinity,
      "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "chirp"_a = 0, "delay"_a = 0, "poling"_a = defArrayf);


/*
 * Chi2SHG
 */

  Chi2SHG.def(
      py::init<double, double, double, double, double, double, double, double, double, double,
               double, double, uint, uint, const Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLengthH"_a, "nlLengthP"_a, "beta2h"_a, "beta2p"_a, "beta1h"_a = 0, "beta1p"_a = 0,
      "beta3h"_a = 0, "beta3p"_a = 0, "diffBeta0"_a = 0, "rayleighLength"_a = infinity,
      "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "poling"_a = defArrayf);


/*
 * Chi2SHGXPM
 */

  Chi2SHGXPM.def(
      py::init<double, double, double, double, double, double, double, double, double, double, double,
          double, double, uint, uint, const Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLengthH"_a, "nlLengthP"_a, "nlLengthChi3"_a, "beta2h"_a, "beta2p"_a, "beta1h"_a = 0, "beta1p"_a = 0,
      "beta3h"_a = 0, "beta3p"_a = 0, "diffBeta0"_a = 0, "rayleighLength"_a = infinity,
      "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "poling"_a = defArrayf);


/*
 * Chi2DSFG
 */

  Chi2DSFG.def(
      py::init<double, double, double, double, double, double, double, double, double, double, double, double, double,
               double, double, double, uint, uint, const Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLengthP"_a, "nlLengthS"_a, "nlLengthD"_a, "beta2p"_a, "beta2s"_a, "beta2d"_a,
      "beta1p"_a = 0, "beta1s"_a = 0, "beta1d"_a = 0, "beta3p"_a = 0, "beta3s"_a = 0, "beta3d"_a = 0, "diffBeta0"_a = 0,
      "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "poling"_a = defArrayf);


/*
 * Chi2SFGPDC
 */

  Chi2SFGPDC.def(
      py::init<double, double, double, double, double, double, Eigen::Ref<const Arraycd>&, int, double, double, double,
          double, double, double, double, double, double, double, uint, uint, double, double, Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLength"_a, "nlLengthOrig"_a, "beta2"_a, "beta2s"_a, "beta2o"_a, "customPump"_a = defArraycd,
      "pulseType"_a = 0, "beta1"_a = 0, "beta1s"_a = 0, "beta1o"_a = 0, "beta3"_a = 0, "beta3s"_a = 0, "beta3o"_a = 0,
      "diffBeta0"_a = 0, "diffBeta0o"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512,
      "zPrecision"_a = 100, "chirp"_a = 0, "delay"_a = 0, "poling"_a = defArrayf);


/*
 * Chi2SFG
 */

  Chi2SFG.def(
      py::init<double, double, double, double, double, double, Eigen::Ref<const Arraycd>&, int, double, double, double,
               double, double, double, double, double, double, uint, uint, double, double, Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLength"_a, "nlLengthOrig"_a, "beta2"_a, "beta2s"_a, "beta2o"_a,
      "customPump"_a = defArraycd, "pulseType"_a = 0, "beta1"_a = 0, "beta1s"_a = 0, "beta1o"_a = 0, "beta3"_a = 0, "beta3s"_a = 0,
      "beta3o"_a = 0, "diffBeta0"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100,
      "chirp"_a = 0, "delay"_a = 0, "poling"_a = defArrayf);


/*
 * Chi2AFC
 */

  Chi2AFC.def(
      py::init<double, double, double, double, double, double, Eigen::Ref<const Arraycd>&, int, double, double, double,
               double, double, double, double, double, double, double, uint, uint, double, double>(),
      "relativeLength"_a, "nlLength"_a, "nlLengthOrig"_a, "beta2"_a, "beta2s"_a, "beta2o"_a,
      "customPump"_a = defArraycd, "pulseType"_a = 0, "beta1"_a = 0, "beta1s"_a = 0, "beta1o"_a = 0, "beta3"_a = 0, "beta3s"_a = 0,
      "beta3o"_a = 0, "diffBeta0Start"_a = 0, "diffBeta0End"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100,
      "chirp"_a = 0, "delay"_a = 0);


/*
 * Chi2PDCII
 */

  Chi2PDCII.def(
      py::init<double, double, double, double, double, double, Eigen::Ref<const Arraycd>&, int, double, double, double,
               double, double, double, double, double, double, uint, uint, double, double, Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLengthI"_a, "nlLengthII"_a, "beta2"_a, "beta2s"_a, "beta2o"_a,
      "customPump"_a = defArraycd, "pulseType"_a = 0, "beta1"_a = 0, "beta1s"_a = 0, "beta1o"_a = 0, "beta3"_a = 0,
      "beta3s"_a = 0, "beta3o"_a = 0, "diffBeta0"_a = 0, "rayleighLength"_a = infinity,
      "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "chirp"_a = 0, "delay"_a = 0, "poling"_a = defArrayf);


/*
 * Chi2SFGII
 */

  Chi2SFGII.def(
      py::init<double, double, double, double, double, double, double, double, double, double, Eigen::Ref<const Arraycd>&,
               int, double, double, double, double, double, double, double, double, double, double, double, double,
               double, double, double, uint, uint, double, double, Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLengthSignZ"_a, "nlLengthSignY"_a, "nlLengthOrigZ"_a, "nlLengthOrigY"_a,
      "beta2"_a, "beta2sz"_a, "beta2sy"_a, "beta2oz"_a, "beta2oy"_a, "customPump"_a = defArraycd, "pulseType"_a = 0,
      "beta1"_a = 0, "beta1sz"_a = 0, "beta1sy"_a = 0, "beta1oz"_a = 0, "beta1oy"_a = 0, "beta3"_a = 0,
      "beta3sz"_a = 0, "beta3sy"_a = 0, "beta3oz"_a = 0, "beta3oy"_a = 0, "diffBeta0z"_a = 0, "diffBeta0y"_a = 0,
      "diffBeta0s"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100,
      "chirp"_a = 0, "delay"_a = 0, "poling"_a = defArrayf);


/*
 * Chi2SFGOPA
 */

  Chi2SFGOPA.def(
      py::init<double, double, double, double, double, double, double, double, double, const Eigen::Ref<const Arraycd>&,
               int, double, double, double, double, double, double, double, double, double, double, double,
               double, double, uint, uint, double, double, const Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLengthFh"_a, "nlLengthHh"_a, "nlLengthFf"_a, "nlLengthHf"_a, "beta2F"_a, "beta2H"_a, "beta2h"_a,
      "beta2f"_a, "customPump"_a = defArraycd, "pulseType"_a = 0, "beta1F"_a = 0, "beta1H"_a = 0, "beta1h"_a = 0, "beta1f"_a = 0,
      "beta3F"_a = 0, "beta3H"_a = 0, "beta3h"_a = 0, "beta3f"_a = 0, "diffBeta0SFG"_a = 0, "diffBeta0OPA"_a = 0, "diffBeta0DOPA"_a = 0,
      "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "chirp"_a = 0, "delay"_a = 0, "poling"_a = defArrayf);


/*
 * Chi2SHGOPA
 */

  Chi2SHGOPA.def(
      py::init<double, double, double, double, double, double, double, double, double, double, double, double, double,
          double, double, double, double, double, double, double, double, uint , uint , const Eigen::Ref<const Arrayd>& >(),
      "relativeLength"_a, "nlLengthP"_a, "nlLengthSH"_a, "nlLengthPA1"_a, "nlLengthPA2"_a,
      "beta2p"_a, "beta2sh"_a, "beta2pa1"_a, "beta2pa2"_a, "beta1p"_a = 0, "beta1sh"_a = 0, "beta1pa1"_a = 0,
      "beta1pa2"_a = 0, "beta3p"_a = 0, "beta3sh"_a = 0, "beta3pa1"_a = 0, "beta3pa2"_a = 0, "diffBeta0shg"_a = 0,
      "diffBeta0opa"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "poling"_a = defArrayf);


/*
 * Chi2SFGXPM
 */

  Chi2SFGXPM.def(
      py::init<double, double, double, double, double, double, double, double, Eigen::Ref<const Arraycd>&, int, double, double, double,
          double, double, double, double, double, double, uint, uint, double, double, Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLength"_a, "nlLengthOrig"_a, "nlLengthChi3"_a, "nlLengthChi3Orig"_a, "beta2"_a, "beta2s"_a, "beta2o"_a,
      "customPump"_a = defArraycd, "pulseType"_a = 0, "beta1"_a = 0, "beta1s"_a = 0, "beta1o"_a = 0, "beta3"_a = 0, "beta3s"_a = 0,
      "beta3o"_a = 0, "diffBeta0"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100,
      "chirp"_a = 0, "delay"_a = 0, "poling"_a = defArrayf);


/*
 * Chi2ASHG
 */

  Chi2ASHG.def(
      py::init<double, double, double, double, double, double, double, double, double, double, double,
          double, double, uint, uint>(),
      "relativeLength"_a, "nlLengthH"_a, "nlLengthP"_a, "beta2h"_a, "beta2p"_a, "beta1h"_a = 0, "beta1p"_a = 0,
      "beta3h"_a = 0, "beta3p"_a = 0, "diffBeta0Start"_a = 0, "diffBeta0End"_a = 0, "rayleighLength"_a = infinity,
      "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100);


/*
 * Chi3GNLSE
 */

  Chi3GNLSE.def(
      py::init<double, double, double, double, double, double, double, double, double, double, double, double,
          uint, uint>(),
      "relativeLength"_a, "nlLength"_a, "selfSteepLength"_a,  "fr"_a, "fb"_a, "tau1"_a, "tau2"_a, "tau3"_a,
      "beta2"_a, "beta3"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512,"zPrecision"_a = 100);

  Chi3GNLSE.def_property_readonly("ramanResponse", &Chi3GNLSE::getRamanResponse, py::return_value_policy::reference,
                                  "Read-only array of the Raman response function in the frequency domain.");


/*
 * Cascade
 */

  Cascade.def(py::init<const std::vector<std::reference_wrapper<_NonlinearMedium>>&, std::vector<std::map<uint, uint>>&, bool>(),
              "inputMedia"_a, "modeConnections"_a, "sharePump"_a,
              py::keep_alive<1, 2>());

  Cascade.def("setPump",
              py::overload_cast<int, double, double, uint>(&Cascade::setPump),
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
}