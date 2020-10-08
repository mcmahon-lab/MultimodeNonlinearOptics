#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "NonlinearMedium.hpp"

// Pybind11 Python binding

PYBIND11_MODULE(nonlinearmedium, m) {

  namespace py = pybind11;
  using namespace pybind11::literals;

  py::class_<_NonlinearMedium> _NLMBase(m, "_NonlinearMedium");
  py::class_<_NLM2ModeExtension> _NLM2Mode(m, "_NLM2ModeExtension");
  py::class_<Chi3, _NonlinearMedium> Chi3(m, "Chi3");
  py::class_<_Chi2, _NonlinearMedium> _Chi2Base(m, "_Chi2");
  py::class_<Chi2PDC, _Chi2> Chi2PDC(m, "Chi2PDC");
  py::class_<Chi2SFG, _NLM2ModeExtension, _Chi2> Chi2SFG(m, "Chi2SFG");
  py::class_<Chi2PDCII, _NLM2ModeExtension, _Chi2> Chi2PDCII(m, "Chi2PDCII");
  py::class_<Cascade, _NonlinearMedium> Cascade(m, "Cascade");


  // default arguments for Python initialization of empty arrays
  Eigen::Ref<const Arraycd> defArraycd = Eigen::Ref<const Arraycd>(Arraycd{});
  Eigen::Ref<const Arrayd>  defArrayf  = Eigen::Ref<const Arrayd>(Arrayd{});

  constexpr double infinity = std::numeric_limits<double>::infinity();

/*
 * _NonlinearMedium
 */

  _NLMBase.def("setPump",
               py::overload_cast<int, double>(&_NonlinearMedium::setPump),
               "pulseType"_a, "chirp"_a = 0);

  _NLMBase.def("setPump",
               py::overload_cast<const Eigen::Ref<const Arraycd>&, double>(&_NonlinearMedium::setPump),
               "customPump"_a, "chirp"_a = 0);

  _NLMBase.def("runPumpSimulation", &_NonlinearMedium::runPumpSimulation);

  _NLMBase.def("runSignalSimulation",
               py::overload_cast<const Eigen::Ref<const Arraycd>&, bool>(&_NonlinearMedium::runSignalSimulation),
               "inputProf"_a, "inTimeDomain"_a = true);

  _NLMBase.def("computeGreensFunction",
               &_NonlinearMedium::computeGreensFunction, py::return_value_policy::move,
               "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  _NLMBase.def("batchSignalSimulation",
               &_NonlinearMedium::batchSignalSimulation, py::return_value_policy::move, "inputProfs"_a,
               "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  _NLMBase.def_property_readonly("pumpFreq", &_NonlinearMedium::getPumpFreq, py::return_value_policy::reference);
  _NLMBase.def_property_readonly("pumpTime", &_NonlinearMedium::getPumpTime, py::return_value_policy::reference);
  _NLMBase.def_property_readonly("signalFreq", &_NonlinearMedium::getSignalFreq, py::return_value_policy::reference);
  _NLMBase.def_property_readonly("signalTime", &_NonlinearMedium::getSignalTime, py::return_value_policy::reference);
  _NLMBase.def_property_readonly("omega", &_NonlinearMedium::getFrequency, py::return_value_policy::reference);
  _NLMBase.def_property_readonly("tau", &_NonlinearMedium::getTime, py::return_value_policy::reference);


/*
 * _NLM2ModeExtension
 */

  _NLM2Mode.def("runSignalSimulation",
                py::overload_cast<const Eigen::Ref<const Arraycd>&, bool>(&_NLM2ModeExtension::runSignalSimulation),
                "inputProf"_a, "inTimeDomain"_a = true);

  _NLM2Mode.def("computeTotalGreen",
                &_NLM2ModeExtension::computeTotalGreen, py::return_value_policy::move,
                "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  _NLM2Mode.def_property_readonly("originalFreq", &_NLM2ModeExtension::getOriginalFreq, py::return_value_policy::reference);
  _NLM2Mode.def_property_readonly("originalTime", &_NLM2ModeExtension::getOriginalTime, py::return_value_policy::reference);

/*
 * Chi3
 */

  Chi3.def(
      py::init<double, double, double, double, Eigen::Ref<const Arraycd>&, int, double, double, double, double, uint, uint>(),
      "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "customPump"_a = defArraycd, "pulseType"_a = 0,
      "beta3"_a = 0, "chirp"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100);

  Chi3.def("runPumpSimulation", &Chi3::runPumpSimulation);


/*
 * _Chi2
 */

  _Chi2Base.def_property_readonly("poling", &_Chi2::getPoling, py::return_value_policy::reference);


/*
 * Chi2PDC
 */

  Chi2PDC.def(
      py::init<double, double, double, double, double, Eigen::Ref<const Arraycd>&, int,
               double, double, double, double, double, double, double, double, uint, uint, Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "customPump"_a = defArraycd, "pulseType"_a = 0,
      "beta1"_a = 0, "beta1s"_a = 0, "beta3"_a = 0, "beta3s"_a = 0, "diffBeta0"_a = 0,
      "chirp"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "poling"_a = defArrayf);


/*
 * Chi2SFG
 */

  Chi2SFG.def(
      py::init<double, double, double, double, double, double, double, Eigen::Ref<const Arraycd>&, int,
               double, double, double, double, double, double, double, double, double, double, double, uint, uint,
               Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLength"_a, "nlLengthOrig"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "beta2o"_a,
      "customPump"_a = defArraycd, "pulseType"_a = 0, "beta1"_a = 0, "beta1s"_a = 0, "beta1o"_a = 0, "beta3"_a = 0, "beta3s"_a = 0, "beta3o"_a = 0, "diffBeta0"_a = 0,
      "diffBeta0o"_a = 0, "chirp"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "poling"_a = defArrayf);


/*
 * Chi2PDCII
 */

  Chi2PDCII.def(
      py::init<double, double, double, double, double, double, double, double, Eigen::Ref<const Arraycd>&, int,
               double, double, double, double, double, double, double, double, double, double, double, uint, uint,
               Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLength"_a, "nlLengthOrig"_a, "nlLengthI"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "beta2o"_a,
      "customPump"_a = defArraycd, "pulseType"_a = 0, "beta1"_a = 0, "beta1s"_a = 0, "beta1o"_a = 0, "beta3"_a = 0, "beta3s"_a = 0, "beta3o"_a = 0, "diffBeta0"_a = 0,
      "diffBeta0o"_a = 0, "chirp"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "poling"_a = defArrayf);


/*
 * Cascade
 */

  Cascade.def(py::init<bool, const std::vector<std::reference_wrapper<_NonlinearMedium>>&>(),
              "sharePump"_a, "inputMedia"_a);

  Cascade.def("setPump",
              py::overload_cast<int, double>(&Cascade::setPump),
              "pulseType"_a, "chirp"_a = 0);

  Cascade.def("setPump",
              py::overload_cast<const Eigen::Ref<const Arraycd>&, double>(&Cascade::setPump),
               "customPump"_a, "chirp"_a = 0);

  Cascade.def("runPumpSimulation", &Cascade::runPumpSimulation);

  Cascade.def("runSignalSimulation",
              py::overload_cast<const Eigen::Ref<const Arraycd>&, bool>(&Cascade::runSignalSimulation),
              "inputProf"_a, "inTimeDomain"_a = true);

  Cascade.def("computeGreensFunction",
              &Cascade::computeGreensFunction, py::return_value_policy::move,
              "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Cascade.def("batchSignalSimulation",
              &Cascade::batchSignalSimulation, py::return_value_policy::move,
              "inputProfs"_a, "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Cascade.def("addMedium", &Cascade::addMedium,
              "medium"_a);

  Cascade.def_property_readonly("omega", &Cascade::getFrequency, py::return_value_policy::reference);
  Cascade.def_property_readonly("tau", &Cascade::getTime, py::return_value_policy::reference);

  Cascade.def("__getitem__", &Cascade::getMedium, py::return_value_policy::reference);
  Cascade.def_property_readonly("media", &Cascade::getMedia, py::return_value_policy::reference);
  Cascade.def_property_readonly("nMedia", &Cascade::getNMedia);
}