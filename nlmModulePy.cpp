#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "NonlinearMedium.hpp"

// Pybind11 Python binding

PYBIND11_MODULE(nonlinearmedium, m) {

  namespace py = pybind11;
  using namespace pybind11::literals;

  py::class_<_NonlinearMedium> NonlinearMediumBase(m, "_NonlinearMedium");
  py::class_<Chi3, _NonlinearMedium> Chi3(m, "Chi3");
  py::class_<Chi2PDC, _NonlinearMedium> Chi2PDC(m, "Chi2PDC");
  py::class_<Chi2SFG, _NonlinearMedium> Chi2SFG(m, "Chi2SFG");
  py::class_<Chi2PDCII, _NonlinearMedium> Chi2PDCII(m, "Chi2PDCII");
  py::class_<Cascade, _NonlinearMedium> Cascade(m, "Cascade");


  // default arguments for Python initialization of empty arrays
  Eigen::Ref<const Arraycd> defArraycd = Eigen::Ref<const Arraycd>(Arraycd{});
  Eigen::Ref<const Arrayd>  defArrayf  = Eigen::Ref<const Arrayd>(Arrayd{});

  constexpr double infinity = std::numeric_limits<double>::infinity();


/*
 * Chi3
 */

  Chi3.def(
      py::init<double, double, double, double, Eigen::Ref<const Arraycd>&, int, double, double, double, double, uint, uint>(),
      "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "customPump"_a = defArraycd, "pulseType"_a = 0,
      "beta3"_a = 0, "chirp"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100);

  Chi3.def("setPump",
          (void (Chi3::*)(int, double))&Chi3::setPump,
           "pulseType"_a, "chirp"_a = 0);

  Chi3.def("setPump",
           (void (Chi3::*)(const Eigen::Ref<const Arraycd>&, double))&Chi3::setPump,
           "customPump"_a, "chirp"_a = 0);

  Chi3.def("runPumpSimulation", &Chi3::runPumpSimulation);

  Chi3.def("runSignalSimulation",
           (void (Chi3::*)(Eigen::Ref<const Arraycd>, bool))&Chi3::runSignalSimulation,
           "inputProf"_a, "inTimeDomain"_a = true);

  Chi3.def("computeGreensFunction",
           &Chi3::computeGreensFunction, py::return_value_policy::move,
           "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Chi3.def("batchSignalSimulation",
           &Chi3::batchSignalSimulation, py::return_value_policy::move,
           "inputProfs"_a, "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Chi3.def_property_readonly("pumpFreq", &Chi3::getPumpFreq, py::return_value_policy::reference);
  Chi3.def_property_readonly("pumpTime", &Chi3::getPumpTime, py::return_value_policy::reference);
  Chi3.def_property_readonly("signalFreq", &Chi3::getSignalFreq, py::return_value_policy::reference);
  Chi3.def_property_readonly("signalTime", &Chi3::getSignalTime, py::return_value_policy::reference);
  Chi3.def_property_readonly("omega", &Chi3::getFrequency, py::return_value_policy::reference);
  Chi3.def_property_readonly("tau", &Chi3::getTime, py::return_value_policy::reference);


/*
 * Chi2PDC
 */

  Chi2PDC.def(
      py::init<double, double, double, double, double, Eigen::Ref<const Arraycd>&, int,
               double, double, double, double, double, double, double, double, uint, uint, Eigen::Ref<const Arrayd>&>(),
      "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "customPump"_a = defArraycd, "pulseType"_a = 0,
      "beta1"_a = 0, "beta1s"_a = 0, "beta3"_a = 0, "beta3s"_a = 0, "diffBeta0"_a = 0,
      "chirp"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100, "poling"_a = defArrayf);

  Chi2PDC.def("setPump",
              (void (Chi2PDC::*)(int, double))&Chi2PDC::setPump,
              "pulseType"_a, "chirp"_a = 0);

  Chi2PDC.def("setPump",
              (void (Chi2PDC::*)(const Eigen::Ref<const Arraycd>&, double))&Chi2PDC::setPump,
              "customPump"_a, "chirp"_a = 0);

  Chi2PDC.def("runPumpSimulation", &Chi2PDC::runPumpSimulation);

  Chi2PDC.def("runSignalSimulation",
              (void (Chi2PDC::*)(Eigen::Ref<const Arraycd>, bool))&Chi2PDC::runSignalSimulation,
              "inputProf"_a, "inTimeDomain"_a = true);

  Chi2PDC.def("computeGreensFunction",
              &Chi2PDC::computeGreensFunction, py::return_value_policy::move,
              "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Chi2PDC.def("batchSignalSimulation",
              &Chi2PDC::batchSignalSimulation, py::return_value_policy::move,
              "inputProfs"_a, "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Chi2PDC.def_property_readonly("pumpFreq", &Chi2PDC::getPumpFreq, py::return_value_policy::reference);
  Chi2PDC.def_property_readonly("pumpTime", &Chi2PDC::getPumpTime, py::return_value_policy::reference);
  Chi2PDC.def_property_readonly("signalFreq", &Chi2PDC::getSignalFreq, py::return_value_policy::reference);
  Chi2PDC.def_property_readonly("signalTime", &Chi2PDC::getSignalTime, py::return_value_policy::reference);
  Chi2PDC.def_property_readonly("omega", &Chi2PDC::getFrequency, py::return_value_policy::reference);
  Chi2PDC.def_property_readonly("tau", &Chi2PDC::getTime, py::return_value_policy::reference);
  Chi2PDC.def_property_readonly("poling", &Chi2PDC::getPoling, py::return_value_policy::reference);


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

  Chi2SFG.def("setPump",
              (void (Chi2SFG::*)(int, double))&Chi2SFG::setPump,
              "pulseType"_a, "chirp"_a = 0);

  Chi2SFG.def("setPump",
              (void (Chi2SFG::*)(const Eigen::Ref<const Arraycd>&, double))&Chi2SFG::setPump,
              "customPump"_a, "chirp"_a = 0);

  Chi2SFG.def("runPumpSimulation", &Chi2SFG::runPumpSimulation);

  Chi2SFG.def("runSignalSimulation",
              (void (Chi2SFG::*)(Eigen::Ref<const Arraycd>, bool))&Chi2SFG::runSignalSimulation,
              "inputProf"_a, "inTimeDomain"_a = true);

  Chi2SFG.def("computeGreensFunction",
              &Chi2SFG::computeGreensFunction, py::return_value_policy::move,
              "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Chi2SFG.def("computeTotalGreen",
              &Chi2SFG::computeTotalGreen, py::return_value_policy::move,
              "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Chi2SFG.def("batchSignalSimulation",
              &Chi2SFG::batchSignalSimulation, py::return_value_policy::move,
              "inputProfs"_a, "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Chi2SFG.def_property_readonly("pumpFreq", &Chi2SFG::getPumpFreq, py::return_value_policy::reference);
  Chi2SFG.def_property_readonly("pumpTime", &Chi2SFG::getPumpTime, py::return_value_policy::reference);
  Chi2SFG.def_property_readonly("signalFreq", &Chi2SFG::getSignalFreq, py::return_value_policy::reference);
  Chi2SFG.def_property_readonly("signalTime", &Chi2SFG::getSignalTime, py::return_value_policy::reference);
  Chi2SFG.def_property_readonly("originalFreq", &Chi2SFG::getOriginalFreq, py::return_value_policy::reference);
  Chi2SFG.def_property_readonly("originalTime", &Chi2SFG::getOriginalTime, py::return_value_policy::reference);
  Chi2SFG.def_property_readonly("omega", &Chi2SFG::getFrequency, py::return_value_policy::reference);
  Chi2SFG.def_property_readonly("tau", &Chi2SFG::getTime, py::return_value_policy::reference);
  Chi2SFG.def_property_readonly("poling", &Chi2SFG::getPoling, py::return_value_policy::reference);


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

  Chi2PDCII.def("setPump",
               (void (Chi2PDCII::*)(int, double))&Chi2PDCII::setPump,
              " pulseType"_a, "chirp"_a = 0);

  Chi2PDCII.def("setPump",
               (void (Chi2PDCII::*)(const Eigen::Ref<const Arraycd>&, double))&Chi2PDCII::setPump,
               "customPump"_a, "chirp"_a = 0);

  Chi2PDCII.def("runPumpSimulation", &Chi2PDCII::runPumpSimulation);

  Chi2PDCII.def("runSignalSimulation",
               (void (Chi2PDCII::*)(Eigen::Ref<const Arraycd>, bool))&Chi2PDCII::runSignalSimulation,
               "inputProf"_a, "inTimeDomain"_a = true);

  Chi2PDCII.def("computeGreensFunction",
               &Chi2PDCII::computeGreensFunction, py::return_value_policy::move,
               "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Chi2PDCII.def("computeTotalGreen",
               &Chi2PDCII::computeTotalGreen, py::return_value_policy::move,
               "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Chi2PDCII.def("batchSignalSimulation",
               &Chi2PDCII::batchSignalSimulation, py::return_value_policy::move,
              " inputProfs"_a, "inTimeDomain"_a = false, "runPump"_a = true, "nThreads"_a = 1);

  Chi2PDCII.def_property_readonly("pumpFreq", &Chi2PDCII::getPumpFreq, py::return_value_policy::reference);
  Chi2PDCII.def_property_readonly("pumpTime", &Chi2PDCII::getPumpTime, py::return_value_policy::reference);
  Chi2PDCII.def_property_readonly("signalFreq", &Chi2PDCII::getSignalFreq, py::return_value_policy::reference);
  Chi2PDCII.def_property_readonly("signalTime", &Chi2PDCII::getSignalTime, py::return_value_policy::reference);
  Chi2PDCII.def_property_readonly("originalFreq", &Chi2PDCII::getOriginalFreq, py::return_value_policy::reference);
  Chi2PDCII.def_property_readonly("originalTime", &Chi2PDCII::getOriginalTime, py::return_value_policy::reference);
  Chi2PDCII.def_property_readonly("omega", &Chi2PDCII::getFrequency, py::return_value_policy::reference);
  Chi2PDCII.def_property_readonly("tau", &Chi2PDCII::getTime, py::return_value_policy::reference);
  Chi2PDCII.def_property_readonly("poling", &Chi2PDCII::getPoling, py::return_value_policy::reference);


/*
 * Cascade
 */

  Cascade.def(py::init<bool, const std::vector<std::reference_wrapper<_NonlinearMedium>>&>(),
              "sharePump"_a, "inputMedia"_a);

  Cascade.def("runPumpSimulation", &Cascade::runPumpSimulation);

  Cascade.def("runSignalSimulation",
              (void (Cascade::*)(Eigen::Ref<const Arraycd>, bool))&Cascade::runSignalSimulation,
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