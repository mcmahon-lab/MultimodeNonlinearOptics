#include "NonlinearMedium.hpp"

// Pybind11 Python binding

PYBIND11_MODULE(nonlinearmedium, m) {

  namespace py = pybind11;
  using namespace pybind11::literals;

  py::class_<Chi3> Chi3(m, "Chi3");
  py::class_<Chi2> Chi2(m, "Chi2");
  py::class_<Cascade> Cascade(m, "Cascade");

/*
 * Chi3
 */

  Chi3.def(
      py::init<double, double, double, double, double, int, double, double, double, double, double, double, uint, uint>(),
      "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "pulseType"_a = 0,
      "beta1"_a = 0, "beta1s"_a = 0, "beta3"_a = 0, "beta3s"_a = 0,
      "chirp"_a = 0, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100);

  Chi3.def(
      py::init<double, double, double, double, double, Eigen::Ref<const Arraycd>&, int, double, double, double, double, double, double, uint, uint>(),
      "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "customPump"_a, "pulseType"_a = 0,
      "beta1"_a = 0, "beta1s"_a = 0, "beta3"_a = 0, "beta3s"_a = 0,
      "chirp"_a = 0, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100);

  Chi3.def("setLengths", &Chi3::setLengths,
           "relativeLength"_a, "nlLength"_a, "dispLength"_a, "zPrecision"_a = 100);

  Chi3.def("resetGrids", &Chi3::resetGrids,
           "nFreqs"_a = 0, "tMax"_a = 0);

  Chi3.def("setDispersion", &Chi3::setDispersion,
           "beta2"_a, "beta2s"_a, "beta1"_a = 0, "beta1s"_a = 0, "beta3"_a = 0, "beta3s"_a = 0);

  Chi3.def("setPump", (void (Chi3::*)(int, double))&Chi3::setPump,
           "pulseType"_a, "chirp"_a = 0);

  Chi3.def("setPump", (void (Chi3::*)(const Eigen::Ref<const Arraycd>&, double))&Chi3::setPump,
           "customPump"_a, "chirp"_a = 0);

  Chi3.def("runPumpSimulation", &Chi3::runPumpSimulation);

  Chi3.def("runSignalSimulation", &Chi3::runSignalSimulation,
           "inputProf"_a, "timeSignal"_a = true);

  Chi3.def("computeGreensFunction", &Chi3::computeGreensFunction, py::return_value_policy::move,
           "inTimeDomain"_a = false, "runPump"_a = true);

  Chi3.def_property_readonly("pumpFreq", &Chi3::getPumpFreq, py::return_value_policy::reference);
  Chi3.def_property_readonly("pumpTime", &Chi3::getPumpTime, py::return_value_policy::reference);
  Chi3.def_property_readonly("signalFreq", &Chi3::getSignalFreq, py::return_value_policy::reference);
  Chi3.def_property_readonly("signalTime", &Chi3::getSignalTime, py::return_value_policy::reference);
  Chi3.def_property_readonly("omega", &Chi3::getFrequency, py::return_value_policy::reference);
  Chi3.def_property_readonly("tau", &Chi3::getTime, py::return_value_policy::reference);


/*
 * Chi2
 */

  Chi2.def(
      py::init<double, double, double, double, double, int, double, double, double, double, double, double, uint, uint>(),
      "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "pulseType"_a = 0,
      "beta1"_a = 0, "beta1s"_a = 0, "beta3"_a = 0, "beta3s"_a = 0,
      "chirp"_a = 0, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100);

  Chi2.def(py::init<double, double, double, double, double, Eigen::Ref<const Arraycd>&,
                    int, double, double, double, double, double, double, uint, uint>(),
           "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "customPump"_a, "pulseType"_a = 0,
           "beta1"_a = 0, "beta1s"_a = 0, "beta3"_a = 0, "beta3s"_a = 0,
           "chirp"_a = 0, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100);

  Chi2.def("setLengths", &Chi2::setLengths,
           "relativeLength"_a, "nlLength"_a, "dispLength"_a, "zPrecision"_a = 100);

  Chi2.def("resetGrids", &Chi2::resetGrids,
           "nFreqs"_a = 0, "tMax"_a = 0);

  Chi2.def("setDispersion", &Chi2::setDispersion,
           "beta2"_a, "beta2s"_a, "beta1"_a = 0, "beta1s"_a = 0, "beta3"_a = 0, "beta3s"_a = 0);

  Chi2.def("setPump", (void (Chi2::*)(int, double))&Chi2::setPump,
           "pulseType"_a, "chirp"_a = 0);

  Chi2.def("setPump", (void (Chi2::*)(const Eigen::Ref<const Arraycd>&, double))&Chi2::setPump,
           "customPump"_a, "chirp"_a = 0);

  Chi2.def("runPumpSimulation", &Chi2::runPumpSimulation);

  Chi2.def("runSignalSimulation", &Chi2::runSignalSimulation,
           "inputProf"_a, "timeSignal"_a = true);

  Chi2.def("computeGreensFunction",
           &Chi2::computeGreensFunction, py::return_value_policy::move,
           "inTimeDomain"_a = false, "runPump"_a = true);

  Chi2.def_property_readonly("pumpFreq", &Chi2::getPumpFreq, py::return_value_policy::reference);
  Chi2.def_property_readonly("pumpTime", &Chi2::getPumpTime, py::return_value_policy::reference);
  Chi2.def_property_readonly("signalFreq", &Chi2::getSignalFreq, py::return_value_policy::reference);
  Chi2.def_property_readonly("signalTime", &Chi2::getSignalTime, py::return_value_policy::reference);
  Chi2.def_property_readonly("omega", &Chi2::getFrequency, py::return_value_policy::reference);
  Chi2.def_property_readonly("tau", &Chi2::getTime, py::return_value_policy::reference);

/*
 * Cascade
 */

  Cascade.def(py::init<bool, const std::vector<std::reference_wrapper<_NonlinearMedium>>&>(),
              "sharePump"_a, "inputMedia"_a);

  Cascade.def("runPumpSimulation", &Cascade::runPumpSimulation);

  Cascade.def("runSignalSimulation", &Cascade::runSignalSimulation,
              "inputProf"_a, "timeSignal"_a = true);

  Cascade.def("computeGreensFunction",
              &Cascade::computeGreensFunction, py::return_value_policy::move,
              "inTimeDomain"_a = false, "runPump"_a = true);

  Cascade.def("addMedium", &Cascade::addMedium,
              "medium"_a);

  Cascade.def_property_readonly("omega", &Cascade::getFrequency, py::return_value_policy::reference);
  Cascade.def_property_readonly("tau", &Cascade::getTime, py::return_value_policy::reference);

  Cascade.def("get", &Cascade::get, py::return_value_policy::reference);
}