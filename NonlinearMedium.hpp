#ifndef NONLINEARMEDIUM
#define NONLINEARMEDIUM

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


// Eigen 1D arrays are defined with X rows, 1 column, which is annoying when operating on 2D arrays.
// Also must specify row-major order for 2D
typedef Eigen::Array<double, 1, Eigen::Dynamic, Eigen::RowMajor> Arrayf;
typedef Eigen::Array<std::complex<double>, 1, Eigen::Dynamic, Eigen::RowMajor> Arraycd;
typedef Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Array2Dcd;
typedef Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorcd;
// TODO consider making everything row major matrix for FFT compatibility

class _NonlinearMedium {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  _NonlinearMedium(double relativeLength, double nlLength, double dispLength,
                   double beta2, double beta2s, int pulseType=0,
                   double beta1=0, double beta1s=0, double beta3=0, double beta3s=0,
                   double chirp=0, double tMax=10, uint tPrecision=512, uint zPrecision=100);

  _NonlinearMedium(double relativeLength, double nlLength, double dispLength,
                   double beta2, double beta2s, const Eigen::Ref<const Arraycd>& customPump, int pulseType=0,
                   double beta1=0, double beta1s=0, double beta3=0, double beta3s=0,
                   double chirp=0, double tMax=10, uint tPrecision=512, uint zPrecision=100);

  void setLengths(double relativeLength, double nlLength, double dispLength, uint zPrecision=100);
  void resetGrids(uint nFreqs=0, double tMax=0);
  void setDispersion(double beta2, double beta2s, double beta1=0, double beta1s=0, double beta3=0, double beta3s=0);
  void setPump(int pulseType, double chirp=0);
  void setPump(const Eigen::Ref<const Arraycd>& customPump, double chirp=0);

  virtual void runPumpSimulation() = 0;
  virtual void runSignalSimulation(const Arraycd& inputProf, bool timeSignal=true) = 0;
  std::pair<Array2Dcd, Array2Dcd> computeGreensFunction(bool inTimeDomain=false, bool runPump=true);

  const Array2Dcd& getPumpFreq()   {return pumpFreq;};
  const Array2Dcd& getPumpTime()   {return pumpTime;};
  const Array2Dcd& getSignalFreq() {return signalFreq;};
  const Array2Dcd& getSignalTime() {return signalTime;};
  const Arrayf& getTime()      {return _tau;};
  const Arrayf& getFrequency() {return _omega;};

protected:
  double _z;
  double _DS;
  double _NL;
  bool _noDispersion;
  bool _noNonlinear;
  double _Nsquared;
  double _dz;
  uint _nZSteps;
  uint _nFreqs;
  double _tMax;
  double _beta2;
  double _beta2s;
  double _beta1;
  double _beta1s;
  double _beta3;
  double _beta3s;

  Arraycd _dispStepPump;
  Arraycd _dispStepSign;
  std::complex<double> _nlStep;

  Arrayf _tau;
  Arrayf _omega;
  Arrayf _dispersionPump;
  Arrayf _dispersionSign;
  Arraycd _env;

  Array2Dcd pumpFreq;
  Array2Dcd pumpTime;
  Array2Dcd signalFreq;
  Array2Dcd signalTime;

  RowVectorcd fftTemp;
  Eigen::FFT<double> fftObj;

  inline const RowVectorcd& fft(const RowVectorcd& input);
  inline const RowVectorcd& ifft(const RowVectorcd& input);
  inline Arrayf fftshift(const Arrayf& input);
  inline Array2Dcd fftshift(const Array2Dcd& input);
};

class Chi3 : public _NonlinearMedium {
public:
  using _NonlinearMedium::_NonlinearMedium;
  void runPumpSimulation() override;
  void runSignalSimulation(const Arraycd& inputProf, bool timeSignal=true) override;
};

class Chi2 : public _NonlinearMedium {
public:
  using _NonlinearMedium::_NonlinearMedium;
  void runPumpSimulation() override;
  void runSignalSimulation(const Arraycd& inputProf, bool timeSignal=true) override;
};



// Pybind11 Python binding
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(nonlinearmedium, m) {
  py::class_<Chi3> Chi3(m, "Chi3");
  Chi3
    .def(py::init<double, double, double, double, double, int, double, double, double, double, double, double, uint, uint>(),
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "pulseType"_a=0,
         "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0,
         "chirp"_a=0, "tMax"_a=10, "tPrecision"_a=512, "zPrecision"_a=100)

    .def(py::init<double, double, double, double, double, Eigen::Ref<const Arraycd>&, int, double, double, double, double, double, double, uint, uint>(),
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "customPump"_a, "pulseType"_a=0,
         "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0,
         "chirp"_a=0, "tMax"_a=10, "tPrecision"_a=512, "zPrecision"_a=100)

    .def("setLengths", &Chi3::setLengths,
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "zPrecision"_a=100)

    .def("resetGrids", &Chi3::resetGrids,
         "nFreqs"_a=0, "tMax"_a=0)

    .def("setDispersion", &Chi3::setDispersion,
         "beta2"_a, "beta2s"_a, "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0)

    .def("setPump", (void (Chi3::*)(int, double)) &Chi3::setPump,
         "pulseType"_a, "chirp"_a=0)

    .def("setPump", (void (Chi3::*)(const Eigen::Ref<const Arraycd>&, double)) &Chi3::setPump,
         "customPump"_a, "chirp"_a=0)

    .def("runPumpSimulation", &Chi3::runPumpSimulation)

    .def("runSignalSimulation", &Chi3::runSignalSimulation,
         "inputProf"_a, "timeSignal"_a=true)

    .def("computeGreensFunction", &Chi3::computeGreensFunction, py::return_value_policy::move,
         "inTimeDomain"_a=false, "runPump"_a=true)

    .def_property_readonly("pumpFreq",   &Chi3::getPumpFreq,   py::return_value_policy::reference)
    .def_property_readonly("pumpTime",   &Chi3::getPumpTime,   py::return_value_policy::reference)
    .def_property_readonly("signalFreq", &Chi3::getSignalFreq, py::return_value_policy::reference)
    .def_property_readonly("signalTime", &Chi3::getSignalTime, py::return_value_policy::reference)
    .def_property_readonly("omega",      &Chi3::getFrequency,  py::return_value_policy::reference)
    .def_property_readonly("tau",        &Chi3::getTime,       py::return_value_policy::reference);

  py::class_<Chi2> Chi2(m, "Chi2");
  Chi2
    .def(py::init<double, double, double, double, double, int, double, double, double, double, double, double, uint, uint>(),
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "pulseType"_a,
         "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0,
         "chirp"_a=0, "tMax"_a=10, "tPrecision"_a=512, "zPrecision"_a=100)

    .def(py::init<double, double, double, double, double, Eigen::Ref<const Arraycd>&,
                  int, double, double, double, double, double, double, uint, uint>(),
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "customPump"_a, "pulseType"_a=0,
         "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0,
         "chirp"_a=0, "tMax"_a=10, "tPrecision"_a=512, "zPrecision"_a=100)

    .def("setLengths", &Chi3::setLengths,
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "zPrecision"_a=100)

    .def("resetGrids", &Chi3::resetGrids,
         "nFreqs"_a=0, "tMax"_a=0)

    .def("setDispersion", &Chi3::setDispersion,
         "beta2"_a, "beta2s"_a, "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0)

    .def("setPump", (void (Chi2::*)(int, double)) &Chi2::setPump,
         "pulseType"_a, "chirp"_a=0)

    .def("setPump", (void (Chi2::*)(const Eigen::Ref<const Arraycd>&, double)) &Chi2::setPump,
         "customPump"_a, "chirp"_a=0)

    .def("runPumpSimulation", &Chi2::runPumpSimulation)

    .def("runSignalSimulation", &Chi2::runSignalSimulation,
         "inputProf"_a, "timeSignal"_a=true)

    .def("computeGreensFunction",
         &Chi2::computeGreensFunction, py::return_value_policy::move,
         "inTimeDomain"_a=false, "runPump"_a=true)

    .def_property_readonly("pumpFreq",   &Chi2::getPumpFreq,   py::return_value_policy::reference)
    .def_property_readonly("pumpTime",   &Chi2::getPumpTime,   py::return_value_policy::reference)
    .def_property_readonly("signalFreq", &Chi2::getSignalFreq, py::return_value_policy::reference)
    .def_property_readonly("signalTime", &Chi2::getSignalTime, py::return_value_policy::reference)
    .def_property_readonly("omega",      &Chi2::getFrequency,  py::return_value_policy::reference)
    .def_property_readonly("tau",        &Chi2::getTime,       py::return_value_policy::reference);
}


#endif //NONLINEARMEDIUM

