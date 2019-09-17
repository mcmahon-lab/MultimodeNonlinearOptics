#ifndef NONLINEARMEDIUM
#define NONLINEARMEDIUM

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


// Eigen 1D arrays are defined with X rows, 1 column, which is annoying when operating on 2D arrays.
// Also must specify row-major order for 2D
typedef Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor> Arrayf;
typedef Eigen::Array<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> Arraycf;
typedef Eigen::Array<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Array2Dcf;


class _NonlinearMedium {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  _NonlinearMedium(float relativeLength, float nlLength, float dispLength,
                   float beta2, float beta2s, int pulseType=0,
                   float beta1=0, float beta1s=0, float beta3=0, float beta3s=0,
                   float chirp=0, float tMax=10, uint tPrecision=512, uint zPrecision=100);

  _NonlinearMedium(float relativeLength, float nlLength, float dispLength,
                   float beta2, float beta2s, const Eigen::Ref<const Arraycf>& customPump, int pulseType=0,
                   float beta1=0, float beta1s=0, float beta3=0, float beta3s=0,
                   float chirp=0, float tMax=10, uint tPrecision=512, uint zPrecision=100);

  void setLengths(float relativeLength, float nlLength, float dispLength, uint zPrecision=100);
  void resetGrids(uint nFreqs=0, float tMax=0);
  void setDispersion(float beta2, float beta2s, float beta1=0, float beta1s=0, float beta3=0, float beta3s=0);
  void setPump(int pulseType, float chirp=0);
  void setPump(const Eigen::Ref<const Arraycf>& customPump, float chirp=0);

  virtual void runPumpSimulation() = 0;
  virtual void runSignalSimulation(const Arraycf& inputProf, bool timeSignal=true) = 0;
  std::pair<Array2Dcf, Array2Dcf> computeGreensFunction(bool inTimeDomain=false, bool runPump=true);

  const Array2Dcf& getPumpFreq()   {return pumpGridFreq;};
  const Array2Dcf& getPumpTime()   {return pumpGridTime;};
  const Array2Dcf& getSignalFreq() {return signalGridFreq;};
  const Array2Dcf& getSignalTime() {return signalGridTime;};
  const Arrayf& getTime()      {return _tau;};
  const Arrayf& getFrequency() {return _omega;};

protected:
  float _z;
  float _DS;
  float _NL;
  bool _noDispersion;
  bool _noNonlinear;
  float _Nsquared;
  float _dz;
  uint _nZSteps;
  uint _nFreqs;
  float _tMax;
  float _beta2;
  float _beta2s;
  float _beta1;
  float _beta1s;
  float _beta3;
  float _beta3s;

  Arraycf _dispStepPump;
  Arraycf _dispStepSign;
  std::complex<float> _nlStep;

  Arrayf _tau;
  Arrayf _omega;
  Arrayf _dispersionPump;
  Arrayf _dispersionSign;
  Arraycf _env;

  Array2Dcf pumpGridFreq;
  Array2Dcf pumpGridTime;
  Array2Dcf signalGridFreq;
  Array2Dcf signalGridTime;

  Eigen::FFT<float> fftObj;

  Arraycf fft(const Eigen::VectorXcf& input);
  Arraycf ifft(const Eigen::VectorXcf& input);
//  void fft(const Eigen::VectorXcf& input, Eigen::VectorXcf& ouput);
//  void ifft(const Eigen::VectorXcf& input, Eigen::VectorXcf& ouput);
  Arrayf fftshift(const Arrayf& input);
  Array2Dcf fftshift(const Array2Dcf& input);
};

class Chi3 : public _NonlinearMedium {
public:
  using _NonlinearMedium::_NonlinearMedium;
  void runPumpSimulation() override;
  void runSignalSimulation(const Arraycf& inputProf, bool timeSignal=true) override;
};

class Chi2 : public _NonlinearMedium {
public:
  using _NonlinearMedium::_NonlinearMedium;
  void runPumpSimulation() override;
  void runSignalSimulation(const Arraycf& inputProf, bool timeSignal=true) override;
};



// Pybind11 Python binding
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(nonlinearmedium, m) {
  py::class_<Chi3> Chi3(m, "Chi3");
  Chi3
    .def(py::init<float, float, float, float, float, int, float, float, float, float, float, float, uint, uint>(),
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "pulseType"_a=0,
         "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0,
         "chirp"_a=0, "tMax"_a=10, "tPrecision"_a=512, "zPrecision"_a=100)

    .def(py::init<float, float, float, float, float, Eigen::Ref<const Arraycf>&, int, float, float, float, float, float, float, uint, uint>(),
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "customPump"_a, "pulseType"_a=0,
         "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0,
         "chirp"_a=0, "tMax"_a=10, "tPrecision"_a=512, "zPrecision"_a=100)

    .def("setLengths", &Chi3::setLengths,
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "zPrecision"_a=100)

    .def("resetGrids", &Chi3::resetGrids,
         "nFreqs"_a=0, "tMax"_a=0)

    .def("setDispersion", &Chi3::setDispersion,
         "beta2"_a, "beta2s"_a, "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0)

    .def("setPump", (void (Chi3::*)(int, float)) &Chi3::setPump,
         "pulseType"_a, "chirp"_a=0)

    .def("setPump", (void (Chi3::*)(const Eigen::Ref<const Arraycf>&, float)) &Chi3::setPump,
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
    .def(py::init<float, float, float, float, float, int, float, float, float, float, float, float, uint, uint>(),
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "pulseType"_a,
         "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0,
         "chirp"_a=0, "tMax"_a=10, "tPrecision"_a=512, "zPrecision"_a=100)

    .def(py::init<float, float, float, float, float, Eigen::Ref<const Arraycf>&,
                  int, float, float, float, float, float, float, uint, uint>(),
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "beta2"_a, "beta2s"_a, "customPump"_a, "pulseType"_a=0,
         "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0,
         "chirp"_a=0, "tMax"_a=10, "tPrecision"_a=512, "zPrecision"_a=100)

    .def("setLengths", &Chi3::setLengths,
         "relativeLength"_a, "nlLength"_a, "dispLength"_a, "zPrecision"_a=100)

    .def("resetGrids", &Chi3::resetGrids,
         "nFreqs"_a=0, "tMax"_a=0)

    .def("setDispersion", &Chi3::setDispersion,
         "beta2"_a, "beta2s"_a, "beta1"_a=0, "beta1s"_a=0, "beta3"_a=0, "beta3s"_a=0)

    .def("setPump", (void (Chi2::*)(int, float)) &Chi2::setPump,
         "pulseType"_a, "chirp"_a=0)

    .def("setPump", (void (Chi2::*)(const Eigen::Ref<const Arraycf>&, float)) &Chi2::setPump,
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

