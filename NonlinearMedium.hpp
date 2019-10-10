#ifndef NONLINEARMEDIUM
#define NONLINEARMEDIUM

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>


// Eigen 1D arrays are defined with X rows, 1 column, which is annoying when operating on 2D arrays.
// Also must specify row-major order for 2D
typedef Eigen::Array<double, 1, Eigen::Dynamic, Eigen::RowMajor> Arrayf;
typedef Eigen::Array<std::complex<double>, 1, Eigen::Dynamic, Eigen::RowMajor> Arraycd;
typedef Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Array2Dcd;
typedef Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorcd;
// TODO consider making everything row major matrix for FFT compatibility


class _NonlinearMedium {
friend class Cascade;
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
  virtual void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain=true) = 0;
  virtual std::pair<Array2Dcd, Array2Dcd> computeGreensFunction(bool inTimeDomain=false, bool runPump=true);

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

  _NonlinearMedium() = default;
};


class Chi3 : public _NonlinearMedium {
public:
  using _NonlinearMedium::_NonlinearMedium;
  void runPumpSimulation() override;
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain=true) override;
};


class Chi2 : public _NonlinearMedium {
public:
  using _NonlinearMedium::_NonlinearMedium;
  void runPumpSimulation() override;
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain=true) override;
};


class Cascade : public _NonlinearMedium {
public:
  Cascade(bool sharePump, std::vector<std::reference_wrapper<_NonlinearMedium>>& inputMedia);
  void addMedium(_NonlinearMedium& medium);
  void runPumpSimulation() override;
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain=true) override;
  std::pair<Array2Dcd, Array2Dcd> computeGreensFunction(bool inTimeDomain=false, bool runPump=true);

  _NonlinearMedium& get(uint i) {return media.at(i).get();}

private:
  // Disable functions
  using _NonlinearMedium::setLengths;
  using _NonlinearMedium::resetGrids;
  using _NonlinearMedium::setDispersion;
  using _NonlinearMedium::setPump;
  using _NonlinearMedium::setPump;
  using _NonlinearMedium::getPumpFreq;
  using _NonlinearMedium::getPumpTime;
  using _NonlinearMedium::getSignalFreq;
  using _NonlinearMedium::getSignalTime;

  std::vector<std::reference_wrapper<_NonlinearMedium>> media;
  bool sharedPump;
};




#endif //NONLINEARMEDIUM

