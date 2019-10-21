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
  _NonlinearMedium(double relativeLength, double nlLength, double dispLength, double beta2, double beta2s,
                   const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
                   double beta1=0, double beta1s=0, double beta3=0, double beta3s=0, double diffBeta0=0,
                   double chirp=0, double tMax=10, uint tPrecision=512, uint zPrecision=100);

  void setLengths(double relativeLength, double nlLength, double dispLength, uint zPrecision=100);
  virtual void resetGrids(uint nFreqs=0, double tMax=0);
  void setDispersion(double beta2, double beta2s, double beta1=0, double beta1s=0,
                     double beta3=0, double beta3s=0, double diffBeta0=0);
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
  double _diffBeta0;

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
  Chi3(double relativeLength, double nlLength, double dispLength, double beta2,
       const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
       double beta3=0, double chirp=0, double tMax=10, uint tPrecision=512, uint zPrecision=100);

  void runPumpSimulation() override;
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain=true) override;
};


class _Chi2 : public _NonlinearMedium {
public:
  using _NonlinearMedium::_NonlinearMedium;
  void runPumpSimulation() override;
};


class Chi2PDC : public _Chi2 {
public:
  using _Chi2::_Chi2;
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain=true) override;
};


class Chi2SFG : public _Chi2 {
public:
  Chi2SFG(double relativeLength, double nlLength, double nlLengthOrig, double dispLength,
          double beta2, double beta2s, double beta2o,
          const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
          double beta1=0, double beta1s=0, double beta1o=0, double beta3=0, double beta3s=0, double beta3o=0,
          double diffBeta0=0, double diffBeta0o=0,
          double chirp=0, double tMax=10, uint tPrecision=512, uint zPrecision=100);

  void setLengths(double relativeLength, double nlLength, double nlLengthOrig, double dispLength, uint zPrecision=100);
  void resetGrids(uint nFreqs=0, double tMax=0) override;
  void setDispersion(double beta2, double beta2s, double beta2o, double beta1=0, double beta1s=0, double beta1o=0,
                     double beta3=0, double beta3s=0, double beta3o=0, double diffBeta0=0, double diffBeta0o=0);

  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain=true) override;

  const Array2Dcd& getOriginalFreq() {return originalFreq;};
  const Array2Dcd& getOriginalTime() {return originalTime;};

private:
  using _NonlinearMedium::setLengths;
  using _NonlinearMedium::resetGrids;
  using _NonlinearMedium::setDispersion;

protected:
  double _beta2o;
  double _beta1o;
  double _beta3o;
  double _diffBeta0o;
  double _NLo;
  std::complex<double> _nlStepO;

  Arrayf _dispersionOrig;
  Arraycd _dispStepOrig;

  Array2Dcd originalFreq;
  Array2Dcd originalTime;
};


class Cascade : public _NonlinearMedium {
public:
  Cascade(bool sharePump, const std::vector<std::reference_wrapper<_NonlinearMedium>>& inputMedia);
  void addMedium(_NonlinearMedium& medium);
  void runPumpSimulation() override;
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain=true) override;
  std::pair<Array2Dcd, Array2Dcd> computeGreensFunction(bool inTimeDomain=false, bool runPump=true);

  _NonlinearMedium& getMedium(uint i) {return media.at(i).get();}
  const std::vector<std::reference_wrapper<_NonlinearMedium>>& getMedia() {return media;}
  uint getNMedia() {return nMedia;}

  const Arrayf& getTime()      {return media.at(0).get()._tau;};
  const Arrayf& getFrequency() {return media.at(0).get()._omega;};

private:
  // Disable functions (note: still accessible from base class)
  using _NonlinearMedium::setLengths;
  using _NonlinearMedium::resetGrids;
  using _NonlinearMedium::setDispersion;
  using _NonlinearMedium::setPump;
  using _NonlinearMedium::setPump;
  using _NonlinearMedium::getPumpFreq;
  using _NonlinearMedium::getPumpTime;
  using _NonlinearMedium::getSignalFreq;
  using _NonlinearMedium::getSignalTime;

protected:
  std::vector<std::reference_wrapper<_NonlinearMedium>> media;
  uint nMedia;
  bool sharedPump;
};


#endif //NONLINEARMEDIUM

