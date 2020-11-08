#ifndef NONLINEARMEDIUM
#define NONLINEARMEDIUM

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include <utility>


// Eigen default 1D Array is defined with X rows, 1 column, which does not work with row-major order 2D arrays.
// Thus define custom double and complex double 1D arrays. Also define the row-major order 2D double and complex arrays.
// Row vector defined for compatibility with EigenFFT.
typedef Eigen::Array<double, 1, Eigen::Dynamic, Eigen::RowMajor> Arrayd;
typedef Eigen::Array<std::complex<double>, 1, Eigen::Dynamic, Eigen::RowMajor> Arraycd;
typedef Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Array2Dcd;
typedef Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic, Eigen::RowMajor> RowVectorcd;


class _NonlinearMedium {
friend class Cascade;
friend class _NLM2ModeExtension;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  _NonlinearMedium(double relativeLength, double nlLength, double dispLength, double beta2, double beta2s,
                   const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
                   double beta1=0, double beta1s=0, double beta3=0, double beta3s=0, double diffBeta0=0,
                   double chirp=0, double rayleighLength=std::numeric_limits<double>::infinity(),
                   double tMax=10, uint tPrecision=512, uint zPrecision=100);

  virtual void setPump(int pulseType, double chirpLength=0);
  virtual void setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength=0);

  virtual void runPumpSimulation();
  virtual void runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain=true);
  virtual std::pair<Array2Dcd, Array2Dcd> computeGreensFunction(bool inTimeDomain=false, bool runPump=true, uint nThreads=1);
  virtual Array2Dcd batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs, bool inTimeDomain=false, bool runPump=true, uint nThreads=1);

  const Array2Dcd& getPumpFreq()   {return pumpFreq;};
  const Array2Dcd& getPumpTime()   {return pumpTime;};
  const Array2Dcd& getSignalFreq() {return signalFreq;};
  const Array2Dcd& getSignalTime() {return signalTime;};
  const Arrayd& getTime()      {return _tau;};
  const Arrayd& getFrequency() {return _omega;};

protected:
  void setLengths(double relativeLength, double nlLength, double dispLength, uint zPrecision, double rayleighLength);
  void resetGrids(uint nFreqs, double tMax);
  void setDispersion(double beta2, double beta2s, double beta1=0, double beta1s=0,
                     double beta3=0, double beta3s=0, double diffBeta0=0);
  _NonlinearMedium() = default;
  virtual void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain,
                                   Array2Dcd& signalFreq, Array2Dcd& signalTime) = 0;

  inline Arrayd fftshift(const Arrayd& input);
  inline Array2Dcd fftshift2(const Array2Dcd& input);

  double _z;  /// length of medium
  double _DS; /// dispersion length
  double _NL; /// nonlinear length
  bool _noDispersion; /// indicates system is dispersionless
  bool _noNonlinear;  /// indicates system is linear
  double _dz;    /// length increment
  uint _nZSteps; /// number of length steps in simulating the PDE
  uint _nFreqs;  /// number of frequency/time bins in the simulating thte PDE
  double _tMax;  /// positive and negative extent of the simulation window in time
  double _beta2;  /// second order dispersion of the pump's frequency
  double _beta2s; /// second order dispersion of the signal's frequency
  double _beta1;  /// group velocity difference for pump relative to simulation window
  double _beta1s; /// group velocity difference for signal relative to simulation window
  double _beta3;  /// third order dispersion of the pump's frequency
  double _beta3s; /// third order dispersion of the signal's frequency
  double _diffBeta0; /// wave-vector mismatch of the simulated process
  double _rayleighLength; /// Rayleigh length of propagation, assumes focused at medium's center

  Arraycd _dispStepPump; /// incremental phase due to dispersion over length dz for the pump
  Arraycd _dispStepSign; /// incremental phase due to dispersion over length dz for the signal
  std::complex<double> _nlStep; /// strength of nonlinear process over length dz

  Arrayd _tau;   /// array representing the time axis
  Arrayd _omega; /// array representing the frequency axis
  Arrayd _dispersionPump; /// dispersion profile of pump
  Arrayd _dispersionSign; /// dispersion profile of signal
  Arraycd _env; /// initial envelope of the pump

  Array2Dcd pumpFreq;   /// grid for numerically solving PDE, representing pump propagation in frequency domain
  Array2Dcd pumpTime;   /// grid for numerically solving PDE, representing pump propagation in time domain
  Array2Dcd signalFreq; /// grid for numerically solving PDE, representing signal propagation in frequency domain
  Array2Dcd signalTime; /// grid for numerically solving PDE, representing signal propagation in time domain

  Eigen::FFT<double> fftObj; /// fft class object for performing dft
};


class _NLM2ModeExtension {
public:
  std::pair<Array2Dcd, Array2Dcd> computeTotalGreen(bool inTimeDomain=false, bool runPump=true, uint nThreads=1);

  const Array2Dcd& getOriginalFreq() {return originalFreq;};
  const Array2Dcd& getOriginalTime() {return originalTime;};

  _NLM2ModeExtension(_NonlinearMedium& medium, double nlLengthOrig, double beta2o, double beta1o, double beta3o);
  _NLM2ModeExtension(const _NLM2ModeExtension&) = delete;

  void runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain=true);

protected:
  _NonlinearMedium& m; // Store a reference of the actual _NonlinearMedium object, to access variables and methods

  void setLengths(double nlLengthOrig);
  void resetGrids();
  void setDispersion(double beta2o, double beta1o, double beta3o);

  double _beta2o; /// second order dispersion of the original signal's frequency
  double _beta1o; /// group velocity difference for original signal relative to simulation window
  double _beta3o; /// third order dispersion of the original signal's frequency
  // double _NLo; /// like nlLength but with respect to the original signal
  std::complex<double> _nlStepO; /// strength of nonlinear process over length dz; DOPA process of original signal

  Arrayd _dispersionOrig; /// dispersion profile of original signal
  Arraycd _dispStepOrig;  /// incremental phase due to dispersion over length dz for the signal

  Array2Dcd originalFreq; /// grid for numerically solving PDE, representing original signal propagation in frequency domain
  Array2Dcd originalTime; /// grid for numerically solving PDE, representing original signal propagation in time domain
};


class Chi3 : public _NonlinearMedium {
public:
  Chi3(double relativeLength, double nlLength, double dispLength, double beta2,
       const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
       double beta3=0, double chirp=0, double rayleighLength=std::numeric_limits<double>::infinity(),
       double tMax=10, uint tPrecision=512, uint zPrecision=100);

  void runPumpSimulation() override;
  using _NonlinearMedium::runSignalSimulation;

protected:
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain,
                           Array2Dcd& signalFreq, Array2Dcd& signalTime) override;
};


class _Chi2 : public _NonlinearMedium {
public:
  _Chi2(double relativeLength, double nlLength, double dispLength, double beta2, double beta2s,
        const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
        double beta1=0, double beta1s=0, double beta3=0, double beta3s=0, double diffBeta0=0,
        double chirp=0, double rayleighLength=std::numeric_limits<double>::infinity(),
        double tMax=10, uint tPrecision=512, uint zPrecision=100,
        const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));

  const Arrayd& getPoling() {return _poling;};

protected:
  void setPoling(const Eigen::Ref<const Arrayd>& poling);
  _Chi2() = default;

  Arrayd _poling; /// array representing the poling direction at a given point on the grid.
};


class Chi2PDC : public _Chi2 {
public:
  using _Chi2::_Chi2;
  using _NonlinearMedium::runSignalSimulation;
protected:
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain,
                           Array2Dcd& signalFreq, Array2Dcd& signalTime) override;
};


class Chi2SHG : public _Chi2 {
public:
  using _NonlinearMedium::runSignalSimulation;
#ifdef DEPLETESHG
  Chi2SHG(double relativeLength, double nlLength, double nlLengthP, double dispLength, double beta2, double beta2s,
          const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
          double beta1=0, double beta1s=0, double beta3=0, double beta3s=0, double diffBeta0=0,
          double chirp=0, double rayleighLength=std::numeric_limits<double>::infinity(),
          double tMax=10, uint tPrecision=512, uint zPrecision=100,
          const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
protected:
  std::complex<double> _nlstepP;
#else
  using _Chi2::_Chi2;
protected:
#endif
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain,
                           Array2Dcd& signalFreq, Array2Dcd& signalTime) override;
};


class Chi2SFG : public _Chi2, public _NLM2ModeExtension {
public:
  Chi2SFG(double relativeLength, double nlLength, double nlLengthOrig, double dispLength,
          double beta2, double beta2s, double beta2o,
          const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
          double beta1=0, double beta1s=0, double beta1o=0, double beta3=0, double beta3s=0, double beta3o=0,
          double diffBeta0=0, double diffBeta0o=0, double chirp=0, double rayleighLength=std::numeric_limits<double>::infinity(),
          double tMax=10, uint tPrecision=512, uint zPrecision=100,
          const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));

  using _NLM2ModeExtension::runSignalSimulation;

protected:
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain,
                           Array2Dcd& signalFreq, Array2Dcd& signalTime) override;

  double _diffBeta0o; /// wave-vector mismatch of PDC process with the original signal and pump
};


class Chi2PDCII : public _Chi2, public _NLM2ModeExtension {
public:
  Chi2PDCII(double relativeLength, double nlLength, double nlLengthOrig, double nlLengthI, double dispLength,
            double beta2, double beta2s, double beta2o,
            const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
            double beta1=0, double beta1s=0, double beta1o=0, double beta3=0, double beta3s=0, double beta3o=0,
            double diffBeta0=0, double diffBeta0o=0, double chirp=0, double rayleighLength=std::numeric_limits<double>::infinity(),
            double tMax=10, uint tPrecision=512, uint zPrecision=100,
            const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));

  using _NLM2ModeExtension::runSignalSimulation;

protected:
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain,
                           Array2Dcd& signalFreq, Array2Dcd& signalTime) override;

  std::complex<double> _nlStepI; /// strength of type I nonlinear process over length dz; DOPA process of original signal
  double _diffBeta0o; /// wave-vector mismatch of PDC process with the original signal and pump
};


class Cascade : public _NonlinearMedium {
public:
  Cascade(bool sharePump, const std::vector<std::reference_wrapper<_NonlinearMedium>>& inputMedia);
  void addMedium(_NonlinearMedium& medium);
  void setPump(int pulseType, double chirpLength=0) override;
  void setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength=0) override;
  void runPumpSimulation() override;
  void runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain=true) override;
  std::pair<Array2Dcd, Array2Dcd> computeGreensFunction(bool inTimeDomain=false, bool runPump=true, uint nThreads=1) override;
  Array2Dcd batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs, bool inTimeDomain=false, bool runPump=true, uint nThreads=1) override;

  _NonlinearMedium& getMedium(uint i) {return media.at(i).get();}
  const std::vector<std::reference_wrapper<_NonlinearMedium>>& getMedia() {return media;}
  uint getNMedia() {return media.size();}

  const Arrayd& getTime()      {return media.at(0).get().getTime();};
  const Arrayd& getFrequency() {return media.at(0).get().getFrequency();};

private: // Disable functions (note: still accessible from base class)
  using _NonlinearMedium::setLengths;
  using _NonlinearMedium::resetGrids;
  using _NonlinearMedium::setDispersion;
  using _NonlinearMedium::getPumpFreq;
  using _NonlinearMedium::getPumpTime;
  using _NonlinearMedium::getSignalFreq;
  using _NonlinearMedium::getSignalTime;
  void runSignalSimulation(const Arraycd&, bool, Array2Dcd&, Array2Dcd&) override {};

protected:
  std::vector<std::reference_wrapper<_NonlinearMedium>> media; /// collection of nonlinear media objects
  bool sharedPump; /// is the pump shared across media or are they independently pumped
};


#endif //NONLINEARMEDIUM

