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
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual void setPump(int pulseType, double chirpLength=0, double delayLength=0);
  virtual void setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength=0, double delayLength=0);

  virtual void runPumpSimulation();
  virtual void runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain=true, uint inputMode=0);
  virtual std::pair<Array2Dcd, Array2Dcd> computeGreensFunction(bool inTimeDomain=false, bool runPump=true, uint nThreads=1, bool normalize=false,
                                                                const std::vector<char>& useInput={}, const std::vector<char>& useOutput={});
  virtual Array2Dcd batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs, bool inTimeDomain=false, bool runPump=true, uint nThreads=1,
                                          uint inputMode=0, const std::vector<char>& useOutput={});

  const Array2Dcd& getPumpFreq() {return pumpFreq;};
  const Array2Dcd& getPumpTime() {return pumpTime;};
  const Array2Dcd& getSignalFreq(uint i=0) {return signalFreq.at(i);};
  const Array2Dcd& getSignalTime(uint i=0) {return signalTime.at(i);};
  const Arrayd& getTime()      {return _tau;};
  const Arrayd& getFrequency() {return _omega;};

protected:
  _NonlinearMedium(uint nSignalmodes, double relativeLength, std::initializer_list<double> nlLength,
                   double beta2, std::initializer_list<double> beta2s, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                   double beta1, std::initializer_list<double> beta1s, double beta3, std::initializer_list<double> beta3s,
                   std::initializer_list<double> diffBeta0, double rayleighLength, double tMax, uint tPrecision, uint zPrecision,
                   double chirp, double delay);

  inline void setLengths(double relativeLength, const std::vector<double>& nlLength, uint zPrecision, double rayleighLength,
                         double beta2, const std::vector<double>& beta2s, double beta1, const std::vector<double>& beta1s,
                         double beta3, const std::vector<double>& beta3s);
  inline void resetGrids(uint nFreqs, double tMax);
  inline void setDispersion(double beta2, const std::vector<double>& beta2s, double beta1, const std::vector<double>& beta1s,
                            double beta3, const std::vector<double>& beta3s, std::initializer_list<double> diffBeta0);
  _NonlinearMedium() : _nSignalModes() {};
  virtual void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                   std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime) = 0;

  template<class T>
  void signalSimulationTemplate(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime);

  static inline Arrayd fftshift(const Arrayd& input);
  static inline Array2Dcd fftshift2(const Array2Dcd& input);

  const uint _nSignalModes; /// Number of separate signal modes (eg polarizations, frequencies, etc)
  double _z;  /// length of medium
  double _dz;    /// length increment
  uint _nZSteps; /// number of length steps in simulating the PDE
  uint _nFreqs;  /// number of frequency/time bins in the simulating thte PDE
  double _tMax;  /// positive and negative extent of the simulation window in time
  double _beta2;  /// second order dispersion of the pump
  double _beta1;  /// relative group velocity of the pump

  double _rayleighLength; /// Rayleigh length of propagation, assumes focused at medium's center
  Arraycd _dispStepPump; /// incremental phase due to dispersion over length dz for the pump

  std::vector<double> _diffBeta0; /// wave-vector mismatch of the simulated process
  std::vector<std::complex<double>> _nlStep; /// strength of nonlinear process over length dz
  std::vector<Arraycd> _dispStepSign; /// incremental phase due to dispersion over length dz for the signal

  Arrayd _tau;   /// array representing the time axis
  Arrayd _omega; /// array representing the frequency axis
  Arrayd _dispersionPump; /// dispersion profile of pump
  std::vector<Arrayd> _dispersionSign; /// dispersion profile of signal
  Arraycd _env; /// initial envelope of the pump

  Array2Dcd pumpFreq;   /// grid for numerically solving PDE, representing pump propagation in frequency domain
  Array2Dcd pumpTime;   /// grid for numerically solving PDE, representing pump propagation in time domain
  std::vector<Array2Dcd> signalFreq; /// grid for numerically solving PDE, representing signal propagation in frequency domain
  std::vector<Array2Dcd> signalTime; /// grid for numerically solving PDE, representing signal propagation in time domain

  Eigen::FFT<double> fftObj; /// fft class object for performing dft
};

// Repeated code for each NLM ODE class. This takes care of:
// - Allowing _NonlinearMedium friend access to the protected DiffEq function, to use in signalSimulationTemplate
// - Overriding runSignalSimulation with the function created from the template
#define NLM(T, modes) \
protected: \
  friend _NonlinearMedium; \
  constexpr static uint _nSignalModes = modes; \
  inline void DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, std::vector<Arraycd>& k4, \
                     const Arraycd& prevP, const Arraycd& currP, const Arraycd& interpP, const std::vector<Array2Dcd>& signal); \
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain, uint inputMode, \
                           std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime) override \
     { signalSimulationTemplate<T>(inputProf, inTimeDomain, inputMode, signalFreq, signalTime); };


class Chi3 : public _NonlinearMedium {
  NLM(Chi3, 1)
public:
  Chi3(double relativeLength, double nlLength, double beta2,
       const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
       double beta3=0, double rayleighLength=std::numeric_limits<double>::infinity(),
       double tMax=10, uint tPrecision=512, uint zPrecision=100, double chirp=0);

  void runPumpSimulation() override;
};


class _Chi2 : public _NonlinearMedium {
public:
  _Chi2(uint nSignalmodes, double relativeLength, std::initializer_list<double> nlLength,
        double beta2, std::initializer_list<double> beta2s, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
        double beta1, std::initializer_list<double> beta1s, double beta3, std::initializer_list<double> beta3s, std::initializer_list<double> diffBeta0,
        double rayleighLength, double tMax, uint tPrecision, uint zPrecision, double chirp, double delay, const Eigen::Ref<const Arrayd>& poling);

  const Arrayd& getPoling() {return _poling;};

protected:
  void setPoling(const Eigen::Ref<const Arrayd>& poling);
  _Chi2() = default;

  Arrayd _poling; /// array representing the poling direction at a given point on the grid.
};


class Chi2PDC : public _Chi2 {
  NLM(Chi2PDC, 1)
public:
  Chi2PDC(double relativeLength, double nlLength, double beta2, double beta2s,
          const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
          double beta1=0, double beta1s=0, double beta3=0, double beta3s=0, double diffBeta0=0,
          double rayleighLength=std::numeric_limits<double>::infinity(),
          double tMax=10, uint tPrecision=512, uint zPrecision=100, double chirp=0, double delay=0,
          const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


class Chi2SHG : public _Chi2 {
public:
  using _NonlinearMedium::runSignalSimulation;
#ifdef DEPLETESHG
  Chi2SHG(double relativeLength, double nlLength, double nlLengthP, double beta2, double beta2s,
#else
  Chi2SHG(double relativeLength, double nlLength, double beta2, double beta2s,
#endif
          const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
          double beta1=0, double beta1s=0, double beta3=0, double beta3s=0, double diffBeta0=0,
          double rayleighLength=std::numeric_limits<double>::infinity(), double tMax=10, uint tPrecision=512, uint zPrecision=100,
          double chirp=0, double delay=0, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));

protected:
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                           std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime) override;
};


class Chi2SFGPDC : public _Chi2 {
  NLM(Chi2SFGPDC, 2)
public:
  Chi2SFGPDC(double relativeLength, double nlLength, double nlLengthOrig, double beta2, double beta2s, double beta2o,
             const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
             double beta1=0, double beta1s=0, double beta1o=0, double beta3=0, double beta3s=0, double beta3o=0,
             double diffBeta0=0, double diffBeta0o=0, double rayleighLength=std::numeric_limits<double>::infinity(),
             double tMax=10, uint tPrecision=512, uint zPrecision=100, double chirp=0, double delay=0,
             const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


class Chi2SFG : public _Chi2 {
  NLM(Chi2SFG, 2)
public:
  Chi2SFG(double relativeLength, double nlLength, double nlLengthOrig, double beta2, double beta2s, double beta2o,
          const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
          double beta1=0, double beta1s=0, double beta1o=0, double beta3=0, double beta3s=0, double beta3o=0,
          double diffBeta0=0, double rayleighLength=std::numeric_limits<double>::infinity(),
          double tMax=10, uint tPrecision=512, uint zPrecision=100, double chirp=0, double delay=0,
          const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


class Chi2PDCII : public _Chi2 {
  NLM(Chi2PDCII, 2)
public:
  Chi2PDCII(double relativeLength, double nlLength, double nlLengthOrig, double nlLengthI,
            double beta2, double beta2s, double beta2o,
            const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
            double beta1=0, double beta1s=0, double beta1o=0, double beta3=0, double beta3s=0, double beta3o=0,
            double diffBeta0=0, double diffBeta0o=0, double rayleighLength=std::numeric_limits<double>::infinity(),
            double tMax=10, uint tPrecision=512, uint zPrecision=100, double chirp=0, double delay=0,
            const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


class Chi2SFGII : public _Chi2 {
  NLM(Chi2SFGII, 4)
public:
  Chi2SFGII(double relativeLength, double nlLengthSignZ, double nlLengthSignY, double nlLengthOrigZ, double nlLengthOrigY,
            double beta2, double beta2sz, double beta2sy, double beta2oz, double beta2oy,
            const Eigen::Ref<const Arraycd>& customPump=Eigen::Ref<const Arraycd>(Arraycd{}), int pulseType=0,
            double beta1=0, double beta1sz=0, double beta1sy=0, double beta1oz=0, double beta1oy=0, double beta3=0,
            double beta3sz=0, double beta3sy=0, double beta3oz=0, double beta3oy=0,
            double diffBeta0z=0, double diffBeta0y=0, double diffBeta0s=0, double rayleighLength=std::numeric_limits<double>::infinity(),
            double tMax=10, uint tPrecision=512, uint zPrecision=100, double chirp=0, double delay=0,
            const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));
};


class Cascade : public _NonlinearMedium {
public:
  Cascade(bool sharePump, const std::vector<std::reference_wrapper<_NonlinearMedium>>& inputMedia,
          const std::vector<std::map<uint, uint>>& connections);
  void addMedium(_NonlinearMedium& medium, const std::map<uint, uint>& connection);
  void setPump(int pulseType, double chirpLength=0, double delayLength=0) override;
  void setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength=0, double delayLength=0) override;
  void runPumpSimulation() override;
  void runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain=true, uint inputMode=0) override;
  std::pair<Array2Dcd, Array2Dcd> computeGreensFunction(bool inTimeDomain=false, bool runPump=true, uint nThreads=1, bool normalize=false,
                                                        const std::vector<char>& useInput={}, const std::vector<char>& useOutput={}) override;
  Array2Dcd batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs, bool inTimeDomain=false, bool runPump=true,
                                  uint nThreads=1, uint inputMode=0, const std::vector<char>& useOutput={}) override;

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
  void runSignalSimulation(const Arraycd&, bool, uint, std::vector<Array2Dcd>&, std::vector<Array2Dcd>&) override {};

protected:
  std::vector<std::reference_wrapper<_NonlinearMedium>> media; /// collection of nonlinear media objects
  std::vector<std::map<uint, uint>> connections;
  bool sharedPump; /// is the pump shared across media or are they independently pumped
};


#endif //NONLINEARMEDIUM
