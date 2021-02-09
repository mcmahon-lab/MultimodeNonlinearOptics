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

inline constexpr std::complex<double> operator"" _I(long double c) {return std::complex<double> {0, static_cast<double>(c)};}


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

  const Arrayd& getPoling() {return _poling;};

protected:
  _NonlinearMedium(uint nSignalModes, bool canBePoled, double relativeLength, std::initializer_list<double> nlLength,
                   double beta2, std::initializer_list<double> beta2s, const Eigen::Ref<const Arraycd>& customPump, int pulseType,
                   double beta1, std::initializer_list<double> beta1s, double beta3, std::initializer_list<double> beta3s,
                   std::initializer_list<double> diffBeta0, double rayleighLength, double tMax, uint tPrecision, uint zPrecision,
                   double chirp, double delay, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));

  inline void setLengths(double relativeLength, const std::vector<double>& nlLength, uint zPrecision, double rayleighLength,
                         double beta2, const std::vector<double>& beta2s, double beta1, const std::vector<double>& beta1s,
                         double beta3, const std::vector<double>& beta3s);
  inline void resetGrids(uint nFreqs, double tMax);
  inline void setDispersion(double beta2, const std::vector<double>& beta2s, double beta1, const std::vector<double>& beta1s,
                            double beta3, const std::vector<double>& beta3s, std::initializer_list<double> diffBeta0);
  _NonlinearMedium() : _nSignalModes() {};
  _NonlinearMedium(uint nSignalModes) : _nSignalModes(nSignalModes) {}

  virtual void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                   std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime) = 0;

  template<class T>
  void signalSimulationTemplate(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime);

  void setPoling(const Eigen::Ref<const Arrayd>& poling);

  static inline Arrayd fftshift(const Arrayd& input);
  static inline Array2Dcd fftshift2(const Array2Dcd& input);

  const uint _nSignalModes; /// Number of separate signal modes (eg polarizations, frequencies, etc)
  double _z;  /// length of medium
  double _dz;    /// length increment of the signal simulation
  double _dzp;   /// length increment of the pump simulation
  uint _nZSteps; /// number of length steps in simulating the PDE
  uint _nZStepsP; /// number of length steps in simulating the pump, larger to calculate values at RK4 intermediate steps
  uint _nFreqs;  /// number of frequency/time bins in the simulating the PDE
  double _tMax;  /// positive and negative extent of the simulation window in time
  double _beta2;  /// second order dispersion of the pump
  double _beta1;  /// relative group velocity of the pump
  double _rayleighLength; /// Rayleigh length of propagation, assumes focused at medium's center

  std::vector<double> _diffBeta0; /// wave-vector mismatch of the simulated process
  std::vector<std::complex<double>> _nlStep; /// strength of nonlinear process over length dz

  Arraycd _env; /// initial envelope of the pump
  Arrayd _poling; /// array representing the poling direction at a given point on the grid.

  Arrayd _tau;   /// array representing the time axis
  Arrayd _omega; /// array representing the frequency axis

  Arrayd _dispersionPump; /// dispersion profile of pump
  std::vector<Arrayd> _dispersionSign; /// dispersion profile of signal
  Arraycd _dispStepPump; /// incremental phase due to dispersion over length dz for the pump
  std::vector<Arraycd> _dispStepSign; /// incremental phase due to dispersion over length dz for the signal

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
  inline void DiffEq(uint i, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, \
                     std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal); \
  void runSignalSimulation(const Arraycd& inputProf, bool inTimeDomain, uint inputMode, \
                           std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime) override \
     { signalSimulationTemplate<T>(inputProf, inTimeDomain, inputMode, signalFreq, signalTime); };


// This way is verified to be the most efficient use of the FFT functions, avoiding allocation of temporaries.
// Note: it seems that some compilers will throw a taking address of temporary error in EigenFFT.
// This is due to array->matrix casting, the code will work if disabling the warning and compiling.
#define FFT(output, input) { \
  fftObj.fwd(fftTemp, (input).matrix()); \
  output = fftTemp.array(); }
#define FFTtimes(output, input, phase) { \
  fftObj.fwd(fftTemp, (input).matrix()); \
  output = fftTemp.array() * phase; }
#define IFFT(output, input) { \
  fftObj.inv(fftTemp, (input).matrix()); \
  output = fftTemp.array(); }


template<class T>
void _NonlinearMedium::signalSimulationTemplate(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                                std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime) {
  RowVectorcd fftTemp(_nFreqs);

  // Can specify: input to any 1 mode by passing a length N array, or an input to the first x consecutive modes with a length x*N array
  uint nInputChannels = inputProf.size() / _nFreqs;
  if (nInputChannels > 1) inputMode = 0;
  if (T::_nSignalModes <= 1) inputMode = 0; // compiler guarantee

  if (inTimeDomain)
    for (uint m = 0; m < T::_nSignalModes; m++) {
      if (m == inputMode) {
        signalFreq[m].row(0) = inputProf.segment(0, _nFreqs); // hack: fft on inputProf sometimes fails
        FFTtimes(signalFreq[m].row(0), signalFreq[m].row(0), ((0.5_I * _dz) * _dispersionSign[m]).exp())
      }
      else if (inputMode < 1 && m < nInputChannels) {
        signalFreq[m].row(0) = inputProf.segment(m*_nFreqs, _nFreqs); // hack: fft on inputProf sometimes fails
        FFTtimes(signalFreq[m].row(0), signalFreq[m].row(0), ((0.5_I * _dz) * _dispersionSign[m]).exp())
      }
      else
        signalFreq[m].row(0) = 0;
    }
  else
    for (uint m = 0; m < T::_nSignalModes; m++) {
      if (m == inputMode)
        signalFreq[m].row(0) = inputProf.segment(0, _nFreqs) * ((0.5_I * _dz) * _dispersionSign[m]).exp();
      else if (inputMode < 1 && m < nInputChannels)
        signalFreq[m].row(0) = inputProf.segment(m*_nFreqs, _nFreqs) * ((0.5_I * _dz) * _dispersionSign[m]).exp();
      else
        signalFreq[m].row(0) = 0;
    }
  for (uint m = 0; m < T::_nSignalModes; m++) {
    if (m == inputMode || m < nInputChannels)
    IFFT(signalTime[m].row(0), signalFreq[m].row(0))
    else
      signalTime[m].row(0) = 0;
  }

  Arraycd temp(_nFreqs);
  std::vector<Arraycd> k1(T::_nSignalModes), k2(T::_nSignalModes), k3(T::_nSignalModes), k4(T::_nSignalModes);
  for (uint m = 0; m < T::_nSignalModes; m++) {
    k1[m].resize(_nFreqs); k2[m].resize(_nFreqs); k3[m].resize(_nFreqs); k4[m].resize(_nFreqs);
  }
  for (uint i = 1; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the non-linear propagation
    static_cast<T*>(this)->DiffEq(i, k1, k2, k3, k4, signalTime);

    for (uint m = 0; m < T::_nSignalModes; m++) {
      temp = signalTime[m].row(i - 1) + (k1[m] + 2 * k2[m] + 2 * k3[m] + k4[m]) / 6;

      // Dispersion step
      FFTtimes(signalFreq[m].row(i), temp, _dispStepSign[m])
      IFFT(signalTime[m].row(i), signalFreq[m].row(i))
    }
  }

  for (uint m = 0; m < T::_nSignalModes; m++) {
    signalFreq[m].row(_nZSteps - 1) *= ((-0.5_I * _dz) * _dispersionSign[m]).exp();
    IFFT(signalTime[m].row(_nZSteps - 1), signalFreq[m].row(_nZSteps - 1))
  }
}


#endif //NONLINEARMEDIUM