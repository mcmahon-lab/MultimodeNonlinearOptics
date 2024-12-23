#ifndef NONLINEARMEDIUM
#define NONLINEARMEDIUM

#include <eigen3/Eigen/Core>
//#include <eigen3/unsupported/Eigen/FFT>
#include "CustomEigenFFT.h" // Note: using modified version instead
#include <utility>


// Eigen default 1D Array is defined with X rows, 1 column, which does not work with row-major order 2D arrays.
// Thus define custom double and complex double 1D arrays. Also define the row-major order 2D double and complex arrays.

typedef Eigen::Array<double, 1, Eigen::Dynamic, Eigen::RowMajor> Arrayd;
typedef Eigen::Array<std::complex<double>, 1, Eigen::Dynamic, Eigen::RowMajor> Arraycd;
typedef Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Array2Dcd;

inline constexpr std::complex<double> operator"" _I(long double c) {return std::complex<double> {0, static_cast<double>(c)};}


class _NonlinearMedium {
friend class Cascade;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum class PulseType : int {
    Gaussian = 0,
    Sech = 1,
    Sinc = 2,
  };
  enum class IntensityProfile : int {
    GaussianBeam = 0,
    Constant = 1,
    GaussianApodization = 2,
  };

  virtual void setPump(PulseType pulseType, double chirpLength=0, double delayLength=0, uint pumpIndex=0);
  virtual void setPump(const Eigen::Ref<const Arraycd>& customPump, double chirpLength=0, double delayLength=0, uint pumpIndex=0);
  virtual void setPump(const _NonlinearMedium& other, uint signalIndex, double delayLength=0, uint pumpIndex=0);

  virtual void runPumpSimulation();
  virtual void runSignalSimulation(const Eigen::Ref<const Arraycd>& inputProf, bool inTimeDomain=true, uint inputMode=0);
  virtual std::pair<Array2Dcd, Array2Dcd>
      computeGreensFunction(bool inTimeDomain=false, bool runPump=true, uint nThreads=1, bool normalize=false,
                            const std::vector<uint8_t>& useInput={}, const std::vector<uint8_t>& useOutput={});
  virtual Array2Dcd batchSignalSimulation(const Eigen::Ref<const Array2Dcd>& inputProfs, bool inTimeDomain=false,
                                          bool runPump=true, uint nThreads=1, uint inputMode=0, const std::vector<uint8_t>& useOutput={});

  const Array2Dcd& getPumpFreq(uint i=0) {return pumpFreq.at(i);};
  const Array2Dcd& getPumpTime(uint i=0) {return pumpTime.at(i);};
  const Array2Dcd& getSignalFreq(uint i=0) {return signalFreq.at(i);};
  const Array2Dcd& getSignalTime(uint i=0) {return signalTime.at(i);};
  const Arrayd& getTime()      {return _tau;};
  const Arrayd& getFrequency() {return _omega;};

  Array2Dcd& getField(uint i=0) {return field.at(i);};
  const Arrayd& getPoling() {return _poling;};

protected:
  _NonlinearMedium(uint nSignalModes, uint nPumpModes, bool canBePoled, uint nFieldModes,
                   double relativeLength, std::initializer_list<double> nlLength,
                   std::initializer_list<double> beta2, std::initializer_list<double> beta2s,
                   const Eigen::Ref<const Arraycd>& customPump, PulseType pulseType,
                   std::initializer_list<double> beta1, std::initializer_list<double> beta1s,
                   std::initializer_list<double> beta3, std::initializer_list<double> beta3s,
                   std::initializer_list<double> diffBeta0,
                   double rayleighLength, double tMax, uint tPrecision, uint zPrecision, IntensityProfile intensityProfile,
                   double chirp, double delay, const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));

  void setLengths(double relativeLength, const std::vector<double>& nlLength, uint zPrecision, double rayleighLength,
                  const std::vector<double>& beta2, const std::vector<double>& beta2s, const std::vector<double>& beta1,
                  const std::vector<double>& beta1s, const std::vector<double>& beta3, const std::vector<double>& beta3s);
  void resetGrids(uint nFreqs, double tMax);
  void setDispersion(const std::vector<double>& beta2, const std::vector<double>& beta2s, const std::vector<double>& beta1,
                     const std::vector<double>& beta1s, const std::vector<double>& beta3, const std::vector<double>& beta3s,
                     std::initializer_list<double> diffBeta0);
  _NonlinearMedium() : _nSignalModes(), _nPumpModes(), _nFieldModes() {};
  _NonlinearMedium(uint nSignalModes, uint nFieldModes) : _nSignalModes(nSignalModes), _nPumpModes(), _nFieldModes(nFieldModes) {}

  virtual void dispatchSignalSim(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                 std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime,
                                 bool optimized) = 0;

  template<class T>
  void signalSimulationTemplate(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime, bool optimized);

  void setPoling(const Eigen::Ref<const Arrayd>& poling);

  static inline Arrayd fftshift(const Arrayd& input);
  static inline Array2Dcd fftshift2(const Array2Dcd& input);

  const uint _nSignalModes; /// Number of separate signal modes (eg polarizations, wavelengths, etc)
  const uint _nPumpModes;   /// Number of separate pump modes (eg polarizations, wavelengths, etc)
  const uint _nFieldModes;   /// Number of separate field modes (eg index variation, 2D poling, etc)
  double _z;      /// length of medium
  double _dz;     /// length increment of the signal simulation
  double _dzp;    /// length increment of the pump simulation
  uint _nZSteps;  /// number of length steps in simulating the PDE
  uint _nZStepsP; /// number of length steps in simulating the pump, larger to calculate values at RK4 intermediate steps
  uint _nFreqs;   /// number of frequency/time bins in the simulating the PDE
  double _tMax;   /// positive and negative extent of the simulation window in time
  double _rayleighLength; /// Rayleigh length of propagation (or characteristic length of intensity profile), assumes focused at medium's center
  IntensityProfile _intensityProfile; /// Encodes the intensity profile type, if not Gaussian beam propagation
  std::vector<double> _beta2;  /// second order dispersion of the pump
  std::vector<double> _beta1;  /// relative group velocity of the pump

  std::vector<double> _diffBeta0; /// wave-vector mismatch of the simulated process
  std::vector<std::complex<double>> _nlStep; /// strength of nonlinear process over length dz

  std::vector<Arraycd> _envelope; /// initial envelope of the pump
  Arrayd _poling; /// array representing the poling direction at a given point on the grid.

  Arrayd _tau;   /// array representing the time axis
  Arrayd _omega; /// array representing the frequency axis

  std::vector<Arrayd> _dispersionPump; /// dispersion profile of pump
  std::vector<Arrayd> _dispersionSign; /// dispersion profile of signal
  std::vector<Arraycd> _dispStepPump; /// incremental phase due to dispersion over length dz for the pump
  std::vector<Arraycd> _dispStepSign; /// incremental phase due to dispersion over length dz for the signal

  std::vector<Array2Dcd> pumpFreq; /// grid for numerically solving PDE, representing pump propagation in frequency domain
  std::vector<Array2Dcd> pumpTime; /// grid for numerically solving PDE, representing pump propagation in time domain
  std::vector<Array2Dcd> signalFreq; /// grid for numerically solving PDE, representing signal propagation in frequency domain
  std::vector<Array2Dcd> signalTime; /// grid for numerically solving PDE, representing signal propagation in time domain

  std::vector<Array2Dcd> field; /// grid for a user-defined field to include in the PDE

  static Eigen::FFT<double> fftObj; /// fft class object for performing dft

  // DFT Convenience Functions, indexed (for 2D arrays) and regular (for 1D arrays):
  inline void FFT(Arraycd& output, const Arraycd& input) const {
    fftObj.fwd(output, input, _nFreqs);
  }
  inline void IFFT(Arraycd& output, const Arraycd& input) const {
    fftObj.inv(output, input, _nFreqs);
  }
  inline void FFTi(Array2Dcd& output, const Array2Dcd& input, Eigen::DenseIndex rowOut, Eigen::DenseIndex rowIn) const {
    fftObj.fwd(output, input, rowOut, rowIn, _nFreqs);
  }
  inline void IFFTi(Array2Dcd& output, const Array2Dcd& input, Eigen::DenseIndex rowOut, Eigen::DenseIndex rowIn) const {
    fftObj.inv(output, input, rowOut, rowIn, _nFreqs);
  }
};


// Repeated code for each NLM ODE class. This takes care of:
// - Allowing _NonlinearMedium friend access to the protected DiffEq function, to use in signalSimulationTemplate
// - Overriding runSignalSimulation with the function created from the template
#define NLM(T, modes) \
protected: \
  friend _NonlinearMedium; \
  constexpr static uint _nSignalModes = modes; \
  inline void DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, \
                     std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal); \
  void dispatchSignalSim(const Arraycd& inputProf, bool inTimeDomain, uint inputMode, \
                         std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime,         \
                         bool optimized) override \
     { signalSimulationTemplate<T>(inputProf, inTimeDomain, inputMode, signalFreq, signalTime, optimized); };


template<class T>
void _NonlinearMedium::signalSimulationTemplate(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                                std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime,
                                                bool optimized) {
  // Can specify: input to any 1 mode by passing a length N array, or an input to the first x consecutive modes with a length x*N array
  uint nInputChannels = inputProf.size() / _nFreqs;
  if (nInputChannels > 1) inputMode = 0;
  if (T::_nSignalModes <= 1) inputMode = 0; // compiler guarantee

  if (inTimeDomain)
    for (uint m = 0; m < T::_nSignalModes; m++) {
      if (m == inputMode) {
        signalTime[m].row(0) = inputProf.segment(0, _nFreqs); // hack: fft on inputProf sometimes fails
        FFTi(signalFreq[m], signalTime[m], 0, 0);
        signalFreq[m].row(0) *= ((0.5_I * _dz) * _dispersionSign[m]).exp();
      }
      else if (inputMode < 1 && m < nInputChannels) {
        signalTime[m].row(0) = inputProf.segment(m*_nFreqs, _nFreqs); // hack: fft on inputProf sometimes fails
        FFTi(signalFreq[m], signalTime[m], 0, 0);
        signalFreq[m].row(0) *= ((0.5_I * _dz) * _dispersionSign[m]).exp();
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
      IFFTi(signalTime[m], signalFreq[m], 0, 0);
    else
      signalTime[m].row(0) = 0;
  }

  std::vector<Arraycd> k1(T::_nSignalModes), k2(T::_nSignalModes), k3(T::_nSignalModes), k4(T::_nSignalModes);
  for (uint m = 0; m < T::_nSignalModes; m++) {
    k1[m].resize(_nFreqs); k2[m].resize(_nFreqs); k3[m].resize(_nFreqs); k4[m].resize(_nFreqs);
  }
  if (optimized) { // for batchSignalSimulation or computeGreensFunction, where we use a single row instead of a grid
    for (uint i = 1; i < _nZSteps; i++) {
      // Do a Runge-Kutta step for the non-linear propagation
      static_cast<T*>(this)->DiffEq(i, 0, k1, k2, k3, k4, signalTime);

      for (uint m = 0; m < T::_nSignalModes; m++) {
        signalTime[m].row(0) += (k1[m] + 2 * k2[m] + 2 * k3[m] + k4[m]) * (1. / 6.);

        // Dispersion step
        FFTi(signalFreq[m], signalTime[m], 0, 0);
        signalFreq[m].row(0) *= _dispStepSign[m];
        IFFTi(signalTime[m], signalFreq[m], 0, 0);
      }
    }
  }
  else { // for the regular case of filling in the PDE grids
    for (uint i = 1; i < _nZSteps; i++) {
      // Do a Runge-Kutta step for the non-linear propagation
      static_cast<T*>(this)->DiffEq(i, i-1, k1, k2, k3, k4, signalTime);

      for (uint m = 0; m < T::_nSignalModes; m++) {
        signalTime[m].row(i) = signalTime[m].row(i - 1) + (k1[m] + 2 * k2[m] + 2 * k3[m] + k4[m]) * (1. / 6.);

        // Dispersion step
        FFTi(signalFreq[m], signalTime[m], i, i);
        signalFreq[m].row(i) *= _dispStepSign[m];
        IFFTi(signalTime[m], signalFreq[m], i, i);
      }
    }
  }

  for (uint m = 0; m < T::_nSignalModes; m++) {
    signalFreq[m].bottomRows<1>() *= ((-0.5_I * _dz) * _dispersionSign[m]).exp();
    IFFTi(signalTime[m], signalFreq[m], signalTime[m].rows() - 1, signalFreq[m].rows() - 1);
  }
}


#endif //NONLINEARMEDIUM