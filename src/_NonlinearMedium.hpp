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

  virtual void setPump(PulseType pulseType, const std::vector<double>& chirpLength={}, const std::vector<double>& delayLength={}, uint pumpIndex=0);
  virtual void setPump(const Eigen::Ref<const Arraycd>& customPump, const std::vector<double>& chirpLength={}, const std::vector<double>& delayLength={}, uint pumpIndex=0);
  virtual void setPump(const _NonlinearMedium& other, uint signalIndex, const std::vector<double>& delayLength={}, uint pumpIndex=0);

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
  const Arrayd& getTime(uint i=0)      {return _tau.at(i);};
  const Arrayd& getFrequency(uint i=0) {return _omega.at(i);};

  Array2Dcd& getField(uint i=0) {return field.at(i);};
  const Arrayd& getPoling() {return _poling;};

protected:
  _NonlinearMedium(uint nSignalModes, uint nDimensions, uint nPumpModes, bool canBePoled, uint nFieldModes,
                   double relativeLength, std::initializer_list<double> nlLength,
                   std::initializer_list<double> beta2, std::initializer_list<double> beta2s,
                   std::initializer_list<double> beta1, std::initializer_list<double> beta1s,
                   std::initializer_list<double> beta3, std::initializer_list<double> beta3s,
                   std::initializer_list<double> diffBeta0, double rayleighLength,
                   std::initializer_list<double> tMax, std::initializer_list<uint> tPrecision,
                   uint zPrecision, uint ratioStepsToRecord, IntensityProfile intensityProfile,
                   const Eigen::Ref<const Arrayd>& poling=Eigen::Ref<const Arrayd>(Arrayd{}));

  void setLengths(double relativeLength, const std::vector<double>& nlLength, uint zPrecision, double rayleighLength,
                  const std::vector<double>& beta2, const std::vector<double>& beta2s, const std::vector<double>& beta1,
                  const std::vector<double>& beta1s, const std::vector<double>& beta3, const std::vector<double>& beta3s);
  void resetGrids(const std::vector<uint>& nFreqs, const std::vector<double>& tMax, uint ratioStepsToRecord);
  void setDispersion(const std::vector<double>& beta2, const std::vector<double>& beta2s, const std::vector<double>& beta1,
                     const std::vector<double>& beta1s, const std::vector<double>& beta3, const std::vector<double>& beta3s,
                     std::initializer_list<double> diffBeta0);
  _NonlinearMedium() : _nSignalModes(), _nPumpModes(), _nFieldModes(), _nDimensions() {};

  virtual void dispatchSignalSim(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                 std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime,
                                 uint ratioStepsToRecord) = 0;

  template<class T>
  void signalSimulationTemplate(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime, uint ratioStepsToRecord);

  void setPoling(const Eigen::Ref<const Arrayd>& poling);

  static inline Arrayd fftshift(const Arrayd& input);
  static inline Array2Dcd fftshift2(const Array2Dcd& input);

  template<typename ArrayType, bool doMultiply>
  void multiDimensionalArithmetic(ArrayType& ndArray, const std::vector<ArrayType>& factor);
  void setPhases(const std::vector<double>& chirpLength, const std::vector<double>& delayLength, uint pumpIndex);

  const uint _nSignalModes; /// Number of separate signal modes (eg polarizations, wavelengths, etc)
  const uint _nPumpModes;   /// Number of separate pump modes (eg polarizations, wavelengths, etc)
  const uint _nFieldModes;  /// Number of separate field modes (eg index variation, 2D poling, etc)
  const uint _nDimensions;  /// Number dimensions
  double _z;      /// length of medium
  double _dz;     /// length increment of the signal simulation
  double _dzp;    /// length increment of the pump simulation
  uint _nZSteps;  /// number of length steps in simulating the PDE
  uint _nZStepsP; /// number of length steps in simulating the pump, larger to calculate values at RK4 intermediate steps
  uint _nFreqs;   /// number of frequency/time bins in simulating the PDE
  uint _ratioStepsToRecord; /// number of steps to skip when filling in signalFreq and signalTime
  std::vector<uint> _nFreqsPerDim; /// number of frequency/time bins per dimension
  std::vector<double> _tMax;   /// positive and negative extent of the simulation window in time
  double _rayleighLength; /// Rayleigh length of propagation (or characteristic length of intensity profile), assumes focused at medium's center
  IntensityProfile _intensityProfile; /// Encodes the intensity profile type, if not Gaussian beam propagation
  std::vector<double> _beta2;  /// second order dispersion of the pump
  std::vector<double> _beta1;  /// relative group velocity of the pump

  std::vector<double> _diffBeta0; /// wave-vector mismatch of the simulated process
  std::vector<std::complex<double>> _nlStep; /// strength of nonlinear process over length dz

  std::vector<Arraycd> _envelope; /// initial envelope of the pump
  Arrayd _poling; /// array representing the poling direction at a given point on the grid.

  std::vector<Arrayd> _tau;   /// array representing the time or transverse axis (one per dimension)
  std::vector<Arrayd> _omega; /// array representing the frequency axis (one per dimension)

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
    switch (_nDimensions) {
      default:
      case 1:
        fftObj.fwd(output, input, _nFreqs);
        break;
      case 2:
        fftObj.fwd2(output, input, 0, 0, _nFreqsPerDim[0], _nFreqsPerDim[1]);
        break;
      case 3:
        fftObj.fwd3(output, input, 0, 0, _nFreqsPerDim[0], _nFreqsPerDim[1], _nFreqsPerDim[2]);
        break;
    }
  }
  inline void IFFT(Arraycd& output, const Arraycd& input) const {
    switch (_nDimensions) {
      default:
      case 1:
        fftObj.inv(output, input, _nFreqs);
        break;
      case 2:
        fftObj.inv2(output, input, 0, 0, _nFreqsPerDim[0], _nFreqsPerDim[1]);
        break;
      case 3:
        fftObj.inv3(output, input, 0, 0, _nFreqsPerDim[0], _nFreqsPerDim[1], _nFreqsPerDim[2]);
        break;
    }
  }
  inline void FFTi(Array2Dcd& output, const Array2Dcd& input, Eigen::DenseIndex rowOut, Eigen::DenseIndex rowIn) const {
    fftObj.fwd(output, input, rowOut, rowIn, _nFreqs);
  }
  inline void IFFTi(Array2Dcd& output, const Array2Dcd& input, Eigen::DenseIndex rowOut, Eigen::DenseIndex rowIn) const {
    fftObj.inv(output, input, rowOut, rowIn, _nFreqs);
  }
  inline void FFT2i(Array2Dcd& output, const Array2Dcd& input, Eigen::DenseIndex rowOut, Eigen::DenseIndex rowIn) const {
    fftObj.fwd2(output, input, rowOut, rowIn, _nFreqsPerDim[0], _nFreqsPerDim[1]);
  }
  inline void IFFT2i(Array2Dcd& output, const Array2Dcd& input, Eigen::DenseIndex rowOut, Eigen::DenseIndex rowIn) const {
    fftObj.inv2(output, input, rowOut, rowIn, _nFreqsPerDim[0], _nFreqsPerDim[1]);
  }
  inline void FFT3i(Array2Dcd& output, const Array2Dcd& input, Eigen::DenseIndex rowOut, Eigen::DenseIndex rowIn) const {
    fftObj.fwd3(output, input, rowOut, rowIn, _nFreqsPerDim[0], _nFreqsPerDim[1], _nFreqsPerDim[2]);
  }
  inline void IFFT3i(Array2Dcd& output, const Array2Dcd& input, Eigen::DenseIndex rowOut, Eigen::DenseIndex rowIn) const {
    fftObj.inv3(output, input, rowOut, rowIn, _nFreqsPerDim[0], _nFreqsPerDim[1],_nFreqsPerDim[2]);
  }
};


// Repeated code for each NLM ODE class. This takes care of:
// - Allowing _NonlinearMedium friend access to the protected DiffEq function, to use in signalSimulationTemplate
// - Overriding runSignalSimulation with the function created from the template
#define NLM(T, modes, dimensions) \
protected: \
  friend _NonlinearMedium; \
  constexpr static uint _nSignalModes = modes; \
  constexpr static uint _nDimensions = dimensions; \
  static_assert(_nDimensions <= 3, "Only up to 3 dimensions currently supported"); \
  inline void DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3, \
                     std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal); \
  void dispatchSignalSim(const Arraycd& inputProf, bool inTimeDomain, uint inputMode, \
                         std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime,         \
                         uint ratioStepsToRecord) override \
     { signalSimulationTemplate<T>(inputProf, inTimeDomain, inputMode, signalFreq, signalTime, ratioStepsToRecord); };


template<class T>
void _NonlinearMedium::signalSimulationTemplate(const Arraycd& inputProf, bool inTimeDomain, uint inputMode,
                                                std::vector<Array2Dcd>& signalFreq, std::vector<Array2Dcd>& signalTime,
                                                uint ratioStepsToRecord) {
  // Can specify: input to any 1 mode by passing a length N array, or an input to the first x consecutive modes with a length x*N array
  uint nInputChannels = inputProf.size() / _nFreqs;
  if (nInputChannels > 1) inputMode = 0;
  if constexpr (T::_nSignalModes <= 1) inputMode = 0; // compiler guarantee

  auto fft = [this](Array2Dcd& a, Array2Dcd& b, uint i, uint j){
    if constexpr      (T::_nDimensions == 1)  FFTi(a, b, i, j);
    else if constexpr (T::_nDimensions == 2) FFT2i(a, b, i, j);
    else if constexpr (T::_nDimensions == 3) FFT3i(a, b, i, j);
  };
  auto ifft = [this](Array2Dcd& a, Array2Dcd& b, uint i, uint j){
    if constexpr      (T::_nDimensions == 1)  IFFTi(a, b, i, j);
    else if constexpr (T::_nDimensions == 2) IFFT2i(a, b, i, j);
    else if constexpr (T::_nDimensions == 3) IFFT3i(a, b, i, j);
  };

  if (inTimeDomain)
    for (uint m = 0; m < T::_nSignalModes; m++) {
      if (m == inputMode) {
        signalTime[m].row(0) = inputProf.segment(0, _nFreqs); // hack: fft on inputProf sometimes fails
        fft(signalFreq[m], signalTime[m], 0, 0);
        signalFreq[m].row(0) *= ((0.5_I * _dz) * _dispersionSign[m]).exp() * (1. / _nFreqs); // note scale factor included for FFT
      }
      else if (inputMode < 1 && m < nInputChannels) {
        signalTime[m].row(0) = inputProf.segment(m*_nFreqs, _nFreqs); // hack: fft on inputProf sometimes fails
        fft(signalFreq[m], signalTime[m], 0, 0);
        signalFreq[m].row(0) *= ((0.5_I * _dz) * _dispersionSign[m]).exp() * (1. / _nFreqs); // note scale factor included for FFT
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
    if (m == inputMode || m < nInputChannels) {
      ifft(signalTime[m], signalFreq[m], 0, 0);
    }
    else
      signalTime[m].row(0) = 0;
  }

  std::vector<Arraycd> k1(T::_nSignalModes), k2(T::_nSignalModes), k3(T::_nSignalModes), k4(T::_nSignalModes);
  for (uint m = 0; m < T::_nSignalModes; m++) {
    k1[m].resize(_nFreqs); k2[m].resize(_nFreqs); k3[m].resize(_nFreqs); k4[m].resize(_nFreqs);
  }
  for (uint i = 1, gridIndex = 0; i < _nZSteps; i++) {
    // Do a Runge-Kutta step for the nonlinear propagation
    static_cast<T*>(this)->DiffEq(i, gridIndex, k1, k2, k3, k4, signalTime);

    uint prevGridIndex = gridIndex;
    gridIndex = i / ratioStepsToRecord; // only saving one out of every n steps, otherwise overwrite with next step

    for (uint m = 0; m < T::_nSignalModes; m++) {
      signalTime[m].row(gridIndex) = signalTime[m].row(prevGridIndex) + (k1[m] + 2 * k2[m] + 2 * k3[m] + k4[m]) * (1. / 6.);

      // Dispersion step
      fft(signalFreq[m], signalTime[m], gridIndex, gridIndex);
      signalFreq[m].row(gridIndex) *= _dispStepSign[m];
      ifft(signalTime[m], signalFreq[m], gridIndex, gridIndex);
    }
  }

  for (uint m = 0; m < T::_nSignalModes; m++) {
    signalFreq[m].bottomRows<1>() *= ((-0.5_I * _dz) * _dispersionSign[m]).exp(); // note *no* scale factor included for FFT
    ifft(signalTime[m], signalFreq[m], signalTime[m].rows() - 1, signalFreq[m].rows() - 1);
  }
}


#endif //NONLINEARMEDIUM