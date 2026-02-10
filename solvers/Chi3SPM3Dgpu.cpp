#ifndef CHI3SPM3D
#define CHI3SPM3D

#include "_FullyNonlinearMedium.hpp"

class Chi3SPM3D : public _FullyNonlinearMedium {
  NLM_CustomDispersion(Chi3SPM3D, 1, 3, 1)
public:
  Chi3SPM3D(double relativeLength, double nlLength, double indexLength, double groupIndexLength,
            double beta2s, double beta2t, double beta1st,
            const Eigen::Ref<const Arraycd>& indexDiffProfile=Eigen::Ref<const Arraycd>(Arraycd{}),
            double rayleighLength=std::numeric_limits<double>::infinity(), double sMax=10, double tMax=10,
            uint sPrecision=256, uint tPrecision=256, uint zPrecision=100, uint ratioStepsToRecord=1,
            IntensityProfile intensityProfile=IntensityProfile{});

private:
  at::Tensor _indexProfileSPMStep;
  at::Tensor _indexProfileStep;

  void spaceTimeCoupling(double beta2t, double beta1st); /// apply time delay to transverse dimensions
  void groupPhaseIndex(const Eigen::Ref<const Arraycd>& indexDiffProfile); /// apply phase and group delay based on index
};

void Chi3SPM3D::spaceTimeCoupling(double beta2t, double beta1st) {
  // same as multiDimensionalArithmetic but simplified since we only apply to the first dimension (time)
  // switch time dimension
  Arrayd coupledDispersionProfile = (1. + beta1st * _omega[2]).cwiseInverse(); // beta2 should always be negative, so beta1st should be positive

  // iterate over each of the first nx*ny pixels, which are on the boundary of the time dimension, and apply time delay profile
  for (uint i = 0, absIndex = 0; i < _nFreqsPerDim[0] * _nFreqsPerDim[1]; i++, absIndex += _nFreqsPerDim[2]) {
    Eigen::Map<Arrayd, 0> stridedViewDispProf(_dispersionSign[0].data() + absIndex, _nFreqsPerDim[2]);
    stridedViewDispProf *= coupledDispersionProfile;
    stridedViewDispProf += (0.5 * beta2t) * _omega[2].square();
  }
  _dispStepSign[0] = ((1._I * _dz) * _dispersionSign[0]).exp() * (1. / _nFreqs); // Recompute
}

void Chi3SPM3D::groupPhaseIndex(const Eigen::Ref<const Arraycd>& indexDiffProfile) {
  const uint nSlices = _nFreqsPerDim[0] * _nFreqsPerDim[1]; // switch time dimension
  const uint slice = _nFreqsPerDim[2];
  Arraycd indexProfileStep(_nFreqs);
  for (uint j = 0, ind = 0; j < nSlices; j++, ind += slice)
    indexProfileStep.segment(ind, slice) = (_nlStep[1] * indexDiffProfile[j] + _nlStep[2] * indexDiffProfile[j].real() * _omega[2]).exp();

  _indexProfileStep = at::from_blob(indexProfileStep.data(), {1, _nFreqsPerDim[0], _nFreqsPerDim[1], _nFreqsPerDim[2]}, optionsMM).to(theGPUdevice);
}


Chi3SPM3D::Chi3SPM3D(double relativeLength, double nlLength, double indexLength, double groupIndexLength,
                     double beta2t, double beta2s, double beta1st, const Eigen::Ref<const Arraycd>& indexDiffProfile,
                     double rayleighLength, double tMax, double sMax, uint tPrecision, uint sPrecision, uint zPrecision,
                     uint ratioStepsToRecord, IntensityProfile intensityProfile) :
    _FullyNonlinearMedium(_nSignalModes, _nDimensions, false, _nTemps, 0, relativeLength,
                          {nlLength, indexLength / indexDiffProfile.real().abs().maxCoeff(), groupIndexLength / indexDiffProfile.real().abs().maxCoeff()},
                          {beta2s, beta2s, 0}, {0, 0, 0}, // switch time dimension
                          {0, 0, 0}, {}, rayleighLength, {sMax, sMax, tMax}, {sPrecision, sPrecision, tPrecision}, zPrecision, ratioStepsToRecord, intensityProfile)
{
  if (indexDiffProfile.size() != sPrecision * sPrecision)
    throw std::invalid_argument("Index profile array length does not match the number of relevant simulation bins");
  Arraycd indexProfileSPMStep = _nlStep[0] * (1 - 2 * indexDiffProfile.real()); // NB approximation for 1/(n+dn)^2
  _indexProfileSPMStep = at::from_blob(indexProfileSPMStep.data(), {1, _nFreqsPerDim[0], _nFreqsPerDim[1], 1}, optionsMM).to(theGPUdevice);

  Arrayd steepening = (1. + beta1st * _omega[2]) / _nFreqsPerDim[2]; // switch time dimension
  at::Tensor _steepening = at::from_blob(steepening.data(), {1, 1, 1, tPrecision}, optionsR).to(theGPUdevice); // real vector
  _indexProfileSPMStep = _indexProfileSPMStep * _steepening;

  double indexDiff = indexDiffProfile.real().abs().maxCoeff();
  _nlStep[1] /= indexDiff;
  _nlStep[2] /= indexDiff;

  spaceTimeCoupling(beta2t, beta1st); // note we pass 0 for beta2t above so we can apply space-time coupling to purely space dispersion and add time after
  _dispStepSignGPU = at::from_blob(_dispStepSign[0].data(), {1, _nFreqsPerDim[0], _nFreqsPerDim[1], _nFreqsPerDim[2]}, optionsMM).to(theGPUdevice);
  groupPhaseIndex(indexDiffProfile);
}


inline void Chi3SPM3D::Dispersion(at::Tensor& signalTime, at::Tensor& signalFreq) {
  at::Tensor temps = at::fft_fft(signalTime, std::nullopt, -1, "backward");
  temps *= _indexProfileStep;
  signalFreq.index_put_({0}, at::fft_fft2(temps, std::nullopt, {-3, -2}, "backward"));
  signalFreq *= _dispStepSignGPU;
  signalTime.index_put_({0}, at::fft_ifftn(signalFreq, std::nullopt, {-3, -2, -1}, "forward"));
}

void Chi3SPM3D::DiffEq(uint i, at::Tensor& k1, at::Tensor& k2, at::Tensor& k3, at::Tensor& k4, const at::Tensor& signal) {

  const double relIntPrv = relativeIntensity(i- 1);
  const double relIntInt = relativeIntensity(i-.5);
  const double relIntCur = relativeIntensity(i);

  k1.index_put_({0}, relIntPrv * (at::square(at::real(signal)) + at::square(at::imag(signal))) * signal); // TODO use add_ and mul_
  at::Tensor temps = at::fft_fft(k1, std::nullopt, -1, "backward");
  temps *= _indexProfileSPMStep;
  k1.index_put_({0}, at::fft_ifft(temps, std::nullopt, -1, "forward"));

  temps = (signal + .5 * k1);
  k2.index_put_({0}, relIntInt * (at::square(at::real(temps)) + at::square(at::imag(temps))) * temps);
  temps = at::fft_fft(k2, std::nullopt, -1, "backward");
  temps *= _indexProfileSPMStep;
  k2.index_put_({0}, at::fft_ifft(temps, std::nullopt, -1, "forward"));

  temps = (signal + .5 * k2);
  k3.index_put_({0}, relIntInt * (at::square(at::real(temps)) + at::square(at::imag(temps))) * temps);
  temps = at::fft_fft(k3, std::nullopt, -1, "backward");
  temps *= _indexProfileSPMStep;
  k3.index_put_({0}, at::fft_ifft(temps, std::nullopt, -1, "forward"));

  temps = (signal + k3);
  k4.index_put_({0}, relIntCur * (at::square(at::real(temps)) + at::square(at::imag(temps))) * temps);
  temps = at::fft_fft(k4, std::nullopt, -1, "backward");
  temps *= _indexProfileSPMStep;
  k4.index_put_({0}, at::fft_ifft(temps, std::nullopt, -1, "forward"));
}

#endif //CHI3SPM3D

#ifdef NLMMODULE
py::class_<Chi3SPM3D, _FullyNonlinearMedium> Chi3SPM3D(m, "Chi3SPM3D", "Fully nonlinear Kerr medium, with self focusing, steepening, and spatially varying refractive and group indices.");
Chi3SPM3D.def(
    py::init<double, double, double, double, double, double, double,
             const Eigen::Ref<const Arraycd>&,
             double, double, double, uint, uint, uint, uint,
             _NonlinearMedium::IntensityProfile>(),
    "relativeLength"_a, "nlLength"_a,  "indexLength"_a, "groupIndexLength"_a,
    "beta2t"_a, "beta2s"_a,  "beta1st"_a, "indexDiffProfile"_a = defArraycd,
    "rayleighLength"_a = infinity, "tMax"_a = 10, "sMax"_a = 10,
    "tPrecision"_a = 512, "sPrecision"_a = 512, "zPrecision"_a = 100, "ratioStepsToRecord"_a = 1,
    "intensityProfile"_a = _NonlinearMedium::IntensityProfile{});
#endif