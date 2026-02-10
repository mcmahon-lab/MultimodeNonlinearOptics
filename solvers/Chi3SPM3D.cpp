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
  Arraycd _indexProfileSPMStep;
  Arraycd _indexProfileStep;
  Arrayd _steepening;

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
  _indexProfileStep.resize(_nFreqs);
  for (uint j = 0, ind = 0; j < nSlices; j++, ind += slice)
    _indexProfileStep.segment(ind, slice) = (_nlStep[1] * indexDiffProfile[j] + _nlStep[2] * indexDiffProfile[j].real() * _omega[2]).exp();
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
  _indexProfileSPMStep = _nlStep[0] * (1 - 2 * indexDiffProfile.real()); // NB: approximation of 1/(n+dn)^2
  _steepening = (1. + beta1st * _omega[2]) / _nFreqsPerDim[2]; // switch time dimension

   double indexDiff = indexDiffProfile.real().abs().maxCoeff();
  _nlStep[1] /= indexDiff;
  _nlStep[2] /= indexDiff;

  spaceTimeCoupling(beta2t, beta1st); // note we pass 0 for beta2t above so we can apply space-time coupling to purely space dispersion and add time after
  groupPhaseIndex(indexDiffProfile);
}


inline void Chi3SPM3D::Dispersion(uint m, uint gridIndex, std::vector<Array2Dcd>& signalTime, std::vector<Array2Dcd>& signalFreq,
                                  std::vector<Arraycd>& temps) {
  FFTpi(temps[0], signalTime[m], 0, gridIndex, false, false, true); // TODO also need to do the half steps for the beginning and end
  temps[0] *= _indexProfileStep;
  FFTpi(signalFreq[m], temps[0], gridIndex, 0, true, true, false);
  signalFreq[m].row(gridIndex) *= _dispStepSign[m];
  IFFT3i(signalTime[m], signalFreq[m], gridIndex, gridIndex);
}

void Chi3SPM3D::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                       std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal, std::vector<Arraycd>& temps) {

  const auto& prv = signal[0].row(iPrevSig);

  const double relIntPrv = relativeIntensity(i- 1);
  const double relIntInt = relativeIntensity(i-.5);
  const double relIntCur = relativeIntensity(i);

  const uint nSlices = _nFreqsPerDim[0] * _nFreqsPerDim[1]; // switch time dimension
  const uint slice   = _nFreqsPerDim[2];

  k1[0] = relIntPrv * prv.abs2() * prv;
  FFTp(temps[0], k1[0], false, false, true);
  for (uint j = 0, ind = 0; j < nSlices; j++, ind += slice)
    temps[0].segment(ind, slice) *= _steepening * _indexProfileSPMStep[j];
  IFFTp(k1[0], temps[0], false, false, true);

  k2[0] = relIntInt * (prv + .5 * k1[0]) * (prv + .5 * k1[0]).abs2();
  FFTp(temps[0], k2[0], false, false, true);
  for (uint j = 0, ind = 0; j < nSlices; j++, ind += slice)
    temps[0].segment(ind, slice) *= _steepening * _indexProfileSPMStep[j];
  IFFTp(k2[0], temps[0], false, false, true);

  k3[0] = relIntInt * (prv + .5 * k2[0]) * (prv + .5 * k2[0]).abs2();
  FFTp(temps[0], k3[0], false, false, true);
  for (uint j = 0, ind = 0; j < nSlices; j++, ind += slice)
    temps[0].segment(ind, slice) *= _steepening * _indexProfileSPMStep[j];
  IFFTp(k3[0], temps[0], false, false, true);

  k4[0] = relIntCur * (prv + k3[0]) * (prv + k3[0]).abs2();
  FFTp(temps[0], k4[0], false, false, true);
  for (uint j = 0, ind = 0; j < nSlices; j++, ind += slice)
    temps[0].segment(ind, slice) *= _steepening * _indexProfileSPMStep[j];
  IFFTp(k4[0], temps[0], false, false, true);
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