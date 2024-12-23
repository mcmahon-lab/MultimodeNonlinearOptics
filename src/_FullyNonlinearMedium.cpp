#include "_FullyNonlinearMedium.hpp"

_FullyNonlinearMedium::_FullyNonlinearMedium(uint nSignalModes, bool canBePoled, uint nFieldModes, double relativeLength,
                                             std::initializer_list<double> nlLength, std::initializer_list<double> beta2s,
                                             std::initializer_list<double> beta1s, std::initializer_list<double> beta3s,
                                             std::initializer_list<double> diffBeta0, double rayleighLength, double tMax,
                                             uint tPrecision, uint zPrecision, IntensityProfile intensityProfile,
                                             const Eigen::Ref<const Arrayd>& poling) :
  _NonlinearMedium(nSignalModes, nFieldModes)
{
  setLengths(relativeLength, nlLength, zPrecision, rayleighLength, {0}, beta2s, {0}, beta1s, {0}, beta3s);
  _dzp = _nZStepsP = 0;
  resetGrids(tPrecision, tMax);
  _intensityProfile = _rayleighLength != std::numeric_limits<double>::infinity() ?
                      intensityProfile : IntensityProfile::Constant;
  _FullyNonlinearMedium::setDispersion(beta2s, beta1s, beta3s, diffBeta0);
  if (canBePoled)
    setPoling(poling);
}


void _FullyNonlinearMedium::setDispersion(const std::vector<double>& beta2s, const std::vector<double>& beta1s,
                                          const std::vector<double>& beta3s, std::initializer_list<double> diffBeta0) {

  // signal phase mis-match
  _diffBeta0 = diffBeta0;

  // dispersion profile
  _dispersionSign.resize(_nSignalModes);
  for (uint m = 0; m < _nSignalModes; m++)
    _dispersionSign[m] = _omega * (beta1s[m] + _omega * (0.5 * beta2s[m] + _omega * beta3s[m] / 6));

  // incremental phases for each simulation step
  _dispStepSign.resize(_nSignalModes);
  for (uint m = 0; m < _nSignalModes; m++)
    _dispStepSign[m] = (1._I * _dispersionSign[m] * _dz).exp();
}
