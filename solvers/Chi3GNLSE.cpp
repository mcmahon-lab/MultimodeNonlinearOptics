#ifndef CHI3GNLSE
#define CHI3GNLSE

#include "_FullyNonlinearMedium.hpp"

class Chi3GNLSE : public _FullyNonlinearMedium {
  NLM(Chi3GNLSE, 1)
public:
  Chi3GNLSE(double relativeLength, double nlLength, double selfSteepLength, double fr, double fb, double tau1, double tau2, double tau3,
            double beta2, double beta3=0, double rayleighLength=std::numeric_limits<double>::infinity(),
            double tMax=10, uint tPrecision=512, uint zPrecision=100, IntensityProfile intensityProfile=IntensityProfile{});

  const Arraycd& getRamanResponse() {return ramanResponse;};

private:
  Arraycd ramanResponse;
};


Chi3GNLSE::Chi3GNLSE(double relativeLength, double nlLength, double selfSteepLength, double fr, double fb, double tau1, double tau2, double tau3,
                     double beta2, double beta3, double rayleighLength, double tMax, uint tPrecision, uint zPrecision, IntensityProfile intensityProfile) :
  _FullyNonlinearMedium(_nSignalModes, false, 0, relativeLength, {nlLength, selfSteepLength}, {beta2}, {0}, {beta3}, {},
                        rayleighLength, tMax, tPrecision, zPrecision, intensityProfile)
{
  // Precompute Raman response for the convolution
  double coeff1 = fr * (1. - fb) * (tau1 / (tau2*tau2) + 1. / tau1);
  double coeff2 = fr * fb / (tau3*tau3);
  Arraycd ramanResponseTime = Arrayd::Zero(_nFreqs);
  // As given by Agrawal Nonlinear Fiber Optics
  ramanResponseTime.leftCols(_nFreqs/2) = // Only assign values for t >= 0, by causality R(t < 0) = 0
      coeff1 * (-_tau.leftCols(_nFreqs/2) / tau2).exp() * (_tau.leftCols(_nFreqs/2) / tau1).sin() // Delayed Raman response
    + coeff2 * (-_tau.leftCols(_nFreqs/2) / tau3).exp() * (2 * tau3 - _tau.leftCols(_nFreqs/2)); // Boson peak
  ramanResponseTime(0) += 1. - fr; // delta function
  ramanResponse = Arraycd(_nFreqs);
  FFT(ramanResponse, ramanResponseTime);
}


void Chi3GNLSE::DiffEq(uint i, uint iPrevSig, std::vector<Arraycd>& k1, std::vector<Arraycd>& k2, std::vector<Arraycd>& k3,
                       std::vector<Arraycd>& k4, const std::vector<Array2Dcd>& signal) {
  const auto& prev = signal[0].row(iPrevSig);

  const double relIntPrv = relativeIntensity(i- 1);
  const double relIntInt = relativeIntensity(i-.5);
  const double relIntCur = relativeIntensity(i);

  // Need a temporary, thread safe storage array. Hack (Caution) to avoid allocating each time:
  if (i == 1) {k1.emplace_back(_nFreqs);}
  Arraycd& temp = k1[1];

  // Convolution between Raman response function and |A|^2, ie (R * |A|^2)
  k1[0] = relIntPrv * prev.abs2();
  FFT(temp, k1[0]);
  temp *= ramanResponse;
  IFFT(k1[0], temp);
  // Multiply convolution by A, ie A (R * |A|^2)
  k1[0] *= prev;
  // take derivative, ie i (gamma + i gamma' d/dt) (A (R * |A|^2)) = i IF[(gamma + omega gamma') F[A (R * |A|^2)]]
  FFT(temp, k1[0]);
  temp *= _nlStep[0] + _nlStep[1] * _omega;
  IFFT(k1[0], temp);

  // Repeat for k2
  k2[0] = relIntInt * (prev + 0.5 * k1[0]).abs2();
  FFT(temp, k2[0]);
  temp *= ramanResponse;
  IFFT(k2[0], temp);
  k2[0] *= prev + 0.5 * k1[0];
  FFT(temp, k2[0]);
  temp *= _nlStep[0] + _nlStep[1] * _omega;
  IFFT(k2[0], temp);

  // Repeat for k3
  k3[0] = relIntInt * (prev + 0.5 * k2[0]).abs2();
  FFT(temp, k3[0]);
  temp *= ramanResponse;
  IFFT(k3[0], temp);
  k3[0] *= prev + 0.5 * k2[0];
  FFT(temp, k3[0]);
  temp *= _nlStep[0] + _nlStep[1] * _omega;
  IFFT(k3[0], temp);

  // Repeat for k4
  k4[0] = relIntCur * (prev + k3[0]).abs2();
  FFT(temp, k4[0]);
  temp *= ramanResponse;
  IFFT(k4[0], temp);
  k4[0] *= prev + k3[0];
  FFT(temp, k4[0]);
  temp *= _nlStep[0] + _nlStep[1] * _omega;
  IFFT(k4[0], temp);
}

#endif //CHI3GNLSE

#ifdef NLMMODULE
py::class_<Chi3GNLSE, _FullyNonlinearMedium> Chi3GNLSE(m, "Chi3GNLSE", "Fully nonlinear general nonlinear Schrodinger equation");
Chi3GNLSE.def(
    py::init<double, double, double, double, double, double, double, double, double, double, double, double,
             uint, uint, _NonlinearMedium::IntensityProfile>(),
    "relativeLength"_a, "nlLength"_a, "selfSteepLength"_a,  "fr"_a, "fb"_a, "tau1"_a, "tau2"_a, "tau3"_a,
    "beta2"_a, "beta3"_a = 0, "rayleighLength"_a = infinity, "tMax"_a = 10, "tPrecision"_a = 512, "zPrecision"_a = 100,
    "intensityProfile"_a = _NonlinearMedium::IntensityProfile{});
Chi3GNLSE.def_property_readonly("ramanResponse", &Chi3GNLSE::getRamanResponse, py::return_value_policy::reference,
                                "Read-only array of the Raman response function in the frequency domain.");
#endif