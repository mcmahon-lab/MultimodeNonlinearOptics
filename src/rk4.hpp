#ifndef NLMKERNELS
#define NLMKERNELS

#include <ATen/ATen.h>
#include <cuda/std/complex>

void RK4StepLaunch(int numel,
                   const cuda::std::complex<double>* k1, const cuda::std::complex<double>* k2,
                   const cuda::std::complex<double>* k3, const cuda::std::complex<double>* k4,
                   const cuda::std::complex<double>* signal, cudaStream_t stream);

void RK4IncrLaunch(int numel, bool halve, cuda::std::complex<double>* newSignal,
                   const cuda::std::complex<double>* k, const cuda::std::complex<double>* signal, cudaStream_t stream);

inline
void RK4Step(const at::Tensor& k1, const at::Tensor& k2,
             const at::Tensor& k3, const at::Tensor& k4,
             at::Tensor& signal, cudaStream_t stream) {
  const auto* k1ptr = reinterpret_cast<cuda::std::complex<double>*>(k1.data_ptr());
  const auto* k2ptr = reinterpret_cast<cuda::std::complex<double>*>(k2.data_ptr());
  const auto* k3ptr = reinterpret_cast<cuda::std::complex<double>*>(k3.data_ptr());
  const auto* k4ptr = reinterpret_cast<cuda::std::complex<double>*>(k4.data_ptr());
  auto* signalptr = reinterpret_cast<cuda::std::complex<double>*>(signal.data_ptr());

  auto numel = signal.numel();
  RK4StepLaunch(numel, k1ptr, k2ptr, k3ptr, k4ptr, signalptr, stream);
}

inline
void RK4Incr(bool halve, at::Tensor& newSignal, const at::Tensor& k,
             const at::Tensor& signal, cudaStream_t stream) {
  auto* newSignalptr = reinterpret_cast<cuda::std::complex<double>*>(newSignal.data_ptr());
  const auto* kptr = reinterpret_cast<cuda::std::complex<double>*>(k.data_ptr());
  const auto* signalptr = reinterpret_cast<cuda::std::complex<double>*>(signal.data_ptr());

  auto numel = signal.numel();
  RK4IncrLaunch(numel, halve, newSignalptr, kptr, signalptr, stream);
}

#endif //NLMKERNELS
