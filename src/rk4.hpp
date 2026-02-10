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
  const auto* k1ptr = k1.data_ptr<cuda::std::complex<double>>();
  const auto* k2ptr = k2.data_ptr<cuda::std::complex<double>>();
  const auto* k3ptr = k3.data_ptr<cuda::std::complex<double>>();
  const auto* k4ptr = k4.data_ptr<cuda::std::complex<double>>();
  auto* signalptr = signal.data_ptr<cuda::std::complex<double>>();

  auto numel = signal.numel();
  RK4StepLaunch(numel, k1ptr, k2ptr, k3ptr, k4ptr, signalptr, stream);
}

inline
void RK4Incr(bool halve, at::Tensor& newSignal, const at::Tensor& k,
             const at::Tensor& signal, cudaStream_t stream) {
  auto* newSignalptr = newSignal.data_ptr<cuda::std::complex<double>>();
  const auto* kptr = k.data_ptr<cuda::std::complex<double>>();
  const auto* signalptr = signal.data_ptr<cuda::std::complex<double>>();

  auto numel = signal.numel();
  RK4IncrLaunch(numel, halve, newSignalptr, kptr, signalptr, stream);
}

#endif //NLMKERNELS
