#include "rk4.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

__global__
void RK4StepKernel(int numel,
                   const cuda::std::complex<double>* k1, const cuda::std::complex<double>* k2,
                   const cuda::std::complex<double>* k3, const cuda::std::complex<double>* k4,
                   cuda::std::complex<double>* signal) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    signal[idx] += (k1[idx] + 2. * (k2[idx] + k3[idx]) + k4[idx]) / 6.;
  }
}

inline
void RK4StepLaunch(int numel,
                   const cuda::std::complex<double>* k1, const cuda::std::complex<double>* k2,
                   const cuda::std::complex<double>* k3, const cuda::std::complex<double>* k4,
                   cuda::std::complex<double>* signal, cudaStream_t stream) {
  static constexpr int BLOCKSIZE = 512; // multiple of 32 and smaller than 1024
  RK4StepKernel<<<(numel+(BLOCKSIZE-1))/BLOCKSIZE, BLOCKSIZE, 0, stream>>>(numel, k1, k2, k3, k4, signal);
}

__global__
void RK4IncrKernel(int numel, bool halve, cuda::std::complex<double>* newSignal,
                   const cuda::std::complex<double>* k, const cuda::std::complex<double>* signal) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    if (halve)
      newSignal[idx] = signal[idx] + 0.5 * k[idx];
    else
      newSignal[idx] = signal[idx] + k[idx];
  }
}

inline
void RK4IncrLaunch(int numel, bool halve, cuda::std::complex<double>* newSignal,
                   const cuda::std::complex<double>* k, const cuda::std::complex<double>* signal,
                   cudaStream_t stream) {
  static constexpr int BLOCKSIZE = 512; // multiple of 32 and smaller than 1024
  RK4IncrKernel<<<(numel+(BLOCKSIZE-1))/BLOCKSIZE, BLOCKSIZE, 0, stream>>>(numel, halve, newSignal, k, signal);
}
