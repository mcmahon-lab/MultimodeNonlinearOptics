#include <cuda_runtime.h>

__global__
void Chi3SPM3DKernel(int numel, double intensity,
                     cuda::std::complex<double>* k, const cuda::std::complex<double>* signal) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    k[idx] = (intensity * (signal[idx].real() * signal[idx].real() + signal[idx].imag() * signal[idx].imag())) * signal[idx];
  }
}

inline
void Chi3SPM3DLaunch(int numel, double intensity, void* k, const void* signal, cudaStream_t stream) {
  static constexpr int BLOCKSIZE = 512; // multiple of 32 and smaller than 1024
  Chi3SPM3DKernel<<<(numel+(BLOCKSIZE-1))/BLOCKSIZE, BLOCKSIZE, 0, stream>>>(numel, intensity,
                                                                             reinterpret_cast<cuda::std::complex<double>*>(k), reinterpret_cast<const cuda::std::complex<double>*>(signal));
}
