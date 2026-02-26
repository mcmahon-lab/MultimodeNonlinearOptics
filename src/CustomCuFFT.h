#ifndef NLMCUFFT
#define NLMCUFFT

#include <cstdint>
#include <map>
#include <vector>
#include <type_traits>

#include <ATen/ATen.h>
#include <cuda/std/complex>
#include <cufft.h>
#include <cuda_runtime.h> // cudaStream_t

// note not multi CUDA stream (thread) safe
// for multiple streams (eg across different threads; one stream per thread)
// we would need to allocate resources for each concurrent stream with cufftSetWorkArea()
// and each plan would need to be properly handled wrt cufftSetStream()

// cuFFT only has float and double, and only implementation difference is in the cufftType and execution
template<typename T,
         typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
struct cufftPlan {
  typedef T Real;
  typedef cuda::std::complex<T> Complex;

  static constexpr cufftType_t c2ctype() {
    if constexpr (std::is_same_v<T, float>) {return CUFFT_C2C;}
    if constexpr (std::is_same_v<T, double>) {return CUFFT_Z2Z;}
  }
  static constexpr cufftType_t c2rtype() {
    if constexpr (std::is_same_v<T, float>) {return CUFFT_C2R;}
    if constexpr (std::is_same_v<T, double>) {return CUFFT_Z2D;}
  }
  static constexpr cufftType_t r2ctype() {
    if constexpr (std::is_same_v<T, float>) {return CUFFT_R2C;}
    if constexpr (std::is_same_v<T, double>) {return CUFFT_D2Z;}
  }


  cufftHandle _plan;

  ~cufftPlan() {if (_plan) cufftDestroy(_plan);}

  cufftPlan(int n0, int n1, int n2, bool isC2C, bool inv) {
    cufftCreate(&_plan);

    cufftType_t transformType;
    if (isC2C) transformType = c2ctype();
    else {
      if (inv) transformType = c2rtype();
      else     transformType = r2ctype();
    }

    if (n2)      cufftPlan3d(&_plan, n2, n1, n0, transformType);
    else if (n1) cufftPlan2d(&_plan, n1, n0, transformType);
    else         cufftPlan1d(&_plan, n0, transformType, 1);
  }

  cufftPlan(int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2, bool isC2C, bool inv) {
    int rank = (static_cast<int>(doDim0) + doDim1) + doDim2;
    int howmany = (doDim0? 1 : n0) * (doDim1? 1 : n1) * (doDim2? 1 : n2);
    int stride = doDim2? 1 : n2 * (doDim1? 1 : n1 * (doDim0? 1 : n0));
    int distance = doDim2? n2 * (doDim1? n1 * (doDim0? n0: 1): 1) : 1;

    std::vector<int> dimensions(rank);
    int i = 0;
    if (doDim2) dimensions[i++] = n2;
    if (doDim1) dimensions[i++] = n1;
    if (doDim0) dimensions[i++] = n0;

    cufftType_t transformType;
    if (isC2C) transformType = c2ctype();
    else {
      if (inv) transformType = c2rtype();
      else     transformType = r2ctype();
    }

    cufftPlanMany(&_plan, rank, dimensions.data(),
                  NULL, stride, distance,
                  NULL, stride, distance,
                  transformType, howmany);
  }


  template<typename U, typename P>
  static constexpr P* ptrCast(const U* ptr) {
    return reinterpret_cast<P*>(const_cast<U*>(ptr));
  }

  inline void execute(Complex* dst, const Complex* src, bool fwd) const {
    if constexpr (std::is_same_v<T, float>)
      cufftExecC2C(_plan,
                   ptrCast<Complex, cufftComplex>(src),
                   ptrCast<Complex, cufftComplex>(dst),
                   fwd? CUFFT_FORWARD : CUFFT_INVERSE);
    else
      cufftExecZ2Z(_plan,
                   ptrCast<Complex, cufftDoubleComplex>(src),
                   ptrCast<Complex, cufftDoubleComplex>(dst),
                   fwd? CUFFT_FORWARD : CUFFT_INVERSE);
  }
  inline void execute(Complex* dst, const Real* src)  const {
    if constexpr (std::is_same_v<T, float>)
      cufftExecR2C(_plan,
                   ptrCast<Real, cufftReal>(src),
                   ptrCast<Complex, cufftComplex>(dst),
                   CUFFT_FORWARD);
    else
      cufftExecD2Z(_plan,
                   ptrCast<Real, cufftDoubleReal>(src),
                   ptrCast<Complex, cufftDoubleComplex>(dst),
                   CUFFT_FORWARD);
  }
  inline void execute(Real* dst, const Complex* src) const {
    if constexpr (std::is_same_v<T, float>)
      cufftExecC2R(_plan,
                   ptrCast<Complex, cufftComplex>(src),
                   ptrCast<Real, cufftReal>(dst),
                   CUFFT_INVERSE);
    else
      cufftExecZ2D(_plan,
                   ptrCast<Complex, cufftDoubleComplex>(src),
                   ptrCast<Real, cufftDoubleReal>(dst),
                   CUFFT_INVERSE);
  }

  inline void setStream(cudaStream_t stream) {
    cufftSetStream(_plan, stream);
  }
};


template<typename T>
struct cuFFTimpl {
  typedef T Real;
  typedef cuda::std::complex<T> Complex;

  inline void clear() {
    _plans.clear();
  }

  inline void setStream(cudaStream_t stream) {
    for (auto& [key, plan] : _plans) {
      plan.setStream(stream);
    }
  }

  /// real-to-complex forward FFT
  inline void fwd(Complex* dst, const Real* src, int n0, int n1=0, int n2=0) {
    getPlan(n0, n1, n2, true, false).execute(dst, src);
  }

  /// complex-to-complex forward FFT
  inline void fwd(Complex* dst, const Complex* src, int n0, int n1=0, int n2=0) {
    getPlan(n0, n1, n2, true, false).execute(dst, src, true);
  }

  /// Forward FFT over a subset of dimensions (real-to-complex)
  inline void fwd(Complex* dst, const Real* src, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2) {
    getPlanMany(n0, n1, n2, doDim0, doDim1, doDim2, false, false).execute(dst, src);
  }

  /// Forward FFT over a subset of dimensions
  inline void fwd(Complex* dst, const Complex* src, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2) {
    getPlanMany(n0, n1, n2, doDim0, doDim1, doDim2, true, false).execute(dst, src, true);
  }

  /// complex to real inverse FFT
  inline void inv(Real* dst, const Complex* src, int n0, int n1=0, int n2=0) {
    getPlan(n0, n1, n2, false, true).execute(dst, src);
  }

  /// complex-to-complex inverse FFT
  inline void inv(Complex* dst, const Complex* src, int n0, int n1=0, int n2=0) {
    getPlan(n0, n1, n2, true, true).execute(dst, src, false);
  }

  /// Inverse FFT over a subset of dimensions (complex-to-real)
  inline void inv(Real* dst, const Complex* src, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2) {
    getPlanMany(n0, n1, n2, doDim0, doDim1, doDim2, false, true).execute(dst, src);
  }

  /// Inverse FFT over a subset of dimensions
  inline void inv(Complex* dst, const Complex* src, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2) {
    getPlanMany(n0, n1, n2, doDim0, doDim1, doDim2, true, true).execute(dst, src, false);
  }

protected:
  typedef cufftPlan<T> Plan;

  std::map<std::pair<uint64_t, uint64_t>, Plan> _plans;

  inline Plan& getPlan(int n0, int n1, int n2, bool isC2C, bool inverse) {
    uint64_t keyValue1 = static_cast<uint64_t>(isC2C) << 1;
    if (!isC2C) keyValue1 += inverse;
    keyValue1 = (keyValue1 << 32) + n0;
    uint64_t keyValue2 = (static_cast<uint64_t>(n1) << 32) + n2;

    return (*_plans.try_emplace({keyValue1, keyValue2}, n0, n1, n2, isC2C, inverse).first).second;
  }

  inline Plan& getPlanMany(int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2, bool isC2C, bool inverse) {
    uint64_t keyValue1 = ((((((static_cast<uint64_t>(doDim0) << 1) + doDim1) << 1) + doDim2) << 1) + isC2C) << 1;
    if (!isC2C) keyValue1 += inverse;
    keyValue1 = (keyValue1 << 32) + n0;
    uint64_t keyValue2 = (static_cast<uint64_t>(n1) << 32) + n2;

    return (*_plans.try_emplace({keyValue1, keyValue2}, n0, n1, n2, doDim0, doDim1, doDim2, isC2C, inverse).first).second;
  }
};


template <typename T>
class FFTGPU {
public:
  FFTGPU() = default;

  inline void fwd(at::Tensor& dst, const at::Tensor& src, int n0) {
    const auto* srcPtr = reinterpret_cast<const cuda::std::complex<T>*>(src.data_ptr());
    auto* dstPtr = reinterpret_cast<cuda::std::complex<T>*>(dst.data_ptr());
    impl.fwd(dstPtr, srcPtr, n0);
  };

  inline void inv(at::Tensor& dst, const at::Tensor& src, int n0) {
    const auto* srcPtr = reinterpret_cast<const cuda::std::complex<T>*>(src.data_ptr());
    auto* dstPtr = reinterpret_cast<cuda::std::complex<T>*>(dst.data_ptr());
    impl.inv(dstPtr, srcPtr, n0);
  };

  inline void fwd2(at::Tensor& dst, const at::Tensor& src, int n1, int n0) {
    const auto* srcPtr = reinterpret_cast<const cuda::std::complex<T>*>(src.data_ptr());
    auto* dstPtr = reinterpret_cast<cuda::std::complex<T>*>(dst.data_ptr());
    impl.fwd(dstPtr, srcPtr, n0, n1);
  };

  inline void inv2(at::Tensor& dst, const at::Tensor& src, int n1, int n0) {
    const auto* srcPtr = reinterpret_cast<const cuda::std::complex<T>*>(src.data_ptr());
    auto* dstPtr = reinterpret_cast<cuda::std::complex<T>*>(dst.data_ptr());
    impl.inv(dstPtr, srcPtr, n0, n1);
  };

  inline void fwd3(at::Tensor& dst, const at::Tensor& src, int n2, int n1, int n0) {
    const auto* srcPtr = reinterpret_cast<const cuda::std::complex<T>*>(src.data_ptr());
    auto* dstPtr = reinterpret_cast<cuda::std::complex<T>*>(dst.data_ptr());
    impl.fwd(dstPtr, srcPtr, n0, n1, n2);
  };

  inline void inv3(at::Tensor& dst, const at::Tensor& src, int n2, int n1, int n0) {
    const auto* srcPtr = reinterpret_cast<const cuda::std::complex<T>*>(src.data_ptr());
    auto* dstPtr = reinterpret_cast<cuda::std::complex<T>*>(dst.data_ptr());
    impl.inv(dstPtr, srcPtr, n0, n1, n2);
  };

  inline void fwdPartial(at::Tensor& dst, const at::Tensor& src,
                         int n2, int n1, int n0,
                         bool doDim0, bool doDim1, bool doDim2) {
    const auto* srcPtr = reinterpret_cast<const cuda::std::complex<T>*>(src.data_ptr());
    auto* dstPtr = reinterpret_cast<cuda::std::complex<T>*>(dst.data_ptr());
    impl.fwd(dstPtr, srcPtr, n0, n1, n2, doDim0, doDim1, doDim2);
  };

  inline void invPartial(at::Tensor& dst, const at::Tensor& src,
                         int n2, int n1, int n0,
                         bool doDim0, bool doDim1, bool doDim2) {
    const auto* srcPtr = reinterpret_cast<const cuda::std::complex<T>*>(src.data_ptr());
    auto* dstPtr = reinterpret_cast<cuda::std::complex<T>*>(dst.data_ptr());
    impl.fwd(dstPtr, srcPtr, n0, n1, n2, doDim0, doDim1, doDim2);
  };

  inline void setStream(cudaStream_t stream) {
    impl.setStream(stream);
  };

private:
  cuFFTimpl<T> impl;
};

#endif //NLMCUFFT
