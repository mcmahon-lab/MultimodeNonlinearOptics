#ifndef NONLINEARMEDIUM_CUSTOMEIGENFFT_H
#define NONLINEARMEDIUM_CUSTOMEIGENFFT_H

// This file is a modified version of Eigen's unsupported/FFT module

#ifndef EIGEN_FFT_H
#define EIGEN_FFT_H

#include <complex>
#include <vector>
#include <map>
#include <eigen3/Eigen/Core>


#include "Eigen/src/Core/util/DisableStupidWarnings.h"

#ifdef EIGEN_FFTW_DEFAULT
// FFTW: faster, GPL -- incompatible with Eigen in LGPL form, bigger code size
#  include <fftw3.h>
#  include "unsupported/Eigen/src/FFT/ei_fftw_impl.h"
namespace Eigen {
  //template <typename T> typedef struct internal::fftw_impl  default_fft_impl; this does not work
  template <typename T> struct default_fft_impl : public internal::fftw_impl<T> {};
}
#elif defined EIGEN_MKL_DEFAULT
// TODO
// intel Math Kernel Library: fastest, commercial -- may be incompatible with Eigen in GPL form
#  include "unsupported/Eigen/src/FFT/ei_imklfft_impl.h"
   namespace Eigen {
     template <typename T> struct default_fft_impl : public internal::imklfft_impl {};
   }
#else
// internal::kissfft_impl:  small, free, reasonably efficient default, derived from kissfft
//
# include "unsupported/Eigen/src/FFT/ei_kissfft_impl.h"
  namespace Eigen {
     template <typename T>
       struct default_fft_impl : public internal::kissfft_impl<T> {};
  }
#endif

namespace Eigen {

  template <typename T_Scalar, typename T_Impl=default_fft_impl<T_Scalar>>
  class FFT {
  public:
    typedef T_Impl impl_type;
    typedef DenseIndex Index;
    typedef typename impl_type::Scalar Scalar;
    typedef typename impl_type::Complex Complex;

    FFT(const impl_type& impl=impl_type()) : m_impl(impl) {}

    inline
    void fwd(Complex* dst, const Scalar* src, Index nfft) {
      m_impl.fwd(dst, src, static_cast<int>(nfft));
    }

    inline
    void fwd(Complex* dst, const Complex* src, Index nfft) {
      m_impl.fwd(dst, src, static_cast<int>(nfft));
    }

    template<typename InputDerived, typename ComplexDerived>
    inline
    void fwd(DenseBase<ComplexDerived>& dst, const DenseBase<InputDerived>& src, Index nfft) {
      typedef typename ComplexDerived::Scalar dst_type;
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(InputDerived)
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(ComplexDerived)
      EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ComplexDerived, InputDerived) // size at compile-time
      EIGEN_STATIC_ASSERT((internal::is_same<dst_type, Complex>::value),
                          YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_DenseBase_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
      EIGEN_STATIC_ASSERT(int(InputDerived::Flags)&int(ComplexDerived::Flags)&DirectAccessBit,
                          THIS_METHOD_IS_ONLY_FOR_EXPRESSIONS_WITH_DIRECT_MEMORY_ACCESS_SUCH_AS_MAP_OR_PLAIN_MATRICES)

      fwd(&dst[0], &src[0], nfft);
    }

    template<typename InputDerived, typename ComplexDerived>
    inline
    void fwd(DenseBase<ComplexDerived>& dst, const DenseBase<InputDerived>& src, Index dstInd, Index srcInd, Index nfft) {
      typedef typename ComplexDerived::Scalar dst_type;
      EIGEN_STATIC_ASSERT((internal::is_same<dst_type, Complex>::value),
                          YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_DenseBase_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
      EIGEN_STATIC_ASSERT(int(InputDerived::Flags)&int(ComplexDerived::Flags)&DirectAccessBit,
                          THIS_METHOD_IS_ONLY_FOR_EXPRESSIONS_WITH_DIRECT_MEMORY_ACCESS_SUCH_AS_MAP_OR_PLAIN_MATRICES)

      fwd(&dst(dstInd, 0), &src(srcInd, 0), nfft);
    }

    inline
    void inv(Complex* dst, const Complex* src, Index nfft) {
      m_impl.inv(dst, src, static_cast<int>(nfft));
      scale(dst, Scalar(1. / nfft), nfft);
    }

    inline
    void inv(Scalar* dst, const Complex* src, Index nfft) {
      m_impl.inv(dst, src, static_cast<int>(nfft));
      scale(dst, Scalar(1. / nfft), nfft);
    }

    template<typename OutputDerived, typename ComplexDerived>
    inline
    void inv(DenseBase<OutputDerived>& dst, const DenseBase<ComplexDerived>& src, Index nfft) {
      typedef typename ComplexDerived::Scalar src_type;
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(OutputDerived)
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(ComplexDerived)
      EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(ComplexDerived, OutputDerived) // size at compile-time
      EIGEN_STATIC_ASSERT((internal::is_same<src_type, Complex>::value),
                          YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_DenseBase_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
      EIGEN_STATIC_ASSERT(int(OutputDerived::Flags)&int(ComplexDerived::Flags)&DirectAccessBit,
                          THIS_METHOD_IS_ONLY_FOR_EXPRESSIONS_WITH_DIRECT_MEMORY_ACCESS_SUCH_AS_MAP_OR_PLAIN_MATRICES)

      inv(&dst[0], &src[0], nfft);
    }

    template<typename OutputDerived, typename ComplexDerived>
    inline
    void inv(DenseBase<OutputDerived>& dst, const DenseBase<ComplexDerived>& src, Index dstInd, Index srcInd, Index nfft) {
      typedef typename ComplexDerived::Scalar src_type;
      EIGEN_STATIC_ASSERT((internal::is_same<src_type, Complex>::value),
                          YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_DenseBase_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
      EIGEN_STATIC_ASSERT(int(OutputDerived::Flags)&int(ComplexDerived::Flags)&DirectAccessBit,
                          THIS_METHOD_IS_ONLY_FOR_EXPRESSIONS_WITH_DIRECT_MEMORY_ACCESS_SUCH_AS_MAP_OR_PLAIN_MATRICES)

      inv(&dst(dstInd, 0), &src(srcInd, 0), nfft);
    }

  private:

    template <typename T_Data>
    inline
    void scale(T_Data* x, Scalar s, Index nx) {
      for (int k = 0; k < nx; ++k)
        *x++ *= s;
    }

    impl_type m_impl;
  };
}

#include "Eigen/src/Core/util/ReenableStupidWarnings.h"
#endif
#endif //NONLINEARMEDIUM_CUSTOMEIGENFFT_H
