#ifndef NONLINEARMEDIUM_EIGEN_FFTW_H
#define NONLINEARMEDIUM_EIGEN_FFTW_H

#include <fftw3.h>
#include <complex>
#include <cstdint>
#include <map>
#include <vector>

namespace Eigen {
  namespace internal {

    // FFTW uses non-const arguments, so we must use const_cast calls for all the args it uses
    // This should be safe as long as
    // 1. we use FFTW_ESTIMATE for all our planning
    //    see the FFTW docs section 4.3.2 "Planner Flags"
    // 2. fftw_complex is compatible with std::complex
    //    This assumes std::complex<T> layout is array of size 2 with real, imag

    template<typename T> struct fftwPlan {};

    template<>
    struct fftwPlan<float> {
      typedef float scalar_type;
      typedef fftwf_complex complex_type;

      fftwf_plan _plan;

      ~fftwPlan() {if (_plan) fftwf_destroy_plan(_plan);}

      fftwPlan(const void* _src, const void* _dst, bool inv, int nfft, int n1, int n2, bool isC2C) {
        if (isC2C) {
          complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
          complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
          if (inv) {
            if (n2)      _plan = fftwf_plan_dft_3d(nfft, n1, n2, src, dst, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftwf_plan_dft_2d(nfft, n1, src, dst, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftwf_plan_dft_1d(nfft, src, dst, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          } else {
            if (n2)      _plan = fftwf_plan_dft_3d(nfft, n1, n2, src, dst, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftwf_plan_dft_2d(nfft, n1, src, dst, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftwf_plan_dft_1d(nfft, src, dst, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
        } else {
          if (inv) {
            complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
            scalar_type* dst = const_cast<scalar_type*>(static_cast<const scalar_type*>(_dst));
            if (n2)      _plan = fftwf_plan_dft_c2r_3d(nfft, n1, n2, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftwf_plan_dft_c2r_2d(nfft, n1, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftwf_plan_dft_c2r_1d(nfft, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          } else {
            scalar_type* src = const_cast<scalar_type*>(static_cast<const scalar_type*>(_src));
            complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
            if (n2)      _plan = fftwf_plan_dft_r2c_3d(nfft, n1, n2, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftwf_plan_dft_r2c_2d(nfft, n1, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftwf_plan_dft_r2c_1d(nfft, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
        }
      }

      fftwPlan(const void* _src, const void* _dst, bool inv, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2, bool isC2C) {
        int rank = (static_cast<int>(doDim0) + doDim1) + doDim2;
        int howmany = (doDim0? 1 : n0) * (doDim1? 1 : n1) * (doDim2? 1 : n2);
        int stride = doDim2? 1 : n2 * (doDim1? 1 : n1 * (doDim0? 1 : n0));
        int distance = doDim2? n2 * (doDim1? n1 * (doDim0? n0: 1): 1) : 1;

        std::vector<int> dimensions(rank);
        int i = 0;
        if (doDim2) dimensions[i++] = n2;
        if (doDim1) dimensions[i++] = n1;
        if (doDim0) dimensions[i++] = n0;

        if (isC2C) {
          complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
          complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
          _plan = fftwf_plan_many_dft(rank, dimensions.data(), howmany,
                                      src, NULL, stride, distance,
                                      dst, NULL, stride, distance,
                                      (inv? FFTW_BACKWARD : FFTW_FORWARD), FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
        } else {
          if (inv) {
            complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
            scalar_type* dst = const_cast<scalar_type*>(static_cast<const scalar_type*>(_dst));
            _plan = fftwf_plan_many_dft_c2r(rank, dimensions.data(), howmany,
                                            src, NULL, stride, distance,
                                            dst, NULL, stride, distance,
                                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
          else {
            scalar_type* src = const_cast<scalar_type*>(static_cast<const scalar_type*>(_src));
            complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
            _plan = fftwf_plan_many_dft_r2c(rank, dimensions.data(), howmany,
                                            src, NULL, stride, distance,
                                            dst, NULL, stride, distance,
                                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
        }
      }

      inline void execute(complex_type* dst, complex_type* src) const {fftwf_execute_dft(_plan, src, dst);}
      inline void execute(complex_type* dst, scalar_type* src)  const {fftwf_execute_dft_r2c(_plan, src, dst);}
      inline void execute(scalar_type* dst,  complex_type* src) const {fftwf_execute_dft_c2r(_plan, src, dst);}
    };

    template<>
    struct fftwPlan<double> {
      typedef double scalar_type;
      typedef fftw_complex complex_type;

      fftw_plan _plan;

      ~fftwPlan() {if (_plan) fftw_destroy_plan(_plan);}

      fftwPlan(const void* _src, const void* _dst, bool inv, int nfft, int n1, int n2, bool isC2C) {
        if (isC2C) {
          complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
          complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
          if (inv) {
            if (n2)      _plan = fftw_plan_dft_3d(nfft, n1, n2, src, dst, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftw_plan_dft_2d(nfft, n1, src, dst, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftw_plan_dft_1d(nfft, src, dst, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          } else {
            if (n2)      _plan = fftw_plan_dft_3d(nfft, n1, n2, src, dst, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftw_plan_dft_2d(nfft, n1, src, dst, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftw_plan_dft_1d(nfft, src, dst, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
        } else {
          if (inv) {
            complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
            scalar_type* dst = const_cast<scalar_type*>(static_cast<const scalar_type*>(_dst));
            if (n2)      _plan = fftw_plan_dft_c2r_3d(nfft, n1, n2, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftw_plan_dft_c2r_2d(nfft, n1, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftw_plan_dft_c2r_1d(nfft, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          } else {
            scalar_type* src = const_cast<scalar_type*>(static_cast<const scalar_type*>(_src));
            complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
            if (n2)      _plan = fftw_plan_dft_r2c_3d(nfft, n1, n2, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftw_plan_dft_r2c_2d(nfft, n1, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftw_plan_dft_r2c_1d(nfft, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
        }
      }

      fftwPlan(const void* _src, const void* _dst, bool inv, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2, bool isC2C) {
        int rank = (static_cast<int>(doDim0) + doDim1) + doDim2;
        int howmany = (doDim0? 1 : n0) * (doDim1? 1 : n1) * (doDim2? 1 : n2);
        int stride = doDim2? 1 : n2 * (doDim1? 1 : n1 * (doDim0? 1 : n0));
        int distance = doDim2? n2 * (doDim1? n1 * (doDim0? n0: 1): 1) : 1;

        std::vector<int> dimensions(rank);
        int i = 0;
        if (doDim2) dimensions[i++] = n2;
        if (doDim1) dimensions[i++] = n1;
        if (doDim0) dimensions[i++] = n0;

        if (isC2C) {
          complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
          complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
          _plan = fftw_plan_many_dft(rank, dimensions.data(), howmany,
                                     src, NULL, stride, distance,
                                     dst, NULL, stride, distance,
                                     (inv? FFTW_BACKWARD : FFTW_FORWARD), FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
        } else {
          if (inv) {
            complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
            scalar_type* dst = const_cast<scalar_type*>(static_cast<const scalar_type*>(_dst));
            _plan = fftw_plan_many_dft_c2r(rank, dimensions.data(), howmany,
                                           src, NULL, stride, distance,
                                           dst, NULL, stride, distance,
                                           FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
          else {
            scalar_type* src = const_cast<scalar_type*>(static_cast<const scalar_type*>(_src));
            complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
            _plan = fftw_plan_many_dft_r2c(rank, dimensions.data(), howmany,
                                           src, NULL, stride, distance,
                                           dst, NULL, stride, distance,
                                           FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
        }
      }

      inline void execute(complex_type* dst, complex_type* src) const {fftw_execute_dft(_plan, src, dst);}
      inline void execute(complex_type* dst, scalar_type* src)  const {fftw_execute_dft_r2c(_plan, src, dst);}
      inline void execute(scalar_type* dst,  complex_type* src) const {fftw_execute_dft_c2r(_plan, src, dst);}
    };

    template<>
    struct fftwPlan<long double> {
      typedef long double scalar_type;
      typedef fftwl_complex complex_type;

      fftwl_plan _plan;

      ~fftwPlan() {if (_plan) fftwl_destroy_plan(_plan);}

      fftwPlan(const void* _src, const void* _dst, bool inv, int nfft, int n1, int n2, bool isC2C) {
        if (isC2C) {
          complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
          complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
          if (inv) {
            if (n2)      _plan = fftwl_plan_dft_3d(nfft, n1, n2, src, dst, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftwl_plan_dft_2d(nfft, n1, src, dst, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftwl_plan_dft_1d(nfft, src, dst, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          } else {
            if (n2)      _plan = fftwl_plan_dft_3d(nfft, n1, n2, src, dst, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftwl_plan_dft_2d(nfft, n1, src, dst, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftwl_plan_dft_1d(nfft, src, dst, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
        } else {
          if (inv) {
            complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
            scalar_type* dst = const_cast<scalar_type*>(static_cast<const scalar_type*>(_dst));
            if (n2)      _plan = fftwl_plan_dft_c2r_3d(nfft, n1, n2, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftwl_plan_dft_c2r_2d(nfft, n1, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftwl_plan_dft_c2r_1d(nfft, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          } else {
            scalar_type* src = const_cast<scalar_type*>(static_cast<const scalar_type*>(_src));
            complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
            if (n2)      _plan = fftwl_plan_dft_r2c_3d(nfft, n1, n2, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else if (n1) _plan = fftwl_plan_dft_r2c_2d(nfft, n1, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
            else         _plan = fftwl_plan_dft_r2c_1d(nfft, src, dst, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
        }
      }

      fftwPlan(const void* _src, const void* _dst, bool inv, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2, bool isC2C) {
        int rank = (static_cast<int>(doDim0) + doDim1) + doDim2;
        int howmany = (doDim0? 1 : n0) * (doDim1? 1 : n1) * (doDim2? 1 : n2);
        int stride = doDim2? 1 : n2 * (doDim1? 1 : n1 * (doDim0? 1 : n0));
        int distance = doDim2? n2 * (doDim1? n1 * (doDim0? n0: 1): 1) : 1;

        std::vector<int> dimensions(rank);
        int i = 0;
        if (doDim2) dimensions[i++] = n2;
        if (doDim1) dimensions[i++] = n1;
        if (doDim0) dimensions[i++] = n0;

        if (isC2C) {
          complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
          complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
          _plan = fftwl_plan_many_dft(rank, dimensions.data(), howmany,
                                      src, NULL, stride, distance,
                                      dst, NULL, stride, distance,
                                      (inv? FFTW_BACKWARD : FFTW_FORWARD), FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
        } else {
          if (inv) {
            complex_type* src = const_cast<complex_type*>(static_cast<const complex_type*>(_src));
            scalar_type* dst = const_cast<scalar_type*>(static_cast<const scalar_type*>(_dst));
            _plan = fftwl_plan_many_dft_c2r(rank, dimensions.data(), howmany,
                                            src, NULL, stride, distance,
                                            dst, NULL, stride, distance,
                                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
          else {
            scalar_type* src = const_cast<scalar_type*>(static_cast<const scalar_type*>(_src));
            complex_type* dst = const_cast<complex_type*>(static_cast<const complex_type*>(_dst));
            _plan = fftwl_plan_many_dft_r2c(rank, dimensions.data(), howmany,
                                            src, NULL, stride, distance,
                                            dst, NULL, stride, distance,
                                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
          }
        }
      }

      inline void execute(complex_type* dst, complex_type* src) const {fftwl_execute_dft(_plan, src, dst);}
      inline void execute(complex_type* dst, scalar_type* src)  const {fftwl_execute_dft_r2c(_plan, src, dst);}
      inline void execute(scalar_type* dst,  complex_type* src) const {fftwl_execute_dft_c2r(_plan, src, dst);}
    };


    template<typename T>
    inline T* fftw_cast(const T* p) { // For non-complex
      return const_cast<T*>(p);
    }

    inline fftwf_complex* fftw_cast(const std::complex<float>* p) {
      return const_cast<fftwf_complex*>(reinterpret_cast<const fftwf_complex*>(p));
    }

    inline fftw_complex* fftw_cast(const std::complex<double>* p) {
      return const_cast<fftw_complex*>(reinterpret_cast<const fftw_complex*>(p));
    }

    inline fftwl_complex* fftw_cast(const std::complex<long double>* p) {
      return const_cast<fftwl_complex*>(reinterpret_cast<const fftwl_complex*>(p));
    }

    template<typename _scalar>
    struct fftw_impl {
      typedef _scalar Scalar;
      typedef std::complex<Scalar> Complex;

      inline void clear() {
        _plans.clear();
      }

      /// real-to-complex forward FFT
      inline void fwd(Complex* dst, const Scalar* src, int n0, int n1=0, int n2=0) {
        getPlan(dst, src, false, n0, n1, n2, true).execute(fftw_cast(dst), fftw_cast(src));
      }

      /// complex-to-complex forward FFT
      inline void fwd(Complex* dst, const Complex* src, int n0, int n1=0, int n2=0) {
        getPlan(dst, src, false, n0, n1, n2, true).execute(fftw_cast(dst), fftw_cast(src));
      }

      /// Forward FFT over a subset of dimensions (real-to-complex)
      inline void fwd(Complex* dst, const Scalar* src, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2) {
        getPlanMany(dst, src, false, n0, n1, n2, doDim0, doDim1, doDim2, false).execute(fftw_cast(dst), fftw_cast(src));
      }

      /// Forward FFT over a subset of dimensions
      inline void fwd(Complex* dst, const Complex* src, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2) {
        getPlanMany(dst, src, false, n0, n1, n2, doDim0, doDim1, doDim2, true).execute(fftw_cast(dst), fftw_cast(src));
      }

      /// complex to real inverse FFT
      inline void inv(Scalar* dst, const Complex* src, int n0, int n1=0, int n2=0) {
        getPlan(dst, src, true, n0, n1, n2, false).execute(fftw_cast(dst), fftw_cast(src));
      }

      /// complex-to-complex inverse FFT
      inline void inv(Complex* dst, const Complex* src, int n0, int n1=0, int n2=0) {
        getPlan(dst, src, true, n0, n1, n2, true).execute(fftw_cast(dst), fftw_cast(src));
      }

      /// Inverse FFT over a subset of dimensions (complex-to-real)
      inline void inv(Scalar* dst, const Complex* src, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2) {
        getPlanMany(dst, src, true, n0, n1, n2, doDim0, doDim1, doDim2, false).execute(fftw_cast(dst), fftw_cast(src));
      }

      /// Inverse FFT over a subset of dimensions
      inline void inv(Complex* dst, const Complex* src, int n0, int n1, int n2, bool doDim0, bool doDim1, bool doDim2) {
        getPlanMany(dst, src, true, n0, n1, n2, doDim0, doDim1, doDim2, true).execute(fftw_cast(dst), fftw_cast(src));
      }

    protected:
      typedef fftwPlan<Scalar> Plan;

      std::map<std::pair<uint64_t, uint64_t>, Plan> _plans;

      inline Plan& getPlan(void* dst, const void* src, bool inverse, int n0, int n1, int n2, bool isC2C) {
        bool inplace = (dst == src);
        bool aligned = ((reinterpret_cast<size_t>(src) & 0b1111) | (reinterpret_cast<size_t>(dst) & 0b1111)) == 0;

        uint64_t keyValue1 = (((((((static_cast<uint64_t>(inverse) << 1) + inplace) << 1) + aligned) << 1) + isC2C) << 32) + n0;
        uint64_t keyValue2 = (static_cast<uint64_t>(n1) << 32) + n2;

        return (*_plans.try_emplace({keyValue1, keyValue2}, src, dst, inverse, n0, n1, n2, isC2C).first).second;
      }

      inline Plan& getPlanMany(void* dst, const void* src, bool inverse, int n0, int n1, int n2,
                               bool doDim0, bool doDim1, bool doDim2, bool isC2C) {
        bool inplace = (dst == src);
        bool aligned = ((reinterpret_cast<size_t>(src) & 0b1111) | (reinterpret_cast<size_t>(dst) & 0b1111)) == 0;

        uint64_t keyValue1 = (((((((((((((static_cast<uint64_t>(doDim0) << 1) + doDim1) << 1) + doDim2) << 1) + inverse) << 1) + inplace) << 1) + aligned) + isC2C) << 1) << 32) + n0;
        uint64_t keyValue2 = (static_cast<uint64_t>(n1) << 32) + n2;

        return (*_plans.try_emplace({keyValue1, keyValue2}, src, dst, inverse, n0, n1, n2, doDim0, doDim1, doDim2, isC2C).first).second;
      }
    };

  } // end namespace internal
} // end namespace Eigen

#endif //NONLINEARMEDIUM_EIGEN_FFTW_H
