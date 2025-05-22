#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_F64X4_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_F64X4_HPP

#include <bit>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/f32x8.hpp"
#include "stado/vector/native/types/f64x2.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX
namespace stado {
template<>
struct NativeVector<f64, 4> : public NativeVectorBase<f64, 4> {
  using Register = __m256d;

  static NativeVector expand_undef(f64 x) {
    return _mm256_castpd128_pd256(f64x2::expand_undef(x));
  }
  static NativeVector expand_zero(f64 x) {
    return {x, 0, 0, 0};
  }

  // Default constructor:
  NativeVector() = default;
  // Constructor to broadcast the same value into all elements:
  NativeVector(f64 d) : ymm_(_mm256_set1_pd(d)) {}
  // Constructor to build from all elements:
  NativeVector(f64 d0, f64 d1, f64 d2, f64 d3) : ymm_(_mm256_setr_pd(d0, d1, d2, d3)) {}
  // Constructor to build from two f64x2:
  NativeVector(const f64x2 a0, const f64x2 a1) : ymm_(_mm256_set_m128d(a1, a0)) {}
  // Constructor to convert from type __m256d used in intrinsics:
  NativeVector(const __m256d x) : ymm_(x) {}
  // Assignment operator to convert from type __m256d used in intrinsics:
  NativeVector& operator=(const __m256d x) {
    ymm_ = x;
    return *this;
  }
  // Type cast operator to convert to __m256d used in intrinsics
  operator __m256d() const {
    return ymm_;
  }
  // Member function to load from array (unaligned)
  NativeVector& load(const f64* p) {
    ymm_ = _mm256_loadu_pd(p);
    return *this;
  }
  // Member function to load from array, aligned by 32
  // You may use load_a instead of load if you are certain that p points to an address
  // divisible by 32
  NativeVector& load_a(const f64* p) {
    ymm_ = _mm256_load_pd(p);
    return *this;
  }
  // Member function to store into array (unaligned)
  void store(f64* p) const {
    _mm256_storeu_pd(p, ymm_);
  }
  // Member function storing into array, aligned by 32
  // You may use store_a instead of store if you are certain that p points to an address
  // divisible by 32
  void store_a(f64* p) const {
    _mm256_store_pd(p, ymm_);
  }
  // Member function storing to aligned uncached memory (non-temporal store).
  // This may be more efficient than store_a when storing large blocks of memory if it
  // is unlikely that the data will stay in the cache until it is read again.
  // Note: Will generate runtime error if p is not aligned by 32
  void store_nt(f64* p) const {
    _mm256_stream_pd(p, ymm_);
  }
  // Partial load. Load n elements and set the rest to 0
  NativeVector& load_partial(std::size_t n, const f64* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm_ = _mm256_maskz_loadu_pd(__mmask8((1U << n) - 1), p);
#else
    if (n > 0 && n <= 2) {
      *this = NativeVector(f64x2().load_partial(n, p), _mm_setzero_pd());
    } else if (n > 2 && n <= 4) {
      *this = NativeVector(f64x2().load(p), f64x2().load_partial(n - 2, p + 2));
    } else {
      ymm_ = _mm256_setzero_pd();
    }
#endif
    return *this;
  }
  template<std::size_t n>
  NativeVector& load_partial(const f64* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm_ = _mm256_maskz_loadu_pd(__mmask8((1U << n) - 1), p);
#else
    if constexpr (n > 0 && n <= 2) {
      *this = NativeVector(f64x2().load_partial<n>(p), _mm_setzero_pd());
    } else if constexpr (n <= 4) {
      *this = NativeVector(f64x2().load(p), f64x2().load_partial<n - 2>(p + 2));
    } else {
      ymm_ = _mm256_setzero_pd();
    }
#endif
    return *this;
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, f64* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    _mm256_mask_storeu_pd(p, __mmask8((1U << n) - 1), ymm_);
#else
    if (n <= 2) {
      get_low().store_partial(n, p);
    } else if (n <= 4) {
      get_low().store(p);
      get_high().store_partial(n - 2, p + 2);
    }
#endif
  }
  // cut off vector to n elements. The last 4-n elements are set to zero
  NativeVector& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    ymm_ = _mm256_maskz_mov_pd(__mmask8((1U << n) - 1), ymm_);
#else
    ymm_ = _mm256_castps_pd(f32x8(_mm256_castpd_ps(ymm_)).cutoff(n * 2));
#endif
    return *this;
  }
  // Member function to change a single element in vector
  // Note: This function is inefficient. Use load function if changing more than one element
  NativeVector& insert(std::size_t index, f64 value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm_ = _mm256_mask_broadcastsd_pd(ymm_, __mmask8(1U << index), _mm_set_sd(value));
#else
    __m256d v0 = _mm256_broadcast_sd(&value);
    switch (index) {
    case 0: ymm_ = _mm256_blend_pd(ymm_, v0, 1); break;
    case 1: ymm_ = _mm256_blend_pd(ymm_, v0, 2); break;
    case 2: ymm_ = _mm256_blend_pd(ymm_, v0, 4); break;
    default: ymm_ = _mm256_blend_pd(ymm_, v0, 8); break;
    }
#endif
    return *this;
  }
  // Member function extract a single element from vector
  [[nodiscard]] f64 extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    const __m256d x = _mm256_maskz_compress_pd(__mmask8(1U << index), ymm_);
    return _mm256_cvtsd_f64(x);
#else
    f64 x[4];
    store(x);
    return x[index & 3U];
#endif
  }
  template<std::size_t tIdx>
  requires(tIdx < 4)
  [[nodiscard]] f64 extract() const {
    return std::bit_cast<f64>(_mm256_extract_epi64(_mm256_castpd_si256(ymm_), tIdx));
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  f64 operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two f64x2:
  [[nodiscard]] f64x2 get_low() const {
    return _mm256_castpd256_pd128(ymm_);
  }
  [[nodiscard]] f64x2 get_high() const {
    return _mm256_extractf128_pd(ymm_, 1);
  }

private:
  __m256d ymm_;
};

using f64x4 = NativeVector<f64, 4>;
} // namespace stado
#endif // AVX

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_F64X4_HPP
