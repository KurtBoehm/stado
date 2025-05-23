#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_F64X02_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_F64X02_HPP

#include <bit>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/f32x04.hpp"

namespace stado {
template<>
struct NativeVector<f64, 2> : public NativeVectorBase<f64, 2> {
  using Register = __m128d;

  static NativeVector expand_undef(f64 x) {
#if defined(__GNUC__) && !defined(__clang__)
    __m128d retval;
    asm("" : "=x"(retval) : "0"(x));
    return retval;
#elif defined(__clang__)
    f64 data[2];
    data[0] = x;
    return _mm_load_pd(data);
#else
    return _mm_set_sd(x);
#endif
  }
  static NativeVector expand_zero(f64 x) {
    return {x, 0};
  }

  // Default constructor:
  NativeVector() = default;
  // Constructor to broadcast the same value into all elements:
  NativeVector(f64 d) : xmm_(_mm_set1_pd(d)) {}
  // Constructor to build from all elements:
  NativeVector(f64 d0, f64 d1) : xmm_(_mm_setr_pd(d0, d1)) {}
  // Constructor to convert from type __m128d used in intrinsics:
  NativeVector(const __m128d x) : xmm_(x) {}
  // Assignment operator to convert from type __m128d used in intrinsics:
  NativeVector& operator=(const __m128d x) {
    xmm_ = x;
    return *this;
  }
  // Type cast operator to convert to __m128d used in intrinsics
  operator __m128d() const {
    return xmm_;
  }
  // Member function to load from array (unaligned)
  NativeVector& load(const f64* p) {
    xmm_ = _mm_loadu_pd(p);
    return *this;
  }
  // Member function to load from array, aligned by 16
  // "load_a" is faster than "load" on older Intel processors (Pentium 4, Pentium M, Core 1,
  // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
  // You may use load_a instead of load if you are certain that p points to an address
  // divisible by 16.
  NativeVector& load_a(const f64* p) {
    xmm_ = _mm_load_pd(p);
    return *this;
  }
  // Member function to store into array (unaligned)
  void store(f64* p) const {
    _mm_storeu_pd(p, xmm_);
  }
  // Member function storing into array, aligned by 16
  // "store_a" is faster than "store" on older Intel processors (Pentium 4, Pentium M, Core 1,
  // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
  // You may use store_a instead of store if you are certain that p points to an address
  // divisible by 16.
  void store_a(f64* p) const {
    _mm_store_pd(p, xmm_);
  }
  // Member function storing to aligned uncached memory (non-temporal store).
  // This may be more efficient than store_a when storing large blocks of memory if it
  // is unlikely that the data will stay in the cache until it is read again.
  // Note: Will generate runtime error if p is not aligned by 16
  void store_nt(f64* p) const {
    _mm_stream_pd(p, xmm_);
  }
  // Partial load. Load n elements and set the rest to 0
  NativeVector& load_partial(std::size_t n, const f64* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    xmm_ = _mm_maskz_loadu_pd(__mmask8((1U << n) - 1), p);
#else
    if (n == 1) {
      xmm_ = _mm_load_sd(p);
    } else if (n == 2) {
      load(p);
    } else {
      xmm_ = _mm_setzero_pd();
    }
#endif
    return *this;
  }
  template<std::size_t n>
  NativeVector& load_partial(const f64* p) {
    static_assert(n <= 2);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    xmm_ = _mm_maskz_loadu_pd(__mmask8((1U << n) - 1), p);
#else
    if constexpr (n == 0) {
      xmm_ = _mm_setzero_pd();
    } else if constexpr (n == 1) {
      xmm_ = _mm_load_sd(p);
    } else {
      load(p);
    }
#endif
    return *this;
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, f64* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    _mm_mask_storeu_pd(p, __mmask8((1U << n) - 1), xmm_);
#else
    if (n == 1) {
      _mm_store_sd(p, xmm_);
    } else if (n == 2) {
      store(p);
    }
#endif
  }
  // cut off vector to n elements. The last 4-n elements are set to zero
  NativeVector& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    xmm_ = _mm_maskz_mov_pd(__mmask8((1U << n) - 1), xmm_);
#else
    xmm_ = _mm_castps_pd(f32x4(_mm_castpd_ps(xmm_)).cutoff(n * 2));
#endif
    return *this;
  }
  // Member function to change a single element in vector
  // Note: This function is inefficient. Use load function if changing more than one element
  NativeVector& insert(std::size_t index, f64 value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    xmm_ = _mm_mask_movedup_pd(xmm_, __mmask8(1U << index), _mm_set_sd(value));
#else
    __m128d v2 = _mm_set_sd(value);
    if (index == 0) {
      xmm_ = _mm_shuffle_pd(v2, xmm_, 2);
    } else {
      xmm_ = _mm_shuffle_pd(xmm_, v2, 0);
    }
#endif
    return *this;
  }
  // Member function extract a single element from vector
  [[nodiscard]] f64 extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    const __m128d x = _mm_mask_unpackhi_pd(xmm_, __mmask8(index), xmm_, xmm_);
    return _mm_cvtsd_f64(x);
#else
    f64 x[2];
    store(x);
    return x[index & 1];
#endif
  }
  template<std::size_t tIdx>
  requires(tIdx < 2)
  [[nodiscard]] f64 extract() const {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
    return std::bit_cast<f64>(_mm_extract_epi64(_mm_castpd_si128(xmm_), tIdx));
#else
    f64 x[2];
    store(x);
    return x[tIdx];
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  f64 operator[](std::size_t index) const {
    return extract(index);
  }

private:
  __m128d xmm_;
};

using f64x2 = NativeVector<f64, 2>;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_F64X02_HPP
