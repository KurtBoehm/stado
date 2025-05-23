#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_F32X04_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_F32X04_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"

namespace stado {
template<>
struct NativeVector<f32, 4> : public NativeVectorBase<f32, 4> {
  using Register = __m128;

  static NativeVector expand_undef(f32 x) {
#if defined(__GNUC__) && !defined(__clang__)
    __m128 retval;
    asm("" : "=x"(retval) : "0"(x));
    return retval;
#elif defined(__clang__)
    f32 data[4];
    data[0] = x;
    return _mm_load_ps(data);
#else
    return _mm_set_ss(x);
#endif
  }
  static NativeVector expand_zero(f32 x) {
    return {x, 0, 0, 0};
  }

  // Default constructor:
  NativeVector() = default;
  // Constructor to broadcast the same value into all elements:
  NativeVector(f32 f) : xmm_(_mm_set1_ps(f)) {}
  // Constructor to build from all elements:
  NativeVector(f32 f0, f32 f1, f32 f2, f32 f3) : xmm_(_mm_setr_ps(f0, f1, f2, f3)) {}
  // Constructor to convert from type __m128 used in intrinsics:
  NativeVector(const __m128 x) : xmm_(x) {}
  // Assignment operator to convert from type __m128 used in intrinsics:
  NativeVector& operator=(const __m128 x) {
    xmm_ = x;
    return *this;
  }
  // Type cast operator to convert to __m128 used in intrinsics
  operator __m128() const {
    return xmm_;
  }
  // Member function to load from array (unaligned)
  NativeVector& load(const f32* p) {
    xmm_ = _mm_loadu_ps(p);
    return *this;
  }
  // Member function to load from array, aligned by 16
  // "load_a" is faster than "load" on older Intel processors (Pentium 4, Pentium M, Core 1,
  // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
  // You may use load_a instead of load if you are certain that p points to an address
  // divisible by 16.
  NativeVector& load_a(const f32* p) {
    xmm_ = _mm_load_ps(p);
    return *this;
  }
  // Member function to store into array (unaligned)
  void store(f32* p) const {
    _mm_storeu_ps(p, xmm_);
  }
  // Member function storing into array, aligned by 16
  // "store_a" is faster than "store" on older Intel processors (Pentium 4, Pentium M, Core 1,
  // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
  // You may use store_a instead of store if you are certain that p points to an address
  // divisible by 16.
  void store_a(f32* p) const {
    _mm_store_ps(p, xmm_);
  }
  // Member function storing to aligned uncached memory (non-temporal store).
  // This may be more efficient than store_a when storing large blocks of memory if it
  // is unlikely that the data will stay in the cache until it is read again.
  // Note: Will generate runtime error if p is not aligned by 16
  void store_nt(f32* p) const {
    _mm_stream_ps(p, xmm_);
  }
  // Partial load. Load n elements and set the rest to 0
  NativeVector& load_partial(std::size_t n, const f32* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    xmm_ = _mm_maskz_loadu_ps(__mmask8((1U << n) - 1), p);
#else
    switch (n) {
    case 1: xmm_ = _mm_load_ss(p); break;
    case 2: xmm_ = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const f64*>(p))); break;
    case 3: {
      __m128 t1 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const f64*>(p)));
      __m128 t2 = _mm_load_ss(p + 2);
      xmm_ = _mm_movelh_ps(t1, t2);
      break;
    }
    case 4: load(p); break;
    default: xmm_ = _mm_setzero_ps();
    }
#endif
    return *this;
  }
  template<std::size_t tN>
  NativeVector& load_partial(const f32* p) {
    static_assert(tN <= 4);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    xmm_ = _mm_maskz_loadu_ps(__mmask8((1U << tN) - 1), p);
#else
    if constexpr (tN == 0) {
      xmm_ = _mm_setzero_ps();
    } else if constexpr (tN == 1) {
      xmm_ = _mm_load_ss(p);
    } else if constexpr (tN == 2) {
      xmm_ = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const f64*>(p)));
    } else if (tN == 3) {
      __m128 t1 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const f64*>(p)));
      __m128 t2 = _mm_load_ss(p + 2);
      xmm_ = _mm_movelh_ps(t1, t2);
    } else {
      load(p);
    }
#endif
    return *this;
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, f32* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    _mm_mask_storeu_ps(p, __mmask8((1U << n) - 1), xmm_);
#else
    switch (n) {
    case 1: _mm_store_ss(p, xmm_); break;
    case 2: _mm_store_sd(reinterpret_cast<f64*>(p), _mm_castps_pd(xmm_)); break;
    case 3: {
      _mm_store_sd(reinterpret_cast<f64*>(p), _mm_castps_pd(xmm_));
      __m128 t1 = _mm_movehl_ps(xmm_, xmm_);
      _mm_store_ss(p + 2, t1);
      break;
    }
    case 4: store(p); break;
    default:;
    }
#endif
  }
  // cut off vector to n elements. The last 4-n elements are set to zero
  NativeVector& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    xmm_ = _mm_maskz_mov_ps(__mmask8((1U << n) - 1), xmm_);
#else
    if (u32(n) >= 4) {
      return *this;
    }
    __m128i idxs = _mm_setr_epi32(0, 1, 2, 3);
    __m128i ref = _mm_set1_epi32(i32(n));
    __m128 mask = _mm_castsi128_ps(_mm_cmpgt_epi32(ref, idxs));
    xmm_ = _mm_and_ps(mask, *this);
#endif
    return *this;
  }
  // Member function to change a single element in vector
  NativeVector& insert(std::size_t index, f32 value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    xmm_ = _mm_mask_broadcastss_ps(xmm_, __mmask8(1U << index), _mm_set_ss(value));
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
    switch (index & 3U) {
    case 0: xmm_ = _mm_insert_ps(xmm_, _mm_set_ss(value), 0U << 4U); break;
    case 1: xmm_ = _mm_insert_ps(xmm_, _mm_set_ss(value), 1U << 4U); break;
    case 2: xmm_ = _mm_insert_ps(xmm_, _mm_set_ss(value), 2U << 4U); break;
    default: xmm_ = _mm_insert_ps(xmm_, _mm_set_ss(value), 3U << 4U); break;
    }
#else
    const i32 maskl[8] = {0, 0, 0, 0, -1, 0, 0, 0};
    // broadcast value into all elements
    __m128 broad = _mm_set1_ps(value);
    // mask with FFFFFFFF at index position
    __m128 mask = _mm_loadu_ps(reinterpret_cast<const f32*>(maskl + 4 - (index & 3)));
    xmm_ = _mm_or_ps(_mm_and_ps(mask, broad), _mm_andnot_ps(mask, xmm_));
#endif
    return *this;
  }
  // Member function extract a single element from vector
  [[nodiscard]] f32 extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    const __m128 x = _mm_maskz_compress_ps(__mmask8(1U << index), xmm_);
    return _mm_cvtss_f32(x);
#else
    f32 x[4];
    store(x);
    return x[index & 3];
#endif
  }
  template<std::size_t tIdx>
  requires(tIdx < 4)
  [[nodiscard]] f32 extract() const {
    return _mm_extract_ps(xmm_, tIdx);
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  f32 operator[](std::size_t index) const {
    return extract(index);
  }

private:
  __m128 xmm_; // Float vector
};

using f32x4 = NativeVector<f32, 4>;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_F32X04_HPP
