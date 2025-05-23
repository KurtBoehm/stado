#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_F32X08_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_F32X08_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/f32x04.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX
namespace stado {
template<>
struct NativeVector<f32, 8> : public NativeVectorBase<f32, 8> {
  using Register = __m256;

  static NativeVector expand_undef(f32 x) {
    return _mm256_castps128_ps256(f32x4::expand_undef(x));
  }
  static NativeVector expand_zero(f32 x) {
    return {x, 0, 0, 0, 0, 0, 0, 0};
  }

  // Default constructor:
  NativeVector() = default;
  // Constructor to broadcast the same value into all elements:
  NativeVector(f32 f) : ymm_(_mm256_set1_ps(f)) {}
  // Constructor to build from all elements:
  NativeVector(f32 f0, f32 f1, f32 f2, f32 f3, f32 f4, f32 f5, f32 f6, f32 f7)
      : ymm_(_mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7)) {}
  // Constructor to build from two f32x4:
  NativeVector(const f32x4 a0, const f32x4 a1) : ymm_(_mm256_set_m128(a1, a0)) {}
  // Constructor to convert from type __m256 used in intrinsics:
  NativeVector(const __m256 x) : ymm_(x) {}
  // Assignment operator to convert from type __m256 used in intrinsics:
  NativeVector& operator=(const __m256 x) {
    ymm_ = x;
    return *this;
  }
  // Type cast operator to convert to __m256 used in intrinsics
  operator __m256() const {
    return ymm_;
  }
  // Member function to load from array (unaligned)
  NativeVector& load(const f32* p) {
    ymm_ = _mm256_loadu_ps(p);
    return *this;
  }
  // Member function to load from array, aligned by 32
  // You may use load_a instead of load if you are certain that p points to an address divisible by
  // 32
  NativeVector& load_a(const f32* p) {
    ymm_ = _mm256_load_ps(p);
    return *this;
  }
  // Member function to store into array (unaligned)
  void store(f32* p) const {
    _mm256_storeu_ps(p, ymm_);
  }
  // Member function storing into array, aligned by 32
  // You may use store_a instead of store if you are certain that p points to an address divisible
  // by 32
  void store_a(f32* p) const {
    _mm256_store_ps(p, ymm_);
  }
  // Member function storing to aligned uncached memory (non-temporal store).
  // This may be more efficient than store_a when storing large blocks of memory if it
  // is unlikely that the data will stay in the cache until it is read again.
  // Note: Will generate runtime error if p is not aligned by 32
  void store_nt(f32* p) const {
    _mm256_stream_ps(p, ymm_);
  }
  // Partial load. Load n elements and set the rest to 0
  NativeVector& load_partial(std::size_t n, const f32* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm_ = _mm256_maskz_loadu_ps(__mmask8((1U << n) - 1), p);
#else
    if (n > 0 && n <= 4) {
      *this = NativeVector{f32x4().load_partial(n, p), _mm_setzero_ps()};
    } else if (n > 4 && n <= 8) {
      *this = NativeVector{f32x4().load(p), f32x4().load_partial(n - 4, p + 4)};
    } else {
      ymm_ = _mm256_setzero_ps();
    }
#endif
    return *this;
  }
  template<std::size_t tN>
  NativeVector& load_partial(const f32* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm_ = _mm256_maskz_loadu_ps(__mmask8((1U << tN) - 1), p);
#else
    if constexpr (tN > 0 && tN <= 4) {
      *this = NativeVector{f32x4().load_partial<tN>(p), _mm_setzero_ps()};
    } else if (tN <= 8) {
      *this = NativeVector{f32x4().load(p), f32x4().load_partial<tN - 4>(p + 4)};
    } else {
      ymm_ = _mm256_setzero_ps();
    }
#endif
    return *this;
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, f32* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    _mm256_mask_storeu_ps(p, __mmask8((1U << n) - 1), ymm_);
#else
    if (n <= 4) {
      get_low().store_partial(n, p);
    } else if (n <= 8) {
      get_low().store(p);
      get_high().store_partial(n - 4, p + 4);
    }
#endif
  }
  // cut off vector to n elements. The last 8-n elements are set to zero
  NativeVector& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    ymm_ = _mm256_maskz_mov_ps(__mmask8((1U << n) - 1), ymm_);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
    __m256i idxs = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i ref = _mm256_set1_epi32(i32(n));
    __m256 mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(ref, idxs));
    *this = _mm256_and_ps(mask, *this);
#else
    __m256 idxs = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
    __m256 ref = _mm256_set1_ps(f32(n));
    __m256 mask = _mm256_cmp_ps(idxs, ref, 1);
    *this = _mm256_and_ps(mask, *this);
#endif
    return *this;
  }
  // Member function to change a single element in vector
  NativeVector& insert(std::size_t index, f32 value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm_ = _mm256_mask_broadcastss_ps(ymm_, __mmask8(1U << index), _mm_set_ss(value));
#else
    __m256 v0 = _mm256_broadcast_ss(&value);
    switch (index) {
    case 0: ymm_ = _mm256_blend_ps(ymm_, v0, 1U << 0U); break;
    case 1: ymm_ = _mm256_blend_ps(ymm_, v0, 1U << 1U); break;
    case 2: ymm_ = _mm256_blend_ps(ymm_, v0, 1U << 2U); break;
    case 3: ymm_ = _mm256_blend_ps(ymm_, v0, 1U << 3U); break;
    case 4: ymm_ = _mm256_blend_ps(ymm_, v0, 1U << 4U); break;
    case 5: ymm_ = _mm256_blend_ps(ymm_, v0, 1U << 5U); break;
    case 6: ymm_ = _mm256_blend_ps(ymm_, v0, 1U << 6U); break;
    default: ymm_ = _mm256_blend_ps(ymm_, v0, 1U << 7U); break;
    }
#endif
    return *this;
  }
  // Member function extract a single element from vector
  [[nodiscard]] f32 extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    const __m256 x = _mm256_maskz_compress_ps(__mmask8(1U << index), ymm_);
    return _mm256_cvtss_f32(x);
#else
    f32 x[8];
    store(x);
    return x[index & 7];
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  f32 operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two f32x4:
  [[nodiscard]] f32x4 get_low() const {
    return _mm256_castps256_ps128(ymm_);
  }
  [[nodiscard]] f32x4 get_high() const {
    return _mm256_extractf128_ps(ymm_, 1);
  }

private:
  __m256 ymm_;
};

using f32x8 = NativeVector<f32, 8>;
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_F32X08_HPP
