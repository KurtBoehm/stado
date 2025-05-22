#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_F32X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_F32X16_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/f32x4.hpp"
#include "stado/vector/native/types/f32x8.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
template<>
struct NativeVector<f32, 16> : public NativeVectorBase<f32, 16> {
  using Register = __m512;

  static NativeVector expand_undef(f32 x) {
    return _mm512_castps128_ps512(f32x4::expand_undef(x));
  }
  static NativeVector expand_zero(f32 x) {
    return {x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  }

  // Default constructor:
  NativeVector() = default;
  // Constructor to broadcast the same value into all elements:
  NativeVector(f32 f) : zmm_(_mm512_set1_ps(f)) {}
  // Constructor to build from all elements:
  NativeVector(f32 f0, f32 f1, f32 f2, f32 f3, f32 f4, f32 f5, f32 f6, f32 f7, f32 f8, f32 f9,
               f32 f10, f32 f11, f32 f12, f32 f13, f32 f14, f32 f15)
      : zmm_(_mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15)) {
  }
  // Constructor to build from two f32x8:
  NativeVector(const f32x8 a0, const f32x8 a1)
      : zmm_(_mm512_castpd_ps(_mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(a0)),
                                                 _mm256_castps_pd(a1), 1))) {}
  // Constructor to convert from type __m512 used in intrinsics:
  NativeVector(const __m512 x) : zmm_(x) {}
  // Assignment operator to convert from type __m512 used in intrinsics:
  NativeVector& operator=(const __m512 x) {
    zmm_ = x;
    return *this;
  }
  // Type cast operator to convert to __m512 used in intrinsics
  operator __m512() const {
    return zmm_;
  }
  // Member function to load from array (unaligned)
  NativeVector& load(const f32* p) {
    zmm_ = _mm512_loadu_ps(p);
    return *this;
  }
  // Member function to load from array, aligned by 64
  // You may use load_a instead of load if you are certain that p points to an address divisible by
  // 64
  NativeVector& load_a(const f32* p) {
    zmm_ = _mm512_load_ps(p);
    return *this;
  }
  // Member function to store into array (unaligned)
  void store(f32* p) const {
    _mm512_storeu_ps(p, zmm_);
  }
  // Member function storing into array, aligned by 64
  // You may use store_a instead of store if you are certain that p points to an address divisible
  // by 64
  void store_a(f32* p) const {
    _mm512_store_ps(p, zmm_);
  }
  // Member function storing to aligned uncached memory (non-temporal store).
  // This may be more efficient than store_a when storing large blocks of memory if it
  // is unlikely that the data will stay in the cache until it is read again.
  // Note: Will generate runtime error if p is not aligned by 16
  void store_nt(f32* p) const {
    _mm512_stream_ps(p, zmm_);
  }
  // Partial load. Load n elements and set the rest to 0
  NativeVector& load_partial(std::size_t n, const f32* p) {
    zmm_ = _mm512_maskz_loadu_ps(__mmask16((1U << n) - 1U), p);
    return *this;
  }
  template<std::size_t tN>
  NativeVector& load_partial(const f32* p) {
    zmm_ = _mm512_maskz_loadu_ps(__mmask16((1U << tN) - 1U), p);
    return *this;
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, f32* p) const {
    _mm512_mask_storeu_ps(p, __mmask16((1U << n) - 1), zmm_);
  }
  // cut off vector to n elements. The last 8-n elements are set to zero
  NativeVector& cutoff(std::size_t n) {
    zmm_ = _mm512_maskz_mov_ps(__mmask16((1U << n) - 1), zmm_);
    return *this;
  }
  // Member function to change a single element in vector
  NativeVector& insert(std::size_t index, f32 value) {
    zmm_ = _mm512_mask_broadcastss_ps(zmm_, __mmask16(1U << index), _mm_set_ss(value));
    return *this;
  }
  // Member function extract a single element from vector
  [[nodiscard]] f32 extract(std::size_t index) const {
    const __m512 x = _mm512_maskz_compress_ps(__mmask16(1U << index), zmm_);
    return _mm512_cvtss_f32(x);
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  f32 operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two f32x8:
  [[nodiscard]] f32x8 get_low() const {
    return _mm512_castps512_ps256(zmm_);
  }
  [[nodiscard]] f32x8 get_high() const {
    return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(zmm_), 1));
  }

private:
  __m512 zmm_;
};

using f32x16 = NativeVector<f32, 16>;
} // namespace stado

#endif // AVX512F

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_F32X16_HPP
