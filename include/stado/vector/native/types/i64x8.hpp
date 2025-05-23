#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I64X8_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I64X8_HPP

#include <concepts>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-512.hpp"
#include "stado/vector/native/types/i64x4.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
template<typename TDerived, typename T>
struct x64x8 : public si512 {
  using Element = T;
  using Half = NativeVector<T, 4>;
  static constexpr std::size_t size = 8;

  // Default constructor:
  x64x8() = default;
  // Constructor to broadcast the same value into all elements:
  x64x8(T i) : si512(_mm512_set1_epi64(i64(i))) {}
  // Constructor to build from all elements:
  x64x8(T i0, T i1, T i2, T i3, T i4, T i5, T i6, T i7)
      : si512(_mm512_setr_epi64(i64(i0), i64(i1), i64(i2), i64(i3), i64(i4), i64(i5), i64(i6),
                                i64(i7))) {}
  // Constructor to build from two x64x4:
  x64x8(const Half a0, const Half a1)
      : si512(_mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1)) {}
  // Constructor to convert from type __m512i used in intrinsics:
  x64x8(const __m512i x) : si512(x) {}
  // Assignment operator to convert from type __m512i used in intrinsics:
  TDerived& operator=(const __m512i x) {
    zmm = x;
    return derived();
  }
  // Type cast operator to convert to __m512i used in intrinsics
  operator __m512i() const {
    return zmm;
  }
  // Member function to load from array (unaligned)
  TDerived& load(const void* p) {
    zmm = _mm512_loadu_si512(p);
    return derived();
  }
  // Member function to load from array, aligned by 64
  TDerived& load_a(const void* p) {
    zmm = _mm512_load_si512(p);
    return derived();
  }
  // Partial load. Load n elements and set the rest to 0
  TDerived& load_partial(std::size_t n, const void* p) {
    zmm = _mm512_maskz_loadu_epi64(__mmask16((1U << n) - 1), p);
    return derived();
  }
  template<std::size_t n>
  TDerived& load_partial(const void* p) {
    zmm = _mm512_maskz_loadu_epi64(__mmask16((1U << n) - 1), p);
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
    _mm512_mask_storeu_epi64(p, __mmask16((1U << n) - 1), zmm);
  }
  // cut off vector to n elements. The last 8-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
    zmm = _mm512_maskz_mov_epi64(__mmask16((1U << n) - 1), zmm);
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
    zmm = _mm512_mask_set1_epi64(zmm, __mmask16(1U << index), i64(value));
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
    __m512i x = _mm512_maskz_compress_epi64(__mmask8(1U << index), zmm);
    return T(_mm_cvtsi128_si64(_mm512_castsi512_si128(x)));
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  T operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two x64x4:
  [[nodiscard]] Half get_low() const {
    return _mm512_castsi512_si256(zmm);
  }
  [[nodiscard]] Half get_high() const {
    return _mm512_extracti64x4_epi64(zmm, 1);
  }

private:
  [[nodiscard]] const TDerived& derived() const {
    return static_cast<const TDerived&>(*this);
  }
  [[nodiscard]] TDerived& derived() {
    return static_cast<TDerived&>(*this);
  }
};

template<>
struct NativeVector<i64, 8> : public x64x8<NativeVector<i64, 8>, i64> {
  using x64x8<NativeVector<i64, 8>, i64>::x64x8;
};
using i64x8 = NativeVector<i64, 8>;

template<>
struct NativeVector<u64, 8> : public x64x8<NativeVector<u64, 8>, u64> {
  using x64x8<NativeVector<u64, 8>, u64>::x64x8;
};
using u64x8 = NativeVector<u64, 8>;

template<typename TVec>
concept AnyInt64x8 = std::same_as<TVec, i64x8> || std::same_as<TVec, u64x8>;
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I64X8_HPP
