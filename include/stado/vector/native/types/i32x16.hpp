#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I32X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I32X16_HPP

#include <concepts>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-512.hpp"
#include "stado/vector/native/types/i32x08.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
template<typename TDerived, typename T>
struct x32x16 : public si512 {
  using Value = T;
  using Half = NativeVector<T, 8>;
  static constexpr std::size_t size = 16;

  // Default constructor:
  x32x16() = default;
  // Constructor to broadcast the same value into all elements:
  x32x16(T i) : si512(_mm512_set1_epi32(i32(i))) {}
  // Constructor to build from all elements:
  x32x16(T i0, T i1, T i2, T i3, T i4, T i5, T i6, T i7, T i8, T i9, T i10, T i11, T i12, T i13,
         T i14, T i15)
      : si512(_mm512_setr_epi32(i32(i0), i32(i1), i32(i2), i32(i3), i32(i4), i32(i5), i32(i6),
                                i32(i7), i32(i8), i32(i9), i32(i10), i32(i11), i32(i12), i32(i13),
                                i32(i14), i32(i15))) {}
  // Constructor to build from two x32x8:
  x32x16(const Half a0, const Half a1)
      : si512(_mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1)) {}
  // Constructor to convert from type __m512i used in intrinsics:
  x32x16(const __m512i x) : si512(x) {}
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
    zmm = _mm512_maskz_loadu_epi32(__mmask16((1U << n) - 1), p);
    return derived();
  }
  template<std::size_t n>
  TDerived& load_partial(const void* p) {
    zmm = _mm512_maskz_loadu_epi32(__mmask16((1U << n) - 1), p);
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
    _mm512_mask_storeu_epi32(p, __mmask16((1U << n) - 1), zmm);
  }
  // cut off vector to n elements. The last 16-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
    zmm = _mm512_maskz_mov_epi32(__mmask16((1U << n) - 1), zmm);
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
    zmm = _mm512_mask_set1_epi32(zmm, __mmask16(1U << index), i32(value));
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
    __m512i x = _mm512_maskz_compress_epi32(__mmask16(1U << index), zmm);
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(x));
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  T operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two x32x8:
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
struct NativeVector<i32, 16> : public x32x16<NativeVector<i32, 16>, i32> {
  using x32x16<NativeVector<i32, 16>, i32>::x32x16;
};
using i32x16 = NativeVector<i32, 16>;

template<>
struct NativeVector<u32, 16> : public x32x16<NativeVector<u32, 16>, u32> {
  using x32x16<NativeVector<u32, 16>, u32>::x32x16;
};
using u32x16 = NativeVector<u32, 16>;

template<typename TVec>
concept AnyInt32x16 = std::same_as<TVec, i32x16> || std::same_as<TVec, u32x16>;
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I32X16_HPP
