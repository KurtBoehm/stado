#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I16X32_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I16X32_HPP

#include <concepts>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-512.hpp"
#include "stado/vector/native/types/i16x16.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
namespace stado {
template<typename TDerived, typename T>
struct x16x32 : public si512 {
  using Element = T;
  using Half = NativeVector<T, 16>;
  static constexpr std::size_t size = 32;

  // Default constructor:
  x16x32() = default;
  // Constructor to broadcast the same value into all elements:
  x16x32(T i) : si512(_mm512_set1_epi16(i16(i))) {}
  // Constructor to build from all elements:
  x16x32(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13,
         T v14, T v15, T v16, T v17, T v18, T v19, T v20, T v21, T v22, T v23, T v24, T v25, T v26,
         T v27, T v28, T v29, T v30, T v31)
      : si512(_mm512_set_epi16(i16(v31), i16(v30), i16(v29), i16(v28), i16(v27), i16(v26), i16(v25),
                               i16(v24), i16(v23), i16(v22), i16(v21), i16(v20), i16(v19), i16(v18),
                               i16(v17), i16(v16), i16(v15), i16(v14), i16(v13), i16(v12), i16(v11),
                               i16(v10), i16(v9), i16(v8), i16(v7), i16(v6), i16(v5), i16(v4),
                               i16(v3), i16(v2), i16(v1), i16(v0))) {}
  // Constructor to build from two x16x16:
  x16x32(const NativeVector<T, 16> a0, const NativeVector<T, 16> a1)
      : si512(_mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1)) {}
  // Constructor to convert from type __m512i used in intrinsics:
  x16x32(const __m512i x) : si512(x) {}
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
    zmm = _mm512_maskz_loadu_epi16(__mmask32((u64{1} << n) - 1), p);
    return derived();
  }
  template<std::size_t n>
  TDerived& load_partial(const void* p) {
    zmm = _mm512_maskz_loadu_epi16(__mmask32((u64{1} << n) - 1), p);
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
    _mm512_mask_storeu_epi16(p, __mmask32((u64{1} << n) - 1), zmm);
  }
  // cut off vector to n elements. The last 32-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
    zmm = _mm512_maskz_mov_epi16(__mmask32((u64{1} << n) - 1), zmm);
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
    zmm = _mm512_mask_set1_epi16(zmm, __mmask64(u64{1} << index), i16(value));
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI2__)
    __m512i x = _mm512_maskz_compress_epi16(__mmask32(1U << index), zmm);
    return T(_mm_cvtsi128_si32(_mm512_castsi512_si128(x)));
#else
    T a[32];
    store(a);
    return a[index & 31];
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  T operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two x16x16:
  [[nodiscard]] NativeVector<T, 16> get_low() const {
    return _mm512_castsi512_si256(zmm);
  }
  [[nodiscard]] NativeVector<T, 16> get_high() const {
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
struct NativeVector<i16, 32> : public x16x32<NativeVector<i16, 32>, i16> {
  using x16x32<NativeVector<i16, 32>, i16>::x16x32;
};
using i16x32 = NativeVector<i16, 32>;

template<>
struct NativeVector<u16, 32> : public x16x32<NativeVector<u16, 32>, u16> {
  using x16x32<NativeVector<u16, 32>, u16>::x16x32;
};
using u16x32 = NativeVector<u16, 32>;

template<typename TVec>
concept AnyInt16x32 = std::same_as<TVec, i16x32> || std::same_as<TVec, u16x32>;
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I16X32_HPP
