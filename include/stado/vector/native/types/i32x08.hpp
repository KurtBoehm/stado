#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I32X08_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I32X08_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i08x32.hpp"
#include "stado/vector/native/types/i32x04.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
template<typename TDerived, typename T>
struct x32x8 : public si256 {
  using Element = T;
  using Half = NativeVector<T, 4>;
  static constexpr std::size_t size = 8;

  // Default constructor:
  x32x8() = default;
  // Constructor to broadcast the same value into all elements:
  x32x8(T i) : si256(_mm256_set1_epi32(i32(i))) {}
  // Constructor to build from all elements:
  x32x8(T i0, T i1, T i2, T i3, T i4, T i5, T i6, T i7)
      : si256(_mm256_setr_epi32(i32(i0), i32(i1), i32(i2), i32(i3), i32(i4), i32(i5), i32(i6),
                                i32(i7))) {}
  // Constructor to build from two x32x4:
  x32x8(const Half a0, const Half a1) : si256(_mm256_setr_m128i(a0, a1)) {}
  // Constructor to convert from type __m256i used in intrinsics:
  x32x8(const __m256i x) : si256(x) {}
  // Assignment operator to convert from type __m256i used in intrinsics:
  TDerived& operator=(const __m256i x) {
    ymm = x;
    return derived();
  }
  // Type cast operator to convert to __m256i used in intrinsics
  operator __m256i() const {
    return ymm;
  }
  // Member function to load from array (unaligned)
  TDerived& load(const void* p) {
    ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return derived();
  }
  // Member function to load from array, aligned by 32
  TDerived& load_a(const void* p) {
    ymm = _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    return derived();
  }
  // Partial load. Load n elements and set the rest to 0
  TDerived& load_partial(std::size_t n, const void* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm = _mm256_maskz_loadu_epi32(__mmask8((1U << n) - 1), p);
#else
    if (n <= 0) {
      *this = 0;
    } else if (n <= 4) {
      *this = x32x8(Half{}.load_partial(n, p), 0);
    } else if (n < 8) {
      *this = x32x8(Half{}.load(p), Half{}.load_partial(n - 4, (const T*)p + 4));
    } else {
      load(p);
    }
#endif
    return derived();
  }
  template<std::size_t tN>
  TDerived& load_partial(const void* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm = _mm256_maskz_loadu_epi32(__mmask8((1U << tN) - 1), p);
#else
    if constexpr (tN <= 0) {
      *this = 0;
    } else if constexpr (tN <= 4) {
      *this = x32x8(Half{}.template load_partial<tN>(p), 0);
    } else if constexpr (tN < 8) {
      *this = x32x8(Half{}.load(p), Half{}.template load_partial<tN - 4>((const T*)p + 4));
    } else {
      load(p);
    }
#endif
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    _mm256_mask_storeu_epi32(p, __mmask8((1U << n) - 1), ymm);
#else
    if (n <= 0) {
      return;
    } else if (n <= 4) {
      get_low().store_partial(n, p);
    } else if (n < 8) {
      get_low().store(p);
      get_high().store_partial(n - 4, (T*)p + 4);
    } else {
      store(p);
    }
#endif
  }
  // cut off vector to n elements. The last 8-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    ymm = _mm256_maskz_mov_epi32(__mmask8((1U << n) - 1), ymm);
#else
    *this = i8x32(*this).cutoff(n * 4);
#endif
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    ymm = _mm256_mask_set1_epi32(ymm, __mmask8(1U << index), i32(value));
#else
    // broadcast value into all elements
    __m256i broad = _mm256_set1_epi32(value);
    static constexpr std::array<i32, 16> maskl{0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0};
    // mask with FFFFFFFF at index position
    __m256i mask = si256{}.load(maskl.data() + 8 - (index & 7U));
    ymm = selectb(mask, broad, ymm);
#endif
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    __m256i x = _mm256_maskz_compress_epi32(__mmask8(1U << index), ymm);
    return T(_mm_cvtsi128_si32(_mm256_castsi256_si128(x)));
#else
    T x[8];
    store(x);
    return x[index & 7];
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  T operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two x32x4:
  [[nodiscard]] Half get_low() const {
    return _mm256_castsi256_si128(ymm);
  }
  [[nodiscard]] Half get_high() const {
    return _mm256_extractf128_si256(ymm, 1);
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
struct NativeVector<i32, 8> : public x32x8<NativeVector<i32, 8>, i32> {
  using x32x8<NativeVector<i32, 8>, i32>::x32x8;
};
using i32x8 = NativeVector<i32, 8>;

template<>
struct NativeVector<u32, 8> : public x32x8<NativeVector<u32, 8>, u32> {
  using x32x8<NativeVector<u32, 8>, u32>::x32x8;
};
using u32x8 = NativeVector<u32, 8>;

template<typename TVec>
concept AnyInt32x8 = std::same_as<TVec, i32x8> || std::same_as<TVec, u32x8>;
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I32X08_HPP
