#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I64X4_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I64X4_HPP

#include <concepts>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i64x2.hpp"
#include "stado/vector/native/types/i8x32.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
template<typename TDerived, typename T>
struct x64x4 : public si256 {
  using Element = T;
  using Half = NativeVector<TDerived, 2>;
  static constexpr std::size_t size = 4;

  // Default constructor:
  x64x4() = default;
  // Constructor to broadcast the same value into all elements:
  x64x4(T i) : si256(_mm256_set1_epi64x(i64(i))) {}
  // Constructor to build from all elements:
  x64x4(T i0, T i1, T i2, T i3) : si256(_mm256_setr_epi64x(i64(i0), i64(i1), i64(i2), i64(i3))) {}
  // Constructor to build from two x64x2:
  x64x4(const Half a0, const Half a1) : si256(_mm256_setr_m128i(a0, a1)) {}
  // Constructor to convert from type __m256i used in intrinsics:
  x64x4(const __m256i x) : si256(x) {}
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
    ymm = _mm256_maskz_loadu_epi64(__mmask8((1U << n) - 1), p);
#else
    if (n <= 0) {
      *this = 0;
    } else if (n <= 2) {
      *this = x64x4(Half{}.load_partial(n, p), 0);
    } else if (n < 4) {
      *this = x64x4(Half{}.load(p), Half{}.load_partial(n - 2, (const T*)p + 2));
    } else {
      load(p);
    }
#endif
    return derived();
  }
  template<std::size_t tN>
  TDerived& load_partial(const void* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm = _mm256_maskz_loadu_epi64(__mmask8((1U << tN) - 1), p);
#else
    if constexpr (tN == 0) {
      *this = 0;
    } else if constexpr (tN <= 2) {
      *this = x64x4(Half{}.template load_partial<tN>(p), 0);
    } else if constexpr (tN < 4) {
      *this = x64x4(Half{}.load(p), Half{}.template load_partial<tN - 2>((const T*)p + 2));
    } else {
      load(p);
    }
#endif
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    _mm256_mask_storeu_epi64(p, __mmask8((1U << n) - 1), ymm);
#else
    if (n <= 0) {
      return;
    } else if (n <= 2) {
      get_low().store_partial(n, p);
    } else if (n < 4) {
      get_low().store(p);
      get_high().store_partial(n - 2, (T*)p + 2);
    } else {
      store(p);
    }
#endif
  }
  // cut off vector to n elements. The last 8-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    ymm = _mm256_maskz_mov_epi64(__mmask8((1U << n) - 1), ymm);
#else
    *this = i8x32(*this).cutoff(n * 8);
#endif
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    ymm = _mm256_mask_set1_epi64(ymm, __mmask8(1U << index), i64(value));
#else
    x64x4 x(value);
    switch (index) {
    case 0: ymm = _mm256_blend_epi32(ymm, x, 0x03); break;
    case 1: ymm = _mm256_blend_epi32(ymm, x, 0x0C); break;
    case 2: ymm = _mm256_blend_epi32(ymm, x, 0x30); break;
    case 3: ymm = _mm256_blend_epi32(ymm, x, 0xC0); break;
    default: break;
    }
#endif
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    __m256i x = _mm256_maskz_compress_epi64(__mmask8(1U << index), ymm);
    return T(_mm_cvtsi128_si64(_mm256_castsi256_si128(x)));
#else
    T x[4];
    store(x);
    return x[index & 3];
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  T operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two x64x2:
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
struct NativeVector<i64, 4> : public x64x4<NativeVector<i64, 4>, i64> {
  using x64x4<NativeVector<i64, 4>, i64>::x64x4;
};
using i64x4 = NativeVector<i64, 4>;

template<>
struct NativeVector<u64, 4> : public x64x4<NativeVector<u64, 4>, u64> {
  using x64x4<NativeVector<u64, 4>, u64>::x64x4;
};
using u64x4 = NativeVector<u64, 4>;

template<typename TVec>
concept AnyInt64x4 = std::same_as<TVec, i64x4> || std::same_as<TVec, u64x4>;
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I64X4_HPP
