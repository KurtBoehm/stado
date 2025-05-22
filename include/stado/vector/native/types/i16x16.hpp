#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I16X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I16X16_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i16x8.hpp"
#include "stado/vector/native/types/i8x32.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
template<typename TDerived, typename T>
struct x16x16 : public si256 {
  using Element = T;
  using Half = NativeVector<T, 8>;
  static constexpr std::size_t size = 16;

  // Default constructor:
  x16x16() = default;
  // Constructor to broadcast the same value into all elements:
  x16x16(T i) : si256(_mm256_set1_epi16(i16(i))) {}
  // Constructor to build from all elements:
  x16x16(T i0, T i1, T i2, T i3, T i4, T i5, T i6, T i7, T i8, T i9, T i10, T i11, T i12, T i13,
         T i14, T i15)
      : si256(_mm256_setr_epi16(i16(i0), i16(i1), i16(i2), i16(i3), i16(i4), i16(i5), i16(i6),
                                i16(i7), i16(i8), i16(i9), i16(i10), i16(i11), i16(i12), i16(i13),
                                i16(i14), i16(i15))) {}
  // Constructor to build from two x16x8:
  x16x16(const Half a0, const Half a1) : si256(_mm256_setr_m128i(a0, a1)) {}
  // Constructor to convert from type __m256i used in intrinsics:
  x16x16(const __m256i x) : si256(x) {}
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
    ymm = _mm256_maskz_loadu_epi16(__mmask16((1U << n) - 1), p);
#else
    if (n <= 0) {
      *this = 0;
    } else if (n <= 8) {
      *this = x16x16(Half{}.load_partial(n, p), 0);
    } else if (n < 16) {
      *this = x16x16(Half{}.load(p), Half{}.load_partial(n - 8, (const T*)p + 8));
    } else {
      load(p);
    }
#endif
    return derived();
  }
  template<std::size_t tN>
  TDerived& load_partial(const void* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm = _mm256_maskz_loadu_epi16(__mmask16((1U << tN) - 1), p);
#else
    if constexpr (tN <= 0) {
      *this = 0;
    } else if constexpr (tN <= 8) {
      *this = x16x16(Half{}.template load_partial<tN>(p), 0);
    } else if constexpr (tN < 16) {
      *this = x16x16(Half{}.load(p), Half{}.template load_partial<tN - 8>((const T*)p + 8));
    } else {
      load(p);
    }
#endif
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
    _mm256_mask_storeu_epi16(p, __mmask16((1U << n) - 1), ymm);
#else
    if (n <= 0) {
      return;
    }
    if (n <= 8) {
      get_low().store_partial(n, p);
    } else if (n < 16) {
      get_low().store(p);
      get_high().store_partial(n - 8, (T*)p + 8);
    } else {
      store(p);
    }
#endif
  }
  // cut off vector to n elements. The last 16-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    ymm = _mm256_maskz_mov_epi16(__mmask16((1U << n) - 1), ymm);
#else
    *this = x16x16(i8x32(*this).cutoff(n * 2));
#endif
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    ymm = _mm256_mask_set1_epi16(ymm, __mmask16(1U << index), i16(value));
#else
    static constexpr std::array<i16, 32> m{0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    __m256i mask = si256().load(m.data() + 16 - (index & 0x0FU));
    __m256i broad = _mm256_set1_epi16(value);
    ymm = selectb(mask, broad, ymm);
#endif
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI2__)
    __m256i x = _mm256_maskz_compress_epi16(__mmask16(1U << index), ymm);
    return T(_mm_cvtsi128_si32(_mm256_castsi256_si128(x)));
#else
    T x[16]; // find faster version
    store(x);
    return x[index & 0x0F];
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  T operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two x16x8:
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
struct NativeVector<i16, 16> : public x16x16<NativeVector<i16, 16>, i16> {
  using x16x16<NativeVector<i16, 16>, i16>::x16x16;
};
using i16x16 = NativeVector<i16, 16>;

template<>
struct NativeVector<u16, 16> : public x16x16<NativeVector<u16, 16>, u16> {
  using x16x16<NativeVector<u16, 16>, u16>::x16x16;
};
using u16x16 = NativeVector<u16, 16>;
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I16X16_HPP
