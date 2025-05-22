#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I16X8_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I16X8_HPP

#include <cstddef>
#include <cstring>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i8x16.hpp"

namespace stado {
template<typename TDerived, typename T>
struct x16x8 : public si128 {
  using Element = T;
  static constexpr std::size_t size = 8;

  // Default constructor:
  x16x8() = default;
  // Constructor to broadcast the same value into all elements:
  x16x8(T i) : si128(_mm_set1_epi16(i16(i))) {}
  // Constructor to build from all elements:
  x16x8(T i0, T i1, T i2, T i3, T i4, T i5, T i6, T i7)
      : si128(
          _mm_setr_epi16(i16(i0), i16(i1), i16(i2), i16(i3), i16(i4), i16(i5), i16(i6), i16(i7))) {}
  // Constructor to convert from type __m128i used in intrinsics:
  x16x8(const __m128i x) : si128(x) {}
  // Assignment operator to convert from type __m128i used in intrinsics:
  TDerived& operator=(const __m128i x) {
    xmm = x;
    return derived();
  }
  // Type cast operator to convert to __m128i used in intrinsics
  operator __m128i() const {
    return xmm;
  }
  // Member function to load from array (unaligned)
  TDerived& load(const void* p) {
    xmm = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return derived();
  }
  // Member function to load from array (aligned)
  TDerived& load_a(const void* p) {
    xmm = _mm_load_si128(reinterpret_cast<const __m128i*>(p));
    return derived();
  }
  // Partial load. Load n elements and set the rest to 0
  TDerived& load_partial(std::size_t n, const void* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
    xmm = _mm_maskz_loadu_epi16(__mmask8((1U << n) - 1), p);
#else
    if (n >= 8) {
      load(p);
    } else if (n <= 0) {
      *this = 0;
    } else if ((uintptr_t(p) & 0xFFFU) < 0xFF0U) {
      // p is at least 16 bytes from a page boundary. OK to read 16 bytes
      load(p);
    } else {
      // worst case. read 1 byte at a time and suffer store forwarding penalty
      T x[8];
      std::memcpy(x, p, 2 * n);
      load(x);
    }
    cutoff(n);
#endif
    return derived();
  }
  template<std::size_t tN>
  TDerived& load_partial(const void* p) {
    static_assert(tN <= 8);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
    xmm = _mm_maskz_loadu_epi16(__mmask8((1U << tN) - 1), p);
#else
    if constexpr (tN == 0) {
      *this = 0;
    } else if constexpr (tN == 2) {
      xmm = _mm_castps_si128(_mm_load_ss(reinterpret_cast<const f32*>(p)));
    } else if constexpr (tN == 4) {
      xmm = _mm_castpd_si128(_mm_load_sd(reinterpret_cast<const f64*>(p)));
    } else if constexpr (tN == 8) {
      load(p);
    } else {
      if ((uintptr_t(p) & 0xFFFU) < 0xFF0U) {
        // p is at least 16 bytes from a page boundary. OK to read 16 bytes
        load(p);
      } else {
        // worst case. read 1 byte at a time and suffer store forwarding penalty
        T x[8];
        for (int i = 0; i < tN; i++) {
          x[i] = (reinterpret_cast<const T*>(p))[i];
        }
        load(x);
      }
      cutoff(tN);
    }
#endif
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
    _mm_mask_storeu_epi16(p, __mmask8((1U << n) - 1), xmm);
#else
    if (n >= 8) {
      store(p);
      return;
    }
    if (n <= 0) {
      return;
    }
    // we are not using _mm_maskmoveu_si128 because it is too slow on many processors
    union {
      i8 c[16];
      T s[8];
      i32 i[4];
      i64 q[2];
    } u;
    store(u.c);
    int j = 0;
    if ((n & 4U) != 0) {
      *reinterpret_cast<i64*>(p) = u.q[0];
      j += 8;
    }
    if ((n & 2U) != 0) {
      (reinterpret_cast<i32*>(p))[j / 4] = u.i[j / 4];
      j += 4;
    }
    if ((n & 1U) != 0) {
      (reinterpret_cast<T*>(p))[j / 2] = u.s[j / 2];
    }
#endif
  }

  // cut off vector to n elements. The last 8-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    xmm = _mm_maskz_mov_epi16(__mmask8((1U << n) - 1), xmm);
#else
    *this = i8x16(xmm).cutoff(n * 2);
#endif
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    xmm = _mm_mask_set1_epi16(xmm, __mmask8(1U << index), i16(value));
#else
    switch (index) {
    case 0: xmm = _mm_insert_epi16(xmm, value, 0); break;
    case 1: xmm = _mm_insert_epi16(xmm, value, 1); break;
    case 2: xmm = _mm_insert_epi16(xmm, value, 2); break;
    case 3: xmm = _mm_insert_epi16(xmm, value, 3); break;
    case 4: xmm = _mm_insert_epi16(xmm, value, 4); break;
    case 5: xmm = _mm_insert_epi16(xmm, value, 5); break;
    case 6: xmm = _mm_insert_epi16(xmm, value, 6); break;
    case 7: xmm = _mm_insert_epi16(xmm, value, 7); break;
    default: break;
    }
#endif
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI2__)
    __m128i x = _mm_maskz_compress_epi16(__mmask8(1U << index), xmm);
    return T(_mm_cvtsi128_si32(x));
#else
    switch (index) {
    case 0: return T(_mm_extract_epi16(xmm, 0));
    case 1: return T(_mm_extract_epi16(xmm, 1));
    case 2: return T(_mm_extract_epi16(xmm, 2));
    case 3: return T(_mm_extract_epi16(xmm, 3));
    case 4: return T(_mm_extract_epi16(xmm, 4));
    case 5: return T(_mm_extract_epi16(xmm, 5));
    case 6: return T(_mm_extract_epi16(xmm, 6));
    case 7: return T(_mm_extract_epi16(xmm, 7));
    default: break;
    }
    return 0;
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  T operator[](std::size_t index) const {
    return extract(index);
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
struct NativeVector<i16, 8> : public x16x8<NativeVector<i16, 8>, i16> {
  using x16x8<NativeVector<i16, 8>, i16>::x16x8;
};
using i16x8 = NativeVector<i16, 8>;

template<>
struct NativeVector<u16, 8> : public x16x8<NativeVector<u16, 8>, u16> {
  using x16x8<NativeVector<u16, 8>, u16>::x16x8;
};
using u16x8 = NativeVector<u16, 8>;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I16X8_HPP
