#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I08X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I08X16_HPP

#include <array>
#include <concepts>
#include <cstdint>
#include <cstring>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-128.hpp"

namespace stado {
template<typename TDerived, typename T>
struct x8x16 : public si128 {
  using Value = T;
  static constexpr std::size_t size = 16;

  // Default constructor:
  x8x16() = default;
  // Constructor to broadcast the same value into all elements:
  x8x16(T i) : si128(_mm_set1_epi8(i8(i))) {}
  // Constructor to build from all elements:
  x8x16(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13,
        T v14, T v15)
      : si128(_mm_setr_epi8(i8(v0), i8(v1), i8(v2), i8(v3), i8(v4), i8(v5), i8(v6), i8(v7), i8(v8),
                            i8(v9), i8(v10), i8(v11), i8(v12), i8(v13), i8(v14), i8(v15))) {}
  // Constructor to convert from type __m128i used in intrinsics:
  x8x16(const __m128i x) : si128(x) {}
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
    xmm = _mm_maskz_loadu_epi8(__mmask16((1U << n) - 1), p);
#else
    if (n >= 16) {
      load(p);
      cutoff(n);
    } else if (n <= 0) {
      *this = 0;
    } else {
      // worst case. read 1 byte at a time and suffer store forwarding penalty
      std::array<T, 16> x{};
      std::memcpy(x.data(), p, n);
      load(x.data());
    }
#endif
    return derived();
  }
  template<std::size_t tN>
  TDerived& load_partial(const void* p) {
    static_assert(tN <= 16);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
    xmm = _mm_maskz_loadu_epi8(__mmask16((1U << tN) - 1), p);
#else
    if constexpr (tN == 0) {
      *this = 0;
    } else if constexpr (tN == 1) {
      u8 value{};
      std::memcpy(&value, p, 1);
      xmm = _mm_cvtsi32_si128(value);
    } else if constexpr (tN == 2) {
      u16 pair{};
      std::memcpy(&pair, p, 2);
      xmm = _mm_cvtsi32_si128(pair);
    } else if constexpr (tN == 4) {
      xmm = _mm_castps_si128(_mm_load_ss(reinterpret_cast<const f32*>(p)));
    } else if constexpr (tN == 8) {
      xmm = _mm_castpd_si128(_mm_load_sd(reinterpret_cast<const f64*>(p)));
    } else if constexpr (tN == 16) {
      load(p);
    } else {
      if ((uintptr_t(p) & 0xFFFU) < 0xFF0U) {
        // p is at least 16 bytes from a page boundary. OK to read 16 bytes
        load(p);
      } else {
        // worst case. read 1 byte at a time and suffer store forwarding penalty
        std::array<T, 16> x{};
        std::memcpy(x.data(), p, tN);
        load(x.data());
      }
      cutoff(tN);
    }
#endif
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
    _mm_mask_storeu_epi8(p, __mmask16((1U << n) - 1), xmm);
#else
    if (n >= 16) {
      store(p);
      return;
    }
    if (n <= 0) {
      return;
    }
    // we are not using _mm_maskmoveu_si128 because it is too slow on many processors
    union {
      T c[16];
      i16 s[8];
      i32 i[4];
      i64 q[2];
    } u;
    store(u.c);
    std::size_t j = 0;
    if (n & 8) {
      *reinterpret_cast<i64*>(p) = u.q[0];
      j += 8;
    }
    if (n & 4) {
      reinterpret_cast<i32*>(p)[j / 4] = u.i[j / 4];
      j += 4;
    }
    if (n & 2) {
      reinterpret_cast<i16*>(p)[j / 2] = u.s[j / 2];
      j += 2;
    }
    if (n & 1) {
      ((T*)p)[j] = u.c[j];
    }
#endif
  }

  // cut off vector to n elements. The last 16-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    xmm = _mm_maskz_mov_epi8(__mmask16((1U << n) - 1), xmm);
#else
    x8x16 index_vec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const x8x16 ref_vec{T(n)};
    *this &= _mm_cmpgt_epi8(ref_vec, index_vec);
#endif
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    xmm = _mm_mask_set1_epi8(xmm, __mmask16(1U << index), i8(value));
#else
    static constexpr std::array<i8, 32> maskl{0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    __m128i broad = _mm_set1_epi8(value); // broadcast value into all elements
    __m128i mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
      maskl.data() + 16 - (index & 0x0FU))); // mask with FF at index position
    xmm = selectb(mask, broad, xmm);
#endif
    return derived();
  }

  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI2__)
    const __m128i x = _mm_maskz_compress_epi8(__mmask16(1U << index), xmm);
    return T(_mm_cvtsi128_si32(x));
#else
    T x[16];
    store(x);
    return x[index & 0x0F];
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
struct NativeVector<i8, 16> : public x8x16<NativeVector<i8, 16>, i8> {
  using x8x16<NativeVector<i8, 16>, i8>::x8x16;
};
using i8x16 = NativeVector<i8, 16>;

template<>
struct NativeVector<u8, 16> : public x8x16<NativeVector<u8, 16>, u8> {
  using x8x16<NativeVector<u8, 16>, u8>::x8x16;
};
using u8x16 = NativeVector<u8, 16>;

template<typename TVec>
concept AnyInt8x16 = std::same_as<TVec, i8x16> || std::same_as<TVec, u8x16>;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I08X16_HPP
