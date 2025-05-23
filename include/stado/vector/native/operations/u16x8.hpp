#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U16X8_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U16X8_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i16x8.hpp"

namespace stado {
// vector operator + : add
inline u16x8 operator+(const u16x8 a, const u16x8 b) {
  return u16x8(i16x8(a) + i16x8(b));
}

// vector operator - : subtract
inline u16x8 operator-(const u16x8 a, const u16x8 b) {
  return u16x8(i16x8(a) - i16x8(b));
}

// vector operator * : multiply
inline u16x8 operator*(const u16x8 a, const u16x8 b) {
  return u16x8(i16x8(a) * i16x8(b));
}

// vector operator / : divide
// See bottom of file

// vector operator >> : shift right logical all elements
inline u16x8 operator>>(const u16x8 a, u32 b) {
  return _mm_srl_epi16(a, _mm_cvtsi32_si128(i32(b)));
}

// vector operator >> : shift right logical all elements
inline u16x8 operator>>(const u16x8 a, i32 b) {
  return a >> u32(b);
}

// vector operator >>= : shift right logical
inline u16x8& operator>>=(u16x8& a, int b) {
  a = a >> b;
  return a;
}

// vector operator << : shift left all elements
inline u16x8 operator<<(const u16x8 a, u32 b) {
  return _mm_sll_epi16(a, _mm_cvtsi32_si128(i32(b)));
}

// vector operator << : shift left all elements
inline u16x8 operator<<(const u16x8 a, i32 b) {
  return a << u32(b);
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline Mask<16, 8> operator>=(const u16x8 a, const u16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // broad boolean vectors
  return _mm_cmp_epu16_mask(a, b, 5);
#elif defined(__XOP__) // AMD XOP instruction set
  return (Mask<16, 8>)_mm_comge_epu16(a, b);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
  __m128i max_ab = _mm_max_epu16(a, b); // max(a,b), unsigned
  return _mm_cmpeq_epi16(a, max_ab); // a == max(a,b)
#else // SSE2 instruction set
  __m128i s = _mm_subs_epu16(b, a); // b-a, saturated
  return _mm_cmpeq_epi16(s, _mm_setzero_si128()); // s == 0
#endif
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline Mask<16, 8> operator<=(const u16x8 a, const u16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // broad boolean vectors
  return _mm_cmp_epu16_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline Mask<16, 8> operator>(const u16x8 a, const u16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // broad boolean vectors
  return _mm_cmp_epu16_mask(a, b, 6);
#elif defined(__XOP__) // AMD XOP instruction set
  return (Mask<16, 8>)_mm_comgt_epu16(a, b);
#else // SSE2 instruction set
  return Mask<16, 8>(~(b >= a));
#endif
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline Mask<16, 8> operator<(const u16x8 a, const u16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // broad boolean vectors
  return _mm_cmp_epu16_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator & : bitwise and
inline u16x8 operator&(const u16x8 a, const u16x8 b) {
  return u16x8(si128(a) & si128(b));
}
inline u16x8 operator&&(const u16x8 a, const u16x8 b) {
  return a & b;
}

// vector operator | : bitwise or
inline u16x8 operator|(const u16x8 a, const u16x8 b) {
  return u16x8(si128(a) | si128(b));
}
inline u16x8 operator||(const u16x8 a, const u16x8 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
inline u16x8 operator^(const u16x8 a, const u16x8 b) {
  return u16x8(si128(a) ^ si128(b));
}

// vector operator ~ : bitwise not
inline u16x8 operator~(const u16x8 a) {
  return u16x8(~si128(a));
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
inline u16x8 select(const Mask<16, 8> s, const u16x8 a, const u16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mov_epi16(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u16x8 if_add(const Mask<16, 8> f, const u16x8 a, const u16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_add_epi16(a, f, a, b);
#else
  return a + (u16x8(f) & b);
#endif
}

// Conditional sub: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
inline u16x8 if_sub(const Mask<16, 8> f, const u16x8 a, const u16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_sub_epi16(a, f, a, b);
#else
  return a - (u16x8(f) & b);
#endif
}

// Conditional mul: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
inline u16x8 if_mul(const Mask<16, 8> f, const u16x8 a, const u16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mullo_epi16(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
inline u32 horizontal_add(const u16x8 a) {
#ifdef __XOP__ // AMD XOP instruction set
  __m128i sum1 = _mm_haddq_epu16(a);
  __m128i sum2 = _mm_shuffle_epi32(sum1, 0x0E); // high element
  __m128i sum3 = _mm_add_epi32(sum1, sum2); // sum
  u16 sum4 = _mm_cvtsi128_si32(sum3); // truncate to 16 bits
  return sum4; // zero extend to 32 bits
#else // SSE2
  // The hadd instruction is inefficient, and may be split into two instructions for faster decoding
  const __m128i sum1 = _mm_unpackhi_epi64(a, a); // 4 high elements
  const __m128i sum2 = _mm_add_epi16(a, sum1); // 4 sums
  const __m128i sum3 = _mm_shuffle_epi32(sum2, 0x01); // 2 high elements
  const __m128i sum4 = _mm_add_epi16(sum2, sum3); // 2 sums
  const __m128i sum5 = _mm_shufflelo_epi16(sum4, 0x01); // 1 high element
  const __m128i sum6 = _mm_add_epi16(sum4, sum5); // 1 sum
  const auto sum7 = u16(_mm_cvtsi128_si32(sum6)); // 16 bit sum
  return sum7; // zero extend to 32 bits
#endif
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is zero-extended before addition to avoid overflow
inline u32 horizontal_add_x(const u16x8 a) {
#ifdef __XOP__ // AMD XOP instruction set
  __m128i sum1 = _mm_haddq_epu16(a);
  __m128i sum2 = _mm_shuffle_epi32(sum1, 0x0E); // high element
  __m128i sum3 = _mm_add_epi32(sum1, sum2); // sum
  return u32(_mm_cvtsi128_si32(sum3));
#else
  // The hadd instruction is inefficient, and may be split into two instructions for faster decoding
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
  const __m128i aeven = _mm_maskz_mov_epi16(0x55, a);
#else
  __m128i mask = _mm_set1_epi32(0x0000FFFF); // mask for even positions
  __m128i aeven = _mm_and_si128(a, mask); // even numbered elements of a
#endif
  const __m128i aodd = _mm_srli_epi32(a, 16); // zero extend odd numbered elements
  const __m128i sum1 = _mm_add_epi32(aeven, aodd); // add even and odd elements
  const __m128i sum2 = _mm_unpackhi_epi64(sum1, sum1); // 2 high elements
  const __m128i sum3 = _mm_add_epi32(sum1, sum2);
  const __m128i sum4 = _mm_shuffle_epi32(sum3, 0x01); // 1 high elements
  const __m128i sum5 = _mm_add_epi32(sum3, sum4);
  return u32(_mm_cvtsi128_si32(sum5)); // 16 bit sum
#endif
}

// function add_saturated: add element by element, unsigned with saturation
inline u16x8 add_saturated(const u16x8 a, const u16x8 b) {
  return _mm_adds_epu16(a, b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
inline u16x8 sub_saturated(const u16x8 a, const u16x8 b) {
  return _mm_subs_epu16(a, b);
}

// function max: a > b ? a : b
inline u16x8 max(const u16x8 a, const u16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1 // SSE4.1
  return _mm_max_epu16(a, b);
#else // SSE2
  __m128i signbit = _mm_set1_epi32(0x80008000);
  __m128i a1 = _mm_xor_si128(a, signbit); // add 0x8000
  __m128i b1 = _mm_xor_si128(b, signbit); // add 0x8000
  __m128i m1 = _mm_max_epi16(a1, b1); // signed max
  return _mm_xor_si128(m1, signbit); // sub 0x8000
#endif
}

// function min: a < b ? a : b
inline u16x8 min(const u16x8 a, const u16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1 // SSE4.1
  return _mm_min_epu16(a, b);
#else // SSE2
  __m128i signbit = _mm_set1_epi32(0x80008000);
  __m128i a1 = _mm_xor_si128(a, signbit); // add 0x8000
  __m128i b1 = _mm_xor_si128(b, signbit); // add 0x8000
  __m128i m1 = _mm_min_epi16(a1, b1); // signed min
  return _mm_xor_si128(m1, signbit); // sub 0x8000
#endif
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U16X8_HPP
