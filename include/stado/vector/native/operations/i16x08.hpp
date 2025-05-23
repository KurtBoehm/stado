#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I16X08_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I16X08_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i16x08.hpp"

namespace stado {
// vector operator + : add element by element
inline i16x8 operator+(const i16x8 a, const i16x8 b) {
  return _mm_add_epi16(a, b);
}
// vector operator += : add
inline i16x8& operator+=(i16x8& a, const i16x8 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i16x8 operator++(i16x8& a, int) {
  i16x8 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i16x8& operator++(i16x8& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i16x8 operator-(const i16x8 a, const i16x8 b) {
  return _mm_sub_epi16(a, b);
}
// vector operator - : unary minus
inline i16x8 operator-(const i16x8 a) {
  return _mm_sub_epi16(_mm_setzero_si128(), a);
}
// vector operator -= : subtract
inline i16x8& operator-=(i16x8& a, const i16x8 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i16x8 operator--(i16x8& a, int) {
  i16x8 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i16x8& operator--(i16x8& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i16x8 operator*(const i16x8 a, const i16x8 b) {
  return _mm_mullo_epi16(a, b);
}

// vector operator *= : multiply
inline i16x8& operator*=(i16x8& a, const i16x8 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer. See bottom of file

// vector operator << : shift left
inline i16x8 operator<<(const i16x8 a, i32 b) {
  return _mm_sll_epi16(a, _mm_cvtsi32_si128(b));
}

// vector operator <<= : shift left
inline i16x8& operator<<=(i16x8& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic
inline i16x8 operator>>(const i16x8 a, i32 b) {
  return _mm_sra_epi16(a, _mm_cvtsi32_si128(b));
}

// vector operator >>= : shift right arithmetic
inline i16x8& operator>>=(i16x8& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
template<AnyInt16x8 TVec>
inline Mask<16, 8> operator==(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmpeq_epi16_mask(a, b);
#else
  return _mm_cmpeq_epi16(a, b);
#endif
}

// vector operator != : returns true for elements for which a != b
template<AnyInt16x8 TVec>
inline Mask<16, 8> operator!=(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmpneq_epi16_mask(a, b);
#else
  return Mask<16, 8>(~(a == b));
#endif
}

// vector operator > : returns true for elements for which a > b
inline Mask<16, 8> operator>(const i16x8 a, const i16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi16_mask(a, b, 6);
#else
  return _mm_cmpgt_epi16(a, b);
#endif
}

// vector operator < : returns true for elements for which a < b
inline Mask<16, 8> operator<(const i16x8 a, const i16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi16_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline Mask<16, 8> operator>=(const i16x8 a, const i16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi16_mask(a, b, 5);
#else
  return Mask<16, 8>(~(b > a));
#endif
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline Mask<16, 8> operator<=(const i16x8 a, const i16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi16_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline i16x8 operator&(const i16x8 a, const i16x8 b) {
  return {si128(a) & si128(b)};
}
inline i16x8 operator&&(const i16x8 a, const i16x8 b) {
  return a & b;
}
// vector operator &= : bitwise and
inline i16x8& operator&=(i16x8& a, const i16x8 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i16x8 operator|(const i16x8 a, const i16x8 b) {
  return {si128(a) | si128(b)};
}
inline i16x8 operator||(const i16x8 a, const i16x8 b) {
  return a | b;
}
// vector operator |= : bitwise or
inline i16x8& operator|=(i16x8& a, const i16x8 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i16x8 operator^(const i16x8 a, const i16x8 b) {
  return {si128(a) ^ si128(b)};
}
// vector operator ^= : bitwise xor
inline i16x8& operator^=(i16x8& a, const i16x8 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i16x8 operator~(const i16x8 a) {
  return {~si128(a)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
inline i16x8 select(const Mask<16, 8> s, const i16x8 a, const i16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mov_epi16(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i16x8 if_add(const Mask<16, 8> f, const i16x8 a, const i16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_add_epi16(a, f, a, b);
#else
  return a + (i16x8(f) & b);
#endif
}

// Conditional sub: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
inline i16x8 if_sub(const Mask<16, 8> f, const i16x8 a, const i16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_sub_epi16(a, f, a, b);
#else
  return a - (i16x8(f) & b);
#endif
}

// Conditional mul: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
inline i16x8 if_mul(const Mask<16, 8> f, const i16x8 a, const i16x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mullo_epi16(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i16 horizontal_add(const i16x8 a) {
#ifdef __XOP__ // AMD XOP instruction set
  __m128i sum1 = _mm_haddq_epi16(a);
  __m128i sum2 = _mm_shuffle_epi32(sum1, 0x0E); // high element
  __m128i sum3 = _mm_add_epi32(sum1, sum2); // sum
  i16 sum4 = _mm_cvtsi128_si32(sum3); // truncate to 16 bits
  return sum4; // sign extend to 32 bits
#else // SSE2
  // The hadd instruction is inefficient, and may be split into two instructions for faster decoding
  const __m128i sum1 = _mm_unpackhi_epi64(a, a); // 4 high elements
  const __m128i sum2 = _mm_add_epi16(a, sum1); // 4 sums
  const __m128i sum3 = _mm_shuffle_epi32(sum2, 0x01); // 2 high elements
  const __m128i sum4 = _mm_add_epi16(sum2, sum3); // 2 sums
  const __m128i sum5 = _mm_shufflelo_epi16(sum4, 0x01); // 1 high element
  const __m128i sum6 = _mm_add_epi16(sum4, sum5); // 1 sum
  auto sum7 = i16(_mm_cvtsi128_si32(sum6)); // 16 bit sum
  return sum7; // sign extend to 32 bits
#endif
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are sign extended before adding to avoid overflow
inline i32 horizontal_add_x(const i16x8 a) {
#ifdef __XOP__ // AMD XOP instruction set
  __m128i sum1 = _mm_haddq_epi16(a);
  __m128i sum2 = _mm_shuffle_epi32(sum1, 0x0E); // high element
  __m128i sum3 = _mm_add_epi32(sum1, sum2); // sum
  return _mm_cvtsi128_si32(sum3);
#else
  __m128i aeven = _mm_slli_epi32(a, 16); // even numbered elements of a. get sign bit in position
  aeven = _mm_srai_epi32(aeven, 16); // sign extend even numbered elements
  const __m128i aodd = _mm_srai_epi32(a, 16); // sign extend odd  numbered elements
  const __m128i sum1 = _mm_add_epi32(aeven, aodd); // add even and odd elements
  // The hadd instruction is inefficient, and may be split into two instructions for faster decoding
  const __m128i sum2 = _mm_unpackhi_epi64(sum1, sum1); // 2 high elements
  const __m128i sum3 = _mm_add_epi32(sum1, sum2);
  const __m128i sum4 = _mm_shuffle_epi32(sum3, 1); // 1 high elements
  const __m128i sum5 = _mm_add_epi32(sum3, sum4);
  return _mm_cvtsi128_si32(sum5); // 32 bit sum
#endif
}

// function add_saturated: add element by element, signed with saturation
inline i16x8 add_saturated(const i16x8 a, const i16x8 b) {
  return _mm_adds_epi16(a, b);
}

// function sub_saturated: subtract element by element, signed with saturation
inline i16x8 sub_saturated(const i16x8 a, const i16x8 b) {
  return _mm_subs_epi16(a, b);
}

// function max: a > b ? a : b
inline i16x8 max(const i16x8 a, const i16x8 b) {
  return _mm_max_epi16(a, b);
}

// function min: a < b ? a : b
inline i16x8 min(const i16x8 a, const i16x8 b) {
  return _mm_min_epi16(a, b);
}

// function abs: a >= 0 ? a : -a
inline i16x8 abs(const i16x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  return _mm_abs_epi16(a);
#else // SSE2
  __m128i nega = _mm_sub_epi16(_mm_setzero_si128(), a);
  return _mm_max_epi16(a, nega);
#endif
}

// function abs_saturated: same as abs, saturate if overflow
inline i16x8 abs_saturated(const i16x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_min_epu16(abs(a), i16x8(0x7FFF));
#else
  __m128i absa = abs(a); // abs(a)
  __m128i overfl = _mm_srai_epi16(absa, 15); // sign
  return _mm_add_epi16(absa, overfl); // subtract 1 if 0x8000
#endif
}

// function rotate_left all elements
// Use negative count to rotate right
inline i16x8 rotate_left(const i16x8 a, i32 b) {
#ifdef __XOP__ // AMD XOP instruction set
  return (i16x8)_mm_rot_epi16(a, _mm_set1_epi16(b));
#else // SSE2 instruction set
  const __m128i left = _mm_sll_epi16(a, _mm_cvtsi32_si128(b & 0x0F)); // a << b
  const __m128i right = _mm_srl_epi16(a, _mm_cvtsi32_si128((-b) & 0x0F)); // a >> (16 - b)
  const __m128i rot = _mm_or_si128(left, right); // or
  return rot;
#endif
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I16X08_HPP
