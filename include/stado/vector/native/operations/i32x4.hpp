#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I32X4_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I32X4_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i32x4.hpp"

namespace stado {
// vector operator + : add element by element
inline i32x4 operator+(const i32x4 a, const i32x4 b) {
  return _mm_add_epi32(a, b);
}
// vector operator += : add
inline i32x4& operator+=(i32x4& a, const i32x4 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i32x4 operator++(i32x4& a, int) {
  i32x4 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i32x4& operator++(i32x4& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i32x4 operator-(const i32x4 a, const i32x4 b) {
  return _mm_sub_epi32(a, b);
}
// vector operator - : unary minus
inline i32x4 operator-(const i32x4 a) {
  return _mm_sub_epi32(_mm_setzero_si128(), a);
}
// vector operator -= : subtract
inline i32x4& operator-=(i32x4& a, const i32x4 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i32x4 operator--(i32x4& a, int) {
  i32x4 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i32x4& operator--(i32x4& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i32x4 operator*(const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_mullo_epi32(a, b);
#else
  __m128i a13 = _mm_shuffle_epi32(a, 0xF5); // (-,a3,-,a1)
  __m128i b13 = _mm_shuffle_epi32(b, 0xF5); // (-,b3,-,b1)
  __m128i prod02 = _mm_mul_epu32(a, b); // (-,a2*b2,-,a0*b0)
  __m128i prod13 = _mm_mul_epu32(a13, b13); // (-,a3*b3,-,a1*b1)
  __m128i prod01 = _mm_unpacklo_epi32(prod02, prod13); // (-,-,a1*b1,a0*b0)
  __m128i prod23 = _mm_unpackhi_epi32(prod02, prod13); // (-,-,a3*b3,a2*b2)
  return _mm_unpacklo_epi64(prod01, prod23); // (ab3,ab2,ab1,ab0)
#endif
}

// vector operator *= : multiply
inline i32x4& operator*=(i32x4& a, const i32x4 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer. See bottom of file

// vector operator << : shift left
inline i32x4 operator<<(const i32x4 a, i32 b) {
  return _mm_sll_epi32(a, _mm_cvtsi32_si128(b));
}
// vector operator <<= : shift left
inline i32x4& operator<<=(i32x4& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic
inline i32x4 operator>>(const i32x4 a, i32 b) {
  return _mm_sra_epi32(a, _mm_cvtsi32_si128(b));
}
// vector operator >>= : shift right arithmetic
inline i32x4& operator>>=(i32x4& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
template<AnyInt32x4 TVec>
inline Mask<32, 4> operator==(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi32_mask(a, b, 0);
#else
  return _mm_cmpeq_epi32(a, b);
#endif
}

// vector operator != : returns true for elements for which a != b
template<AnyInt32x4 TVec>
inline Mask<32, 4> operator!=(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi32_mask(a, b, 4);
#else
  return Mask<32, 4>(i32x4(~(a == b)));
#endif
}

// vector operator > : returns true for elements for which a > b
inline Mask<32, 4> operator>(const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi32_mask(a, b, 6);
#else
  return _mm_cmpgt_epi32(a, b);
#endif
}

// vector operator < : returns true for elements for which a < b
inline Mask<32, 4> operator<(const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi32_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline Mask<32, 4> operator>=(const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi32_mask(a, b, 5);
#else
  return Mask<32, 4>(i32x4(~(b > a)));
#endif
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline Mask<32, 4> operator<=(const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi32_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline i32x4 operator&(const i32x4 a, const i32x4 b) {
  return {si128(a) & si128(b)};
}
inline i32x4 operator&&(const i32x4 a, const i32x4 b) {
  return a & b;
}
// vector operator &= : bitwise and
inline i32x4& operator&=(i32x4& a, const i32x4 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i32x4 operator|(const i32x4 a, const i32x4 b) {
  return {si128(a) | si128(b)};
}
inline i32x4 operator||(const i32x4 a, const i32x4 b) {
  return a | b;
}
// vector operator |= : bitwise and
inline i32x4& operator|=(i32x4& a, const i32x4 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i32x4 operator^(const i32x4 a, const i32x4 b) {
  return {si128(a) ^ si128(b)};
}
// vector operator ^= : bitwise and
inline i32x4& operator^=(i32x4& a, const i32x4 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i32x4 operator~(const i32x4 a) {
  return {~si128(a)};
}

// vector operator ! : returns true for elements == 0
inline Mask<32, 4> operator!(const i32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi32_mask(a, _mm_setzero_si128(), 0);
#else
  return _mm_cmpeq_epi32(a, _mm_setzero_si128());
#endif
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
// (s is signed)
inline i32x4 select(const Mask<32, 4> s, const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mov_epi32(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i32x4 if_add(const Mask<32, 4> f, const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_add_epi32(a, f, a, b);
#else
  return a + (i32x4(f) & b);
#endif
}

// Conditional sub: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
inline i32x4 if_sub(const Mask<32, 4> f, const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_sub_epi32(a, f, a, b);
#else
  return a - (i32x4(f) & b);
#endif
}

// Conditional mul: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
inline i32x4 if_mul(const Mask<32, 4> f, const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mullo_epi32(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i32 horizontal_add(const i32x4 a) {
#ifdef __XOP__ // AMD XOP instruction set
  __m128i sum1 = _mm_haddq_epi32(a);
  __m128i sum2 = _mm_shuffle_epi32(sum1, 0x0E); // high element
  __m128i sum3 = _mm_add_epi32(sum1, sum2); // sum
  return _mm_cvtsi128_si32(sum3); // truncate to 32 bits
#else // SSE2
  // The hadd instruction is inefficient, and may be split into two instructions for faster decoding
  const __m128i sum1 = _mm_unpackhi_epi64(a, a); // 2 high elements
  const __m128i sum2 = _mm_add_epi32(a, sum1); // 2 sums
  const __m128i sum3 = _mm_shuffle_epi32(sum2, 0x01); // 1 high element
  const __m128i sum4 = _mm_add_epi32(sum2, sum3); // 2 sums
  return _mm_cvtsi128_si32(sum4); // 32 bit sum
#endif
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are sign extended before adding to avoid overflow
inline i64 horizontal_add_x(const i32x4 a) {
#ifdef __XOP__ // AMD XOP instruction set
  __m128i sum1 = _mm_haddq_epi32(a);
#else // SSE2
  const __m128i signs = _mm_srai_epi32(a, 31); // sign of all elements
  const __m128i a01 = _mm_unpacklo_epi32(a, signs); // sign-extended a0, a1
  const __m128i a23 = _mm_unpackhi_epi32(a, signs); // sign-extended a2, a3
  const __m128i sum1 = _mm_add_epi64(a01, a23); // add
#endif
  const __m128i sum2 = _mm_unpackhi_epi64(sum1, sum1); // high qword
  const __m128i sum3 = _mm_add_epi64(sum1, sum2); // add
  return _mm_cvtsi128_si64(sum3);
}

// function add_saturated: add element by element, signed with saturation
inline i32x4 add_saturated(const i32x4 a, const i32x4 b) {
  // is there a faster method?
  const __m128i sum = _mm_add_epi32(a, b); // a + b
  const __m128i axb = _mm_xor_si128(a, b); // check if a and b have different sign
  const __m128i axs = _mm_xor_si128(a, sum); // check if a and sum have different sign
  const __m128i overf1 = _mm_andnot_si128(axb, axs); // check if sum has wrong sign
  const __m128i overf2 = _mm_srai_epi32(overf1, 31); // -1 if overflow
  const __m128i asign = _mm_srli_epi32(a, 31); // 1  if a < 0
  const __m128i sat1 = _mm_srli_epi32(overf2, 1); // 7FFFFFFF if overflow
  const __m128i sat2 =
    _mm_add_epi32(sat1, asign); // 7FFFFFFF if positive overflow 80000000 if negative overflow
  return selectb(overf2, sat2, sum); // sum if not overflow, else sat2
}

// function sub_saturated: subtract element by element, signed with saturation
inline i32x4 sub_saturated(const i32x4 a, const i32x4 b) {
  const __m128i diff = _mm_sub_epi32(a, b); // a + b
  const __m128i axb = _mm_xor_si128(a, b); // check if a and b have different sign
  const __m128i axs = _mm_xor_si128(a, diff); // check if a and sum have different sign
  const __m128i overf1 = _mm_and_si128(axb, axs); // check if sum has wrong sign
  const __m128i overf2 = _mm_srai_epi32(overf1, 31); // -1 if overflow
  const __m128i asign = _mm_srli_epi32(a, 31); // 1  if a < 0
  const __m128i sat1 = _mm_srli_epi32(overf2, 1); // 7FFFFFFF if overflow
  const __m128i sat2 =
    _mm_add_epi32(sat1, asign); // 7FFFFFFF if positive overflow 80000000 if negative overflow
  return selectb(overf2, sat2, diff); // diff if not overflow, else sat2
}

// function max: a > b ? a : b
inline i32x4 max(const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_max_epi32(a, b);
#else
  __m128i greater = _mm_cmpgt_epi32(a, b);
  return selectb(greater, a, b);
#endif
}

// function min: a < b ? a : b
inline i32x4 min(const i32x4 a, const i32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_min_epi32(a, b);
#else
  __m128i greater = _mm_cmpgt_epi32(a, b);
  return selectb(greater, b, a);
#endif
}

// function abs: a >= 0 ? a : -a
inline i32x4 abs(const i32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  return _mm_abs_epi32(a);
#else // SSE2
  __m128i sign = _mm_srai_epi32(a, 31); // sign of a
  __m128i inv = _mm_xor_si128(a, sign); // invert bits if negative
  return _mm_sub_epi32(inv, sign); // add 1
#endif
}

// function abs_saturated: same as abs, saturate if overflow
inline i32x4 abs_saturated(const i32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_min_epu32(abs(a), i32x4(0x7FFFFFFF));
#else
  __m128i absa = abs(a); // abs(a)
  __m128i overfl = _mm_srai_epi32(absa, 31); // sign
  return _mm_add_epi32(absa, overfl); // subtract 1 if 0x80000000
#endif
}

// function rotate_left all elements
// Use negative count to rotate right
inline i32x4 rotate_left(const i32x4 a, i32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm_rolv_epi32(a, _mm_set1_epi32(b));
#elif defined(__XOP__) // AMD XOP instruction set
  return _mm_rot_epi32(a, _mm_set1_epi32(b));
#else // SSE2
  __m128i left = _mm_sll_epi32(a, _mm_cvtsi32_si128(b & 0x1F)); // a << b
  __m128i right = _mm_srl_epi32(a, _mm_cvtsi32_si128((-b) & 0x1F)); // a >> (32 - b)
  __m128i rot = _mm_or_si128(left, right); // or
  return rot;
#endif
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I32X4_HPP
