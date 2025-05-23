#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I32X08_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I32X08_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i32x08.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
// vector operator + : add element by element
inline i32x8 operator+(const i32x8 a, const i32x8 b) {
  return _mm256_add_epi32(a, b);
}
// vector operator += : add
inline i32x8& operator+=(i32x8& a, const i32x8 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i32x8 operator++(i32x8& a, int) {
  i32x8 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i32x8& operator++(i32x8& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i32x8 operator-(const i32x8 a, const i32x8 b) {
  return _mm256_sub_epi32(a, b);
}
// vector operator - : unary minus
inline i32x8 operator-(const i32x8 a) {
  return _mm256_sub_epi32(_mm256_setzero_si256(), a);
}
// vector operator -= : subtract
inline i32x8& operator-=(i32x8& a, const i32x8 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i32x8 operator--(i32x8& a, int) {
  i32x8 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i32x8& operator--(i32x8& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i32x8 operator*(const i32x8 a, const i32x8 b) {
  return _mm256_mullo_epi32(a, b);
}
// vector operator *= : multiply
inline i32x8& operator*=(i32x8& a, const i32x8 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer. See bottom of file

// vector operator << : shift left
inline i32x8 operator<<(const i32x8 a, i32 b) {
  return _mm256_sll_epi32(a, _mm_cvtsi32_si128(b));
}
// vector operator <<= : shift left
inline i32x8& operator<<=(i32x8& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic
inline i32x8 operator>>(const i32x8 a, i32 b) {
  return _mm256_sra_epi32(a, _mm_cvtsi32_si128(b));
}
// vector operator >>= : shift right arithmetic
inline i32x8& operator>>=(i32x8& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
template<AnyInt32x8 TVec>
inline Mask<32, 8> operator==(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi32_mask(a, b, 0);
#else
  return _mm256_cmpeq_epi32(a, b);
#endif
}

// vector operator != : returns true for elements for which a != b
template<AnyInt32x8 TVec>
inline Mask<32, 8> operator!=(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi32_mask(a, b, 4);
#else
  return ~(a == b);
#endif
}

// vector operator > : returns true for elements for which a > b
inline Mask<32, 8> operator>(const i32x8 a, const i32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi32_mask(a, b, 6);
#else
  return _mm256_cmpgt_epi32(a, b);
#endif
}

// vector operator < : returns true for elements for which a < b
inline Mask<32, 8> operator<(const i32x8 a, const i32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi32_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline Mask<32, 8> operator>=(const i32x8 a, const i32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi32_mask(a, b, 5);
#else
  return ~(b > a);
#endif
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline Mask<32, 8> operator<=(const i32x8 a, const i32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi32_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline i32x8 operator&(const i32x8 a, const i32x8 b) {
  return {si256(a) & si256(b)};
}
inline i32x8 operator&&(const i32x8 a, const i32x8 b) {
  return a & b;
}
// vector operator &= : bitwise and
inline i32x8& operator&=(i32x8& a, const i32x8 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i32x8 operator|(const i32x8 a, const i32x8 b) {
  return {si256(a) | si256(b)};
}
inline i32x8 operator||(const i32x8 a, const i32x8 b) {
  return a | b;
}
// vector operator |= : bitwise or
inline i32x8& operator|=(i32x8& a, const i32x8 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i32x8 operator^(const i32x8 a, const i32x8 b) {
  return {si256(a) ^ si256(b)};
}
// vector operator ^= : bitwise xor
inline i32x8& operator^=(i32x8& a, const i32x8 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i32x8 operator~(const i32x8 a) {
  return {~si256(a)};
}

// vector operator ! : returns true for elements == 0
inline Mask<32, 8> operator!(const i32x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi32_mask(a, _mm256_setzero_si256(), 0);
#else
  return _mm256_cmpeq_epi32(a, _mm256_setzero_si256());
#endif
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
inline i32x8 select(const Mask<32, 8> s, const i32x8 a, const i32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mov_epi32(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i32x8 if_add(const Mask<32, 8> f, const i32x8 a, const i32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_add_epi32(a, f, a, b);
#else
  return a + (i32x8(f) & b);
#endif
}

// Conditional subtract
inline i32x8 if_sub(const Mask<32, 8> f, const i32x8 a, const i32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_sub_epi32(a, f, a, b);
#else
  return a - (i32x8(f) & b);
#endif
}

// Conditional multiply
inline i32x8 if_mul(const Mask<32, 8> f, const i32x8 a, const i32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mullo_epi32(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i32 horizontal_add(const i32x8 a) {
  // The hadd instruction is inefficient, and may be split into two instructions for faster decoding
  const __m128i sum1 = _mm_add_epi32(_mm256_extracti128_si256(a, 1), _mm256_castsi256_si128(a));
  const __m128i sum2 = _mm_add_epi32(sum1, _mm_unpackhi_epi64(sum1, sum1));
  const __m128i sum3 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, 1));
  return i32(_mm_cvtsi128_si32(sum3));
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are sign extended before adding to avoid overflow
// inline i64 horizontal_add_x (i32x8 const a); // defined below

// function add_saturated: add element by element, signed with saturation
inline i32x8 add_saturated(const i32x8 a, const i32x8 b) {
  const __m256i sum = _mm256_add_epi32(a, b); // a + b
  const __m256i axb = _mm256_xor_si256(a, b); // check if a and b have different sign
  const __m256i axs = _mm256_xor_si256(a, sum); // check if a and sum have different sign
  const __m256i overf1 = _mm256_andnot_si256(axb, axs); // check if sum has wrong sign
  const __m256i overf2 = _mm256_srai_epi32(overf1, 31); // -1 if overflow
  const __m256i asign = _mm256_srli_epi32(a, 31); // 1  if a < 0
  const __m256i sat1 = _mm256_srli_epi32(overf2, 1); // 7FFFFFFF if overflow
  const __m256i sat2 =
    _mm256_add_epi32(sat1, asign); // 7FFFFFFF if positive overflow 80000000 if negative overflow
  return selectb(overf2, sat2, sum); // sum if not overflow, else sat2
}

// function sub_saturated: subtract element by element, signed with saturation
inline i32x8 sub_saturated(const i32x8 a, const i32x8 b) {
  const __m256i diff = _mm256_sub_epi32(a, b); // a + b
  const __m256i axb = _mm256_xor_si256(a, b); // check if a and b have different sign
  const __m256i axs = _mm256_xor_si256(a, diff); // check if a and sum have different sign
  const __m256i overf1 = _mm256_and_si256(axb, axs); // check if sum has wrong sign
  const __m256i overf2 = _mm256_srai_epi32(overf1, 31); // -1 if overflow
  const __m256i asign = _mm256_srli_epi32(a, 31); // 1  if a < 0
  const __m256i sat1 = _mm256_srli_epi32(overf2, 1); // 7FFFFFFF if overflow
  const __m256i sat2 =
    _mm256_add_epi32(sat1, asign); // 7FFFFFFF if positive overflow 80000000 if negative overflow
  return selectb(overf2, sat2, diff); // diff if not overflow, else sat2
}

// function max: a > b ? a : b
inline i32x8 max(const i32x8 a, const i32x8 b) {
  return _mm256_max_epi32(a, b);
}

// function min: a < b ? a : b
inline i32x8 min(const i32x8 a, const i32x8 b) {
  return _mm256_min_epi32(a, b);
}

// function abs: a >= 0 ? a : -a
inline i32x8 abs(const i32x8 a) {
  return _mm256_abs_epi32(a);
}

// function abs_saturated: same as abs, saturate if overflow
inline i32x8 abs_saturated(const i32x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_min_epu32(abs(a), i32x8(0x7FFFFFFF));
#else
  __m256i absa = abs(a); // abs(a)
  __m256i overfl = _mm256_srai_epi32(absa, 31); // sign
  return _mm256_add_epi32(absa, overfl); // subtract 1 if 0x80000000
#endif
}

// function rotate_left all elements
// Use negative count to rotate right
inline i32x8 rotate_left(const i32x8 a, i32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm256_rolv_epi32(a, _mm256_set1_epi32(b));
#else
  __m256i left = _mm256_sll_epi32(a, _mm_cvtsi32_si128(b & 0x1F)); // a << b
  __m256i right = _mm256_srl_epi32(a, _mm_cvtsi32_si128((-b) & 0x1F)); // a >> (32 - b)
  __m256i rot = _mm256_or_si256(left, right); // or
  return rot;
#endif
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I32X08_HPP
