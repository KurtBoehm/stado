#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I16X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I16X16_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i16x16.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
// vector operator + : add element by element
inline i16x16 operator+(const i16x16 a, const i16x16 b) {
  return _mm256_add_epi16(a, b);
}
// vector operator += : add
inline i16x16& operator+=(i16x16& a, const i16x16 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i16x16 operator++(i16x16& a, int) {
  i16x16 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i16x16& operator++(i16x16& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i16x16 operator-(const i16x16 a, const i16x16 b) {
  return _mm256_sub_epi16(a, b);
}
// vector operator - : unary minus
inline i16x16 operator-(const i16x16 a) {
  return _mm256_sub_epi16(_mm256_setzero_si256(), a);
}
// vector operator -= : subtract
inline i16x16& operator-=(i16x16& a, const i16x16 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i16x16 operator--(i16x16& a, int) {
  i16x16 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i16x16& operator--(i16x16& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i16x16 operator*(const i16x16 a, const i16x16 b) {
  return _mm256_mullo_epi16(a, b);
}
// vector operator *= : multiply
inline i16x16& operator*=(i16x16& a, const i16x16 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer. See bottom of file

// vector operator << : shift left
inline i16x16 operator<<(const i16x16 a, i32 b) {
  return _mm256_sll_epi16(a, _mm_cvtsi32_si128(b));
}
// vector operator <<= : shift left
inline i16x16& operator<<=(i16x16& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic
inline i16x16 operator>>(const i16x16 a, i32 b) {
  return _mm256_sra_epi16(a, _mm_cvtsi32_si128(b));
}
// vector operator >>= : shift right arithmetic
inline i16x16& operator>>=(i16x16& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
template<AnyInt16x16 TVec>
inline Mask<16, 16> operator==(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi16_mask(a, b, 0);
#else
  return _mm256_cmpeq_epi16(a, b);
#endif
}

// vector operator != : returns true for elements for which a != b
template<AnyInt16x16 TVec>
inline Mask<16, 16> operator!=(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi16_mask(a, b, 4);
#else
  return Mask<16, 16>(i16x16(~(a == b)));
#endif
}

// vector operator > : returns true for elements for which a > b
inline Mask<16, 16> operator>(const i16x16 a, const i16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi16_mask(a, b, 6);
#else
  return _mm256_cmpgt_epi16(a, b);
#endif
}

// vector operator < : returns true for elements for which a < b
inline Mask<16, 16> operator<(const i16x16 a, const i16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi16_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline Mask<16, 16> operator>=(const i16x16 a, const i16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi16_mask(a, b, 5);
#else
  return Mask<16, 16>(i16x16(~(b > a)));
#endif
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline Mask<16, 16> operator<=(const i16x16 a, const i16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi16_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline i16x16 operator&(const i16x16 a, const i16x16 b) {
  return i16x16(si256(a) & si256(b));
}
inline i16x16 operator&&(const i16x16 a, const i16x16 b) {
  return a & b;
}
// vector operator &= : bitwise and
inline i16x16& operator&=(i16x16& a, const i16x16 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i16x16 operator|(const i16x16 a, const i16x16 b) {
  return i16x16(si256(a) | si256(b));
}
inline i16x16 operator||(const i16x16 a, const i16x16 b) {
  return a | b;
}
// vector operator |= : bitwise or
inline i16x16& operator|=(i16x16& a, const i16x16 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i16x16 operator^(const i16x16 a, const i16x16 b) {
  return i16x16(si256(a) ^ si256(b));
}
// vector operator ^= : bitwise xor
inline i16x16& operator^=(i16x16& a, const i16x16 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i16x16 operator~(const i16x16 a) {
  return i16x16(~si256(a));
}

// vector operator ! : logical not, returns true for elements == 0
inline Mask<16, 16> operator!(const i16x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi16_mask(a, _mm256_setzero_si256(), 0);
#else
  return _mm256_cmpeq_epi16(a, _mm256_setzero_si256());
#endif
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
inline i16x16 select(const Mask<16, 16> s, const i16x16 a, const i16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mov_epi16(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i16x16 if_add(const Mask<16, 16> f, const i16x16 a, const i16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_add_epi16(a, f, a, b);
#else
  return a + (i16x16(f) & b);
#endif
}

// Conditional subtract
inline i16x16 if_sub(const Mask<16, 16> f, const i16x16 a, const i16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_sub_epi16(a, f, a, b);
#else
  return a - (i16x16(f) & b);
#endif
}

// Conditional multiply
inline i16x16 if_mul(const Mask<16, 16> f, const i16x16 a, const i16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mullo_epi16(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i16 horizontal_add(const i16x16 a) {
  // The hadd instruction is inefficient, and may be split into two instructions for faster decoding
  const __m128i sum1 = _mm_add_epi16(_mm256_extracti128_si256(a, 1), _mm256_castsi256_si128(a));
  const __m128i sum2 = _mm_add_epi16(sum1, _mm_unpackhi_epi64(sum1, sum1));
  const __m128i sum3 = _mm_add_epi16(sum2, _mm_shuffle_epi32(sum2, 1));
  const __m128i sum4 = _mm_add_epi16(sum3, _mm_shufflelo_epi16(sum3, 1));
  return i16(_mm_cvtsi128_si32(sum4)); // truncate to 16 bits
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are sign extended before adding to avoid overflow
inline i32 horizontal_add_x(const i16x16 a) {
  __m256i aeven = _mm256_slli_epi32(a, 16); // even numbered elements of a. get sign bit in position
  aeven = _mm256_srai_epi32(aeven, 16); // sign extend even numbered elements
  const __m256i aodd = _mm256_srai_epi32(a, 16); // sign extend odd  numbered elements
  const __m256i sum1 = _mm256_add_epi32(aeven, aodd); // add even and odd elements
  const __m128i sum2 =
    _mm_add_epi32(_mm256_extracti128_si256(sum1, 1), _mm256_castsi256_si128(sum1));
  const __m128i sum3 = _mm_add_epi32(sum2, _mm_unpackhi_epi64(sum2, sum2));
  const __m128i sum4 = _mm_add_epi32(sum3, _mm_shuffle_epi32(sum3, 1));
  return i16(_mm_cvtsi128_si32(sum4)); // truncate to 16 bits
}

// function add_saturated: add element by element, signed with saturation
inline i16x16 add_saturated(const i16x16 a, const i16x16 b) {
  return _mm256_adds_epi16(a, b);
}

// function sub_saturated: subtract element by element, signed with saturation
inline i16x16 sub_saturated(const i16x16 a, const i16x16 b) {
  return _mm256_subs_epi16(a, b);
}

// function max: a > b ? a : b
inline i16x16 max(const i16x16 a, const i16x16 b) {
  return _mm256_max_epi16(a, b);
}

// function min: a < b ? a : b
inline i16x16 min(const i16x16 a, const i16x16 b) {
  return _mm256_min_epi16(a, b);
}

// function abs: a >= 0 ? a : -a
inline i16x16 abs(const i16x16 a) {
  return _mm256_abs_epi16(a);
}

// function abs_saturated: same as abs, saturate if overflow
inline i16x16 abs_saturated(const i16x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_min_epu16(abs(a), i16x16(0x7FFF));
#else
  __m256i absa = abs(a); // abs(a)
  __m256i overfl = _mm256_srai_epi16(absa, 15); // sign
  return _mm256_add_epi16(absa, overfl); // subtract 1 if 0x8000
#endif
}

// function rotate_left all elements
// Use negative count to rotate right
inline i16x16 rotate_left(const i16x16 a, i32 b) {
  const __m256i left = _mm256_sll_epi16(a, _mm_cvtsi32_si128(b & 0x0F)); // a << b
  const __m256i right = _mm256_srl_epi16(a, _mm_cvtsi32_si128((-b) & 0x0F)); // a >> (16 - b)
  return _mm256_or_si256(left, right); // or
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I16X16_HPP
