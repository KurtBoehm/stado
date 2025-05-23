#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I08X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I08X16_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i08x16.hpp"

namespace stado {
// vector operator + : add element by element
inline i8x16 operator+(const i8x16 a, const i8x16 b) {
  return _mm_add_epi8(a, b);
}
// vector operator += : add
inline i8x16& operator+=(i8x16& a, const i8x16 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i8x16 operator++(i8x16& a, int) {
  i8x16 a0 = a;
  a = a + 1;
  return a0;
}

// prefix operator ++
inline i8x16& operator++(i8x16& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i8x16 operator-(const i8x16 a, const i8x16 b) {
  return _mm_sub_epi8(a, b);
}
// vector operator - : unary minus
inline i8x16 operator-(const i8x16 a) {
  return _mm_sub_epi8(_mm_setzero_si128(), a);
}
// vector operator -= : add
inline i8x16& operator-=(i8x16& a, const i8x16 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i8x16 operator--(i8x16& a, int) {
  i8x16 a0 = a;
  a = a - 1;
  return a0;
}

// prefix operator --
inline i8x16& operator--(i8x16& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i8x16 operator*(const i8x16 a, const i8x16 b) {
  // There is no 8-bit multiply in SSE2. Split into two 16-bit multiplies
  const __m128i aodd = _mm_srli_epi16(a, 8); // odd numbered elements of a
  const __m128i bodd = _mm_srli_epi16(b, 8); // odd numbered elements of b
  const __m128i muleven = _mm_mullo_epi16(a, b); // product of even numbered elements
  __m128i mulodd = _mm_mullo_epi16(aodd, bodd); // product of odd  numbered elements
  mulodd = _mm_slli_epi16(mulodd, 8); // put odd numbered elements back in place
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
  return _mm_mask_mov_epi8(mulodd, 0x5555, muleven);
#else
  __m128i mask = _mm_set1_epi32(0x00FF00FF); // mask for even positions
  return selectb(mask, muleven, mulodd); // interleave even and odd
#endif
}

// vector operator *= : multiply
inline i8x16& operator*=(i8x16& a, const i8x16 b) {
  a = a * b;
  return a;
}

// vector operator << : shift left all elements
inline i8x16 operator<<(const i8x16 a, i32 b) {
  const u32 mask = u32{0xFF} >> u32(b); // mask to remove bits that are shifted out
  const __m128i am = _mm_and_si128(a, _mm_set1_epi8(i8(mask))); // remove bits that will overflow
  const __m128i res = _mm_sll_epi16(am, _mm_cvtsi32_si128(b)); // 16-bit shifts
  return res;
}
// vector operator <<= : shift left
inline i8x16& operator<<=(i8x16& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic all elements
inline i8x16 operator>>(const i8x16 a, i32 b) {
  __m128i aeven = _mm_slli_epi16(a, 8); // even numbered elements of a. get sign bit in position
  aeven = _mm_sra_epi16(aeven, _mm_cvtsi32_si128(b + 8)); // shift arithmetic, back to position
  const __m128i aodd =
    _mm_sra_epi16(a, _mm_cvtsi32_si128(b)); // shift odd numbered elements arithmetic
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
  return _mm_mask_mov_epi8(aodd, 0x5555, aeven);
#else
  __m128i mask = _mm_set1_epi32(0x00FF00FF); // mask for even positions
  __m128i res = selectb(mask, aeven, aodd); // interleave even and odd
  return res;
#endif
}
// vector operator >>= : shift right arithmetic
inline i8x16& operator>>=(i8x16& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
inline Mask<8, 16> operator==(const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi8_mask(a, b, 0);
#else
  return _mm_cmpeq_epi8(a, b);
#endif
}

// vector operator != : returns true for elements for which a != b
inline Mask<8, 16> operator!=(const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi8_mask(a, b, 4);
#else
  return Mask<8, 16>(i8x16(~(a == b)));
#endif
}

// vector operator > : returns true for elements for which a > b (signed)
inline Mask<8, 16> operator>(const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi8_mask(a, b, 6);
#else
  return _mm_cmpgt_epi8(a, b);
#endif
}

// vector operator < : returns true for elements for which a < b (signed)
inline Mask<8, 16> operator<(const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi8_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline Mask<8, 16> operator>=(const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi8_mask(a, b, 5);
#else
  return Mask<8, 16>(i8x16(~(b > a)));
#endif
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline Mask<8, 16> operator<=(const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi8_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline i8x16 operator&(const i8x16 a, const i8x16 b) {
  return {si128(a) & si128(b)};
}
inline i8x16 operator&&(const i8x16 a, const i8x16 b) {
  return a & b;
}
// vector operator &= : bitwise and
inline i8x16& operator&=(i8x16& a, const i8x16 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i8x16 operator|(const i8x16 a, const i8x16 b) {
  return {si128(a) | si128(b)};
}
inline i8x16 operator||(const i8x16 a, const i8x16 b) {
  return a | b;
}
// vector operator |= : bitwise or
inline i8x16& operator|=(i8x16& a, const i8x16 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i8x16 operator^(const i8x16 a, const i8x16 b) {
  return {si128(a) ^ si128(b)};
}
// vector operator ^= : bitwise xor
inline i8x16& operator^=(i8x16& a, const i8x16 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i8x16 operator~(const i8x16 a) {
  return {~si128(a)};
}

// vector operator ! : logical not, returns true for elements == 0
inline Mask<8, 16> operator!(const i8x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi8_mask(a, _mm_setzero_si128(), 0);
#else
  return _mm_cmpeq_epi8(a, _mm_setzero_si128());
#endif
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or -1 (true). No other values are allowed.
inline i8x16 select(const Mask<8, 16> s, const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mov_epi8(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i8x16 if_add(const Mask<8, 16> f, const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_add_epi8(a, f, a, b);
#else
  return a + (i8x16(f) & b);
#endif
}

// Conditional sub: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
inline i8x16 if_sub(const Mask<8, 16> f, const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_sub_epi8(a, f, a, b);
#else
  return a - (i8x16(f) & b);
#endif
}

// Conditional mul: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
inline i8x16 if_mul(const Mask<8, 16> f, const i8x16 a, const i8x16 b) {
  return select(f, a * b, a);
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i32 horizontal_add(const i8x16 a) {
  const __m128i sum1 = _mm_sad_epu8(a, _mm_setzero_si128());
  const __m128i sum2 = _mm_unpackhi_epi64(sum1, sum1);
  const __m128i sum3 = _mm_add_epi16(sum1, sum2);
  auto sum4 = i8(_mm_cvtsi128_si32(sum3)); // truncate to 8 bits
  return sum4; // sign extend to 32 bits
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is sign-extended before addition to avoid overflow
inline i32 horizontal_add_x(const i8x16 a) {
#ifdef __XOP__ // AMD XOP instruction set
  __m128i sum1 = _mm_haddq_epi8(a);
  __m128i sum2 = _mm_shuffle_epi32(sum1, 0x0E); // high element
  __m128i sum3 = _mm_add_epi32(sum1, sum2); // sum
  return _mm_cvtsi128_si32(sum3);
#else
  __m128i aeven = _mm_slli_epi16(a, 8); // even numbered elements of a. get sign bit in position
  aeven = _mm_srai_epi16(aeven, 8); // sign extend even numbered elements
  const __m128i aodd = _mm_srai_epi16(a, 8); // sign extend odd  numbered elements
  const __m128i sum1 = _mm_add_epi16(aeven, aodd); // add even and odd elements
  // The hadd instruction is inefficient, and may be split into two instructions for faster decoding
  const __m128i sum2 = _mm_add_epi16(sum1, _mm_unpackhi_epi64(sum1, sum1));
  const __m128i sum3 = _mm_add_epi16(sum2, _mm_shuffle_epi32(sum2, 1));
  const __m128i sum4 = _mm_add_epi16(sum3, _mm_shufflelo_epi16(sum3, 1));
  auto sum5 = i16(_mm_cvtsi128_si32(sum4)); // 16 bit sum
  return sum5; // sign extend to 32 bits
#endif
}

// function add_saturated: add element by element, signed with saturation
inline i8x16 add_saturated(const i8x16 a, const i8x16 b) {
  return _mm_adds_epi8(a, b);
}

// function sub_saturated: subtract element by element, signed with saturation
inline i8x16 sub_saturated(const i8x16 a, const i8x16 b) {
  return _mm_subs_epi8(a, b);
}

// function max: a > b ? a : b
inline i8x16 max(const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_max_epi8(a, b);
#else // SSE2
  __m128i signbit = _mm_set1_epi32(0x80808080);
  __m128i a1 = _mm_xor_si128(a, signbit); // add 0x80
  __m128i b1 = _mm_xor_si128(b, signbit); // add 0x80
  __m128i m1 = _mm_max_epu8(a1, b1); // unsigned max
  return _mm_xor_si128(m1, signbit); // sub 0x80
#endif
}

// function min: a < b ? a : b
inline i8x16 min(const i8x16 a, const i8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_min_epi8(a, b);
#else // SSE2
  __m128i signbit = _mm_set1_epi32(0x80808080);
  __m128i a1 = _mm_xor_si128(a, signbit); // add 0x80
  __m128i b1 = _mm_xor_si128(b, signbit); // add 0x80
  __m128i m1 = _mm_min_epu8(a1, b1); // unsigned min
  return _mm_xor_si128(m1, signbit); // sub 0x80
#endif
}

// function abs: a >= 0 ? a : -a
inline i8x16 abs(const i8x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  return _mm_abs_epi8(a);
#else // SSE2
  __m128i nega = _mm_sub_epi8(_mm_setzero_si128(), a);
  return _mm_min_epu8(
    a, nega); // unsigned min (the negative value is bigger when compared as unsigned)
#endif
}

// function abs_saturated: same as abs, saturate if overflow
inline i8x16 abs_saturated(const i8x16 a) {
  const __m128i absa = abs(a); // abs(a)
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_min_epu8(absa, i8x16(0x7F));
#else
  __m128i overfl = _mm_cmpgt_epi8(_mm_setzero_si128(), absa); // 0 > a
  return _mm_add_epi8(absa, overfl); // subtract 1 if 0x80
#endif
}

// function rotate_left: rotate each element left by b bits
// Use negative count to rotate right
inline i8x16 rotate_left(const i8x16 a, i32 b) {
#ifdef __XOP__ // AMD XOP instruction set
  return (i8x16)_mm_rot_epi8(a, _mm_set1_epi8(b));
#else // SSE2 instruction set
  const u8 mask = 0xFFU << b; // mask off overflow bits
  const __m128i m = _mm_set1_epi8(i8(mask));
  const __m128i bb = _mm_cvtsi32_si128(b & 7); // b modulo 8
  const __m128i mbb = _mm_cvtsi32_si128((-b) & 7); // 8-b modulo 8
  __m128i left = _mm_sll_epi16(a, bb); // a << b
  __m128i right = _mm_srl_epi16(a, mbb); // a >> 8-b
  left = _mm_and_si128(m, left); // mask off overflow bits
  right = _mm_andnot_si128(m, right);
  return _mm_or_si128(left, right); // combine left and right shifted bits
#endif
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I08X16_HPP
