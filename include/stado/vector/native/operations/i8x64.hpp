#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I8X64_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I8X64_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/64.hpp"
#include "stado/vector/native/operations/i64x8.hpp"
#include "stado/vector/native/operations/i8x32.hpp"
#include "stado/vector/native/types/i32x16.hpp"
#include "stado/vector/native/types/i64x8.hpp"
#include "stado/vector/native/types/i8x64.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW

namespace stado {
// vector operator + : add element by element
inline i8x64 operator+(const i8x64 a, const i8x64 b) {
  return _mm512_add_epi8(a, b);
}
// vector operator += : add
inline i8x64& operator+=(i8x64& a, const i8x64 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i8x64 operator++(i8x64& a, int) {
  i8x64 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i8x64& operator++(i8x64& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i8x64 operator-(const i8x64 a, const i8x64 b) {
  return _mm512_sub_epi8(a, b);
}
// vector operator - : unary minus
inline i8x64 operator-(const i8x64 a) {
  return _mm512_sub_epi8(_mm512_setzero_epi32(), a);
}
// vector operator -= : subtract
inline i8x64& operator-=(i8x64& a, const i8x64 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i8x64 operator--(i8x64& a, int) {
  i8x64 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i8x64& operator--(i8x64& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i8x64 operator*(const i8x64 a, const i8x64 b) {
  // There is no 8-bit multiply. Split into two 16-bit multiplies
  const __m512i aodd = _mm512_srli_epi16(a, 8); // odd numbered elements of a
  const __m512i bodd = _mm512_srli_epi16(b, 8); // odd numbered elements of b
  const __m512i muleven = _mm512_mullo_epi16(a, b); // product of even numbered elements
  __m512i mulodd = _mm512_mullo_epi16(aodd, bodd); // product of odd  numbered elements
  mulodd = _mm512_slli_epi16(mulodd, 8); // put odd numbered elements back in place
  const __m512i product =
    _mm512_mask_mov_epi8(muleven, 0xAAAAAAAAAAAAAAAA, mulodd); // interleave even and odd
  return product;
}

// vector operator *= : multiply
inline i8x64& operator*=(i8x64& a, const i8x64 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer. See bottom of file

// vector operator << : shift left
inline i8x64 operator<<(const i8x64 a, i32 b) {
  // mask to remove bits that are shifted out
  const u32 mask = u32{0xFF} >> u32(b);
  // remove bits that will overflow
  const __m512i am = _mm512_and_si512(a, _mm512_set1_epi8(i8(mask)));
  const __m512i res = _mm512_sll_epi16(am, _mm_cvtsi32_si128(b)); // 16-bit shifts
  return res;
}

// vector operator <<= : shift left
inline i8x64& operator<<=(i8x64& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic
inline i8x64 operator>>(const i8x64 a, i32 b) {
  __m512i aeven = _mm512_slli_epi16(a, 8); // even numbered elements of a. get sign bit in position
  aeven = _mm512_sra_epi16(aeven, _mm_cvtsi32_si128(b + 8)); // shift arithmetic, back to position
  const __m512i aodd =
    _mm512_sra_epi16(a, _mm_cvtsi32_si128(b)); // shift odd numbered elements arithmetic
  const __m512i res =
    _mm512_mask_mov_epi8(aeven, 0xAAAAAAAAAAAAAAAA, aodd); // interleave even and odd
  return res;
}
// vector operator >>= : shift right arithmetic
inline i8x64& operator>>=(i8x64& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
inline CompactMask<64> operator==(const i8x64 a, const i8x64 b) {
  return _mm512_cmpeq_epi8_mask(a, b);
}

// vector operator != : returns true for elements for which a != b
inline CompactMask<64> operator!=(const i8x64 a, const i8x64 b) {
  return _mm512_cmpneq_epi8_mask(a, b);
}

// vector operator > : returns true for elements for which a > b
inline CompactMask<64> operator>(const i8x64 a, const i8x64 b) {
  return _mm512_cmp_epi8_mask(a, b, 6);
}

// vector operator < : returns true for elements for which a < b
inline CompactMask<64> operator<(const i8x64 a, const i8x64 b) {
  return _mm512_cmp_epi8_mask(a, b, 1);
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline CompactMask<64> operator>=(const i8x64 a, const i8x64 b) {
  return _mm512_cmp_epi8_mask(a, b, 5);
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline CompactMask<64> operator<=(const i8x64 a, const i8x64 b) {
  return _mm512_cmp_epi8_mask(a, b, 2);
}

// vector operator & : bitwise and
inline i8x64 operator&(const i8x64 a, const i8x64 b) {
  return _mm512_and_epi32(a, b);
}

// vector operator &= : bitwise and
inline i8x64& operator&=(i8x64& a, const i8x64 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i8x64 operator|(const i8x64 a, const i8x64 b) {
  return _mm512_or_epi32(a, b);
}

// vector operator |= : bitwise or
inline i8x64& operator|=(i8x64& a, const i8x64 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i8x64 operator^(const i8x64 a, const i8x64 b) {
  return _mm512_xor_epi32(a, b);
}

// vector operator ^= : bitwise xor
inline i8x64& operator^=(i8x64& a, const i8x64 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i8x64 operator~(const i8x64 a) {
  return {~i32x16(a)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
inline i8x64 select(const CompactMask<64> s, const i8x64 a, const i8x64 b) {
  return _mm512_mask_mov_epi8(
    b, s, a); // conditional move may be optimized better by the compiler than blend
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i8x64 if_add(const CompactMask<64> f, const i8x64 a, const i8x64 b) {
  return _mm512_mask_add_epi8(a, f, a, b);
}

// Conditional subtract
inline i8x64 if_sub(const CompactMask<64> f, const i8x64 a, const i8x64 b) {
  return _mm512_mask_sub_epi8(a, f, a, b);
}

// Conditional multiply
inline i8x64 if_mul(const CompactMask<64> f, const i8x64 a, const i8x64 b) {
  const i8x64 m = a * b;
  return select(f, m, a);
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i8 horizontal_add(const i8x64 a) {
  const __m512i sum1 = _mm512_sad_epu8(a, _mm512_setzero_si512());
  return i8(horizontal_add(i64x8(sum1)));
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is sign-extended before addition to avoid overflow
inline i32 horizontal_add_x(const i8x64 a) {
  return horizontal_add_x(a.get_low()) + horizontal_add_x(a.get_high());
}

// function add_saturated: add element by element, signed with saturation
inline i8x64 add_saturated(const i8x64 a, const i8x64 b) {
  return _mm512_adds_epi8(a, b);
}

// function sub_saturated: subtract element by element, signed with saturation
inline i8x64 sub_saturated(const i8x64 a, const i8x64 b) {
  return _mm512_subs_epi8(a, b);
}

// function max: a > b ? a : b
inline i8x64 max(const i8x64 a, const i8x64 b) {
  return _mm512_max_epi8(a, b);
}

// function min: a < b ? a : b
inline i8x64 min(const i8x64 a, const i8x64 b) {
  return _mm512_min_epi8(a, b);
}

// function abs: a >= 0 ? a : -a
inline i8x64 abs(const i8x64 a) {
  return _mm512_abs_epi8(a);
}

// function abs_saturated: same as abs, saturate if overflow
inline i8x64 abs_saturated(const i8x64 a) {
  return _mm512_min_epu8(abs(a), i8x64(0x7F));
}

// function rotate_left all elements
// Use negative count to rotate right
inline i8x64 rotate_left(const i8x64 a, i32 b) {
  const u8 mask = 0xFF << b; // mask off overflow bits
  const __m512i m = _mm512_set1_epi8(i8(mask));
  const __m128i bb = _mm_cvtsi32_si128(b & 7); // b modulo 8
  const __m128i mbb = _mm_cvtsi32_si128((-b) & 7); // 8-b modulo 8
  __m512i left = _mm512_sll_epi16(a, bb); // a << b
  __m512i right = _mm512_srl_epi16(a, mbb); // a >> 8-b
  left = _mm512_and_si512(m, left); // mask off overflow bits
  right = _mm512_andnot_si512(m, right);
  return _mm512_or_si512(left, right); // combine left and right shifted bits
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I8X64_HPP
