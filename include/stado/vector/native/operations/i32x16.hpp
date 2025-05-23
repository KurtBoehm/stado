#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I32X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I32X16_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/16.hpp"
#include "stado/vector/native/operations/i32x8.hpp"
#include "stado/vector/native/types/i32x16.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
// vector operator + : add element by element
inline i32x16 operator+(const i32x16 a, const i32x16 b) {
  return _mm512_add_epi32(a, b);
}
// vector operator += : add
inline i32x16& operator+=(i32x16& a, const i32x16 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i32x16 operator++(i32x16& a, int) {
  i32x16 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i32x16& operator++(i32x16& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i32x16 operator-(const i32x16 a, const i32x16 b) {
  return _mm512_sub_epi32(a, b);
}
// vector operator - : unary minus
inline i32x16 operator-(const i32x16 a) {
  return _mm512_sub_epi32(_mm512_setzero_epi32(), a);
}
// vector operator -= : subtract
inline i32x16& operator-=(i32x16& a, const i32x16 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i32x16 operator--(i32x16& a, int) {
  i32x16 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i32x16& operator--(i32x16& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i32x16 operator*(const i32x16 a, const i32x16 b) {
  return _mm512_mullo_epi32(a, b);
}
// vector operator *= : multiply
inline i32x16& operator*=(i32x16& a, const i32x16 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer. See bottom of file

// vector operator << : shift left
inline i32x16 operator<<(const i32x16 a, i32 b) {
  return _mm512_sll_epi32(a, _mm_cvtsi32_si128(b));
}
// vector operator <<= : shift left
inline i32x16& operator<<=(i32x16& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic
inline i32x16 operator>>(const i32x16 a, i32 b) {
  return _mm512_sra_epi32(a, _mm_cvtsi32_si128(b));
}
// vector operator >>= : shift right arithmetic
inline i32x16& operator>>=(i32x16& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
template<AnyInt32x16 TVec>
inline CompactMask<16> operator==(const TVec a, const TVec b) {
  return _mm512_cmpeq_epi32_mask(a, b);
}

// vector operator != : returns true for elements for which a != b
template<AnyInt32x16 TVec>
inline CompactMask<16> operator!=(const TVec a, const TVec b) {
  return _mm512_cmpneq_epi32_mask(a, b);
}

// vector operator > : returns true for elements for which a > b
inline CompactMask<16> operator>(const i32x16 a, const i32x16 b) {
  return _mm512_cmp_epi32_mask(a, b, 6);
}

// vector operator < : returns true for elements for which a < b
inline CompactMask<16> operator<(const i32x16 a, const i32x16 b) {
  return _mm512_cmp_epi32_mask(a, b, 1);
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline CompactMask<16> operator>=(const i32x16 a, const i32x16 b) {
  return _mm512_cmp_epi32_mask(a, b, 5);
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline CompactMask<16> operator<=(const i32x16 a, const i32x16 b) {
  return _mm512_cmp_epi32_mask(a, b, 2);
}

// vector operator & : bitwise and
inline i32x16 operator&(const i32x16 a, const i32x16 b) {
  return _mm512_and_epi32(a, b);
}
// vector operator &= : bitwise and
inline i32x16& operator&=(i32x16& a, const i32x16 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i32x16 operator|(const i32x16 a, const i32x16 b) {
  return _mm512_or_epi32(a, b);
}
// vector operator |= : bitwise or
inline i32x16& operator|=(i32x16& a, const i32x16 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i32x16 operator^(const i32x16 a, const i32x16 b) {
  return _mm512_xor_epi32(a, b);
}
// vector operator ^= : bitwise xor
inline i32x16& operator^=(i32x16& a, const i32x16 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i32x16 operator~(const i32x16 a) {
  return a ^ i32x16(-1);
  // This is potentially faster, but not on any current compiler:
  // return _mm512_ternarylogic_epi32(_mm512_undefined_epi32(), _mm512_undefined_epi32(), a, 0x55);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
inline i32x16 select(const CompactMask<16> s, const i32x16 a, const i32x16 b) {
  return _mm512_mask_mov_epi32(b, s, a); // conditional move may be optimized better by the compiler
                                         // than blend return _mm512_mask_blend_epi32(s, b, a);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i32x16 if_add(const CompactMask<16> f, const i32x16 a, const i32x16 b) {
  return _mm512_mask_add_epi32(a, f, a, b);
}

// Conditional subtract
inline i32x16 if_sub(const CompactMask<16> f, const i32x16 a, const i32x16 b) {
  return _mm512_mask_sub_epi32(a, f, a, b);
}

// Conditional multiply
inline i32x16 if_mul(const CompactMask<16> f, const i32x16 a, const i32x16 b) {
  return _mm512_mask_mullo_epi32(a, f, a, b);
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i32 horizontal_add(const i32x16 a) {
#if defined(__INTEL_COMPILER)
  return _mm512_reduce_add_epi32(a);
#else
  return horizontal_add(a.get_low() + a.get_high());
#endif
}

// function add_saturated: add element by element, signed with saturation
// (is it faster to up-convert to 64 bit integers, and then downconvert the sum with saturation?)
inline i32x16 add_saturated(const i32x16 a, const i32x16 b) {
  const __m512i sum = _mm512_add_epi32(a, b); // a + b
  const __m512i axb = _mm512_xor_epi32(a, b); // check if a and b have different sign
  const __m512i axs = _mm512_xor_epi32(a, sum); // check if a and sum have different sign
  const __m512i ovf1 = _mm512_andnot_epi32(axb, axs); // check if sum has wrong sign
  const __m512i ovf2 = _mm512_srai_epi32(ovf1, 31); // -1 if overflow
  const __mmask16 ovf3 = _mm512_cmpneq_epi32_mask(ovf2, _mm512_setzero_epi32()); // same, as mask
  const __m512i asign = _mm512_srli_epi32(a, 31); // 1  if a < 0
  const __m512i sat1 = _mm512_srli_epi32(ovf2, 1); // 7FFFFFFF if overflow
  const __m512i sat2 =
    _mm512_add_epi32(sat1, asign); // 7FFFFFFF if positive overflow 80000000 if negative overflow
  return _mm512_mask_blend_epi32(ovf3, sum, sat2); // sum if not overflow, else sat2
}

// function sub_saturated: subtract element by element, signed with saturation
inline i32x16 sub_saturated(const i32x16 a, const i32x16 b) {
  const __m512i diff = _mm512_sub_epi32(a, b); // a + b
  const __m512i axb = _mm512_xor_si512(a, b); // check if a and b have different sign
  const __m512i axs = _mm512_xor_si512(a, diff); // check if a and sum have different sign
  const __m512i ovf1 = _mm512_and_si512(axb, axs); // check if sum has wrong sign
  const __m512i ovf2 = _mm512_srai_epi32(ovf1, 31); // -1 if overflow
  const __mmask16 ovf3 = _mm512_cmpneq_epi32_mask(ovf2, _mm512_setzero_epi32()); // same, as mask
  const __m512i asign = _mm512_srli_epi32(a, 31); // 1  if a < 0
  const __m512i sat1 = _mm512_srli_epi32(ovf2, 1); // 7FFFFFFF if overflow
  const __m512i sat2 =
    _mm512_add_epi32(sat1, asign); // 7FFFFFFF if positive overflow 80000000 if negative overflow
  return _mm512_mask_blend_epi32(ovf3, diff, sat2); // sum if not overflow, else sat2
}

// function max: a > b ? a : b
inline i32x16 max(const i32x16 a, const i32x16 b) {
  return _mm512_max_epi32(a, b);
}

// function min: a < b ? a : b
inline i32x16 min(const i32x16 a, const i32x16 b) {
  return _mm512_min_epi32(a, b);
}

// function abs: a >= 0 ? a : -a
inline i32x16 abs(const i32x16 a) {
  return _mm512_abs_epi32(a);
}

// function abs_saturated: same as abs, saturate if overflow
inline i32x16 abs_saturated(const i32x16 a) {
  return _mm512_min_epu32(abs(a), i32x16(0x7FFFFFFF));
}

// function rotate_left all elements
// Use negative count to rotate right
inline i32x16 rotate_left(const i32x16 a, i32 b) {
  return _mm512_rolv_epi32(a, i32x16(b));
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I32X16_HPP
