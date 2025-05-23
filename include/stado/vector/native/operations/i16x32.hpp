#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I16X32_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I16X32_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/mask-32.hpp"
#include "stado/vector/native/operations/i16x16.hpp"
#include "stado/vector/native/types/i16x16.hpp"
#include "stado/vector/native/types/i16x32.hpp"
#include "stado/vector/native/types/i32x16.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
namespace stado {
// vector operator + : add element by element
inline i16x32 operator+(const i16x32 a, const i16x32 b) {
  return _mm512_add_epi16(a, b);
}
// vector operator += : add
inline i16x32& operator+=(i16x32& a, const i16x32 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i16x32 operator++(i16x32& a, int) {
  i16x32 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i16x32& operator++(i16x32& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i16x32 operator-(const i16x32 a, const i16x32 b) {
  return _mm512_sub_epi16(a, b);
}
// vector operator - : unary minus
inline i16x32 operator-(const i16x32 a) {
  return _mm512_sub_epi16(_mm512_setzero_epi32(), a);
}
// vector operator -= : subtract
inline i16x32& operator-=(i16x32& a, const i16x32 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i16x32 operator--(i16x32& a, int) {
  i16x32 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i16x32& operator--(i16x32& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i16x32 operator*(const i16x32 a, const i16x32 b) {
  return _mm512_mullo_epi16(a, b);
}

// vector operator *= : multiply
inline i16x32& operator*=(i16x32& a, const i16x32 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer. See bottom of file

// vector operator << : shift left
inline i16x32 operator<<(const i16x32 a, i32 b) {
  return _mm512_sll_epi16(a, _mm_cvtsi32_si128(b));
}
// vector operator <<= : shift left
inline i16x32& operator<<=(i16x32& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic
inline i16x32 operator>>(const i16x32 a, i32 b) {
  return _mm512_sra_epi16(a, _mm_cvtsi32_si128(b));
}
// vector operator >>= : shift right arithmetic
inline i16x32& operator>>=(i16x32& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
template<AnyInt16x32 TVec>
inline CompactMask<32> operator==(const TVec a, const TVec b) {
  return _mm512_cmpeq_epi16_mask(a, b);
}

// vector operator != : returns true for elements for which a != b
template<AnyInt16x32 TVec>
inline CompactMask<32> operator!=(const TVec a, const TVec b) {
  return _mm512_cmpneq_epi16_mask(a, b);
}

// vector operator > : returns true for elements for which a > b
inline CompactMask<32> operator>(const i16x32 a, const i16x32 b) {
  return _mm512_cmp_epi16_mask(a, b, 6);
}

// vector operator < : returns true for elements for which a < b
inline CompactMask<32> operator<(const i16x32 a, const i16x32 b) {
  return _mm512_cmp_epi16_mask(a, b, 1);
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline CompactMask<32> operator>=(const i16x32 a, const i16x32 b) {
  return _mm512_cmp_epi16_mask(a, b, 5);
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline CompactMask<32> operator<=(const i16x32 a, const i16x32 b) {
  return _mm512_cmp_epi16_mask(a, b, 2);
}

// vector operator & : bitwise and
inline i16x32 operator&(const i16x32 a, const i16x32 b) {
  return _mm512_and_epi32(a, b);
}

// vector operator &= : bitwise and
inline i16x32& operator&=(i16x32& a, const i16x32 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i16x32 operator|(const i16x32 a, const i16x32 b) {
  return _mm512_or_epi32(a, b);
}

// vector operator |= : bitwise or
inline i16x32& operator|=(i16x32& a, const i16x32 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i16x32 operator^(const i16x32 a, const i16x32 b) {
  return _mm512_xor_epi32(a, b);
}

// vector operator ^= : bitwise xor
inline i16x32& operator^=(i16x32& a, const i16x32 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i16x32 operator~(const i16x32 a) {
  return i16x32(~i32x16(a));
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
inline i16x32 select(const CompactMask<32> s, const i16x32 a, const i16x32 b) {
  return _mm512_mask_mov_epi16(
    b, s, a); // conditional move may be optimized better by the compiler than blend
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i16x32 if_add(const CompactMask<32> f, const i16x32 a, const i16x32 b) {
  return _mm512_mask_add_epi16(a, f, a, b);
}

// Conditional subtract
inline i16x32 if_sub(const CompactMask<32> f, const i16x32 a, const i16x32 b) {
  return _mm512_mask_sub_epi16(a, f, a, b);
}

// Conditional multiply
inline i16x32 if_mul(const CompactMask<32> f, const i16x32 a, const i16x32 b) {
  return _mm512_mask_mullo_epi16(a, f, a, b);
}

// Horizontal add: Calculates the sum of all vector elements.
// Overflow will wrap around
inline i16 horizontal_add(const i16x32 a) {
  const i16x16 s = a.get_low() + a.get_high();
  return horizontal_add(s);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is sign-extended before addition to avoid overflow
inline i32 horizontal_add_x(const i16x32 a) {
  return horizontal_add_x(a.get_low()) + horizontal_add_x(a.get_high());
}

// function add_saturated: add element by element, signed with saturation
inline i16x32 add_saturated(const i16x32 a, const i16x32 b) {
  return _mm512_adds_epi16(a, b);
}

// function sub_saturated: subtract element by element, signed with saturation
inline i16x32 sub_saturated(const i16x32 a, const i16x32 b) {
  return _mm512_subs_epi16(a, b);
}

// function max: a > b ? a : b
inline i16x32 max(const i16x32 a, const i16x32 b) {
  return _mm512_max_epi16(a, b);
}

// function min: a < b ? a : b
inline i16x32 min(const i16x32 a, const i16x32 b) {
  return _mm512_min_epi16(a, b);
}

// function abs: a >= 0 ? a : -a
inline i16x32 abs(const i16x32 a) {
  return _mm512_abs_epi16(a);
}

// function abs_saturated: same as abs, saturate if overflow
inline i16x32 abs_saturated(const i16x32 a) {
  return _mm512_min_epu16(abs(a), i16x32(0x7FFF));
}

// function rotate_left all elements
// Use negative count to rotate right
inline i16x32 rotate_left(const i16x32 a, i32 b) {
  const __m512i left = _mm512_sll_epi16(a, _mm_cvtsi32_si128(b & 0xF)); // a << b
  const __m512i right = _mm512_srl_epi16(a, _mm_cvtsi32_si128((16 - b) & 0xF)); // a >> (32 - b)
  const __m512i rot = _mm512_or_si512(left, right); // or
  return rot;
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I16X32_HPP
