#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U32X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U32X16_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/mask-16.hpp"
#include "stado/vector/native/operations/i32x16.hpp"
#include "stado/vector/native/types/i32x16.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
// vector operator + : add
inline u32x16 operator+(const u32x16 a, const u32x16 b) {
  return {i32x16(a) + i32x16(b)};
}

// vector operator - : subtract
inline u32x16 operator-(const u32x16 a, const u32x16 b) {
  return {i32x16(a) - i32x16(b)};
}

// vector operator * : multiply
inline u32x16 operator*(const u32x16 a, const u32x16 b) {
  return {i32x16(a) * i32x16(b)};
}

// vector operator / : divide
// See bottom of file

// vector operator >> : shift right logical all elements
inline u32x16 operator>>(const u32x16 a, u32 b) {
  return _mm512_srl_epi32(a, _mm_cvtsi32_si128(i32(b)));
}
inline u32x16 operator>>(const u32x16 a, i32 b) {
  return a >> u32(b);
}

// vector operator >>= : shift right logical
inline u32x16& operator>>=(u32x16& a, u32 b) {
  a = a >> b;
  return a;
}

// vector operator >>= : shift right logical
inline u32x16& operator>>=(u32x16& a, i32 b) {
  a = a >> u32(b);
  return a;
}

// vector operator << : shift left all elements
inline u32x16 operator<<(const u32x16 a, u32 b) {
  return {i32x16(a) << i32(b)};
}

// vector operator << : shift left all elements
inline u32x16 operator<<(const u32x16 a, i32 b) {
  return {i32x16(a) << b};
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline CompactMask<16> operator<(const u32x16 a, const u32x16 b) {
  return _mm512_cmp_epu32_mask(a, b, 1);
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline CompactMask<16> operator>(const u32x16 a, const u32x16 b) {
  return _mm512_cmp_epu32_mask(a, b, 6);
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline CompactMask<16> operator>=(const u32x16 a, const u32x16 b) {
  return _mm512_cmp_epu32_mask(a, b, 5);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline CompactMask<16> operator<=(const u32x16 a, const u32x16 b) {
  return _mm512_cmp_epu32_mask(a, b, 2);
}

// vector operator & : bitwise and
inline u32x16 operator&(const u32x16 a, const u32x16 b) {
  return {i32x16(a) & i32x16(b)};
}

// vector operator | : bitwise or
inline u32x16 operator|(const u32x16 a, const u32x16 b) {
  return {i32x16(a) | i32x16(b)};
}

// vector operator ^ : bitwise xor
inline u32x16 operator^(const u32x16 a, const u32x16 b) {
  return {i32x16(a) ^ i32x16(b)};
}

// vector operator ~ : bitwise not
inline u32x16 operator~(const u32x16 a) {
  return {~i32x16(a)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
inline u32x16 select(const CompactMask<16> s, const u32x16 a, const u32x16 b) {
  return {select(s, i32x16(a), i32x16(b))};
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u32x16 if_add(const CompactMask<16> f, const u32x16 a, const u32x16 b) {
  return {if_add(f, i32x16(a), i32x16(b))};
}

// Conditional subtract
inline u32x16 if_sub(const CompactMask<16> f, const u32x16 a, const u32x16 b) {
  return {if_sub(f, i32x16(a), i32x16(b))};
}

// Conditional multiply
inline u32x16 if_mul(const CompactMask<16> f, const u32x16 a, const u32x16 b) {
  return {if_mul(f, i32x16(a), i32x16(b))};
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline u32 horizontal_add(const u32x16 a) {
  return u32(horizontal_add(i32x16(a)));
}

// horizontal_add_x: Horizontal add extended: Calculates the sum of all vector elements. Defined
// later in this file

// function add_saturated: add element by element, unsigned with saturation
inline u32x16 add_saturated(const u32x16 a, const u32x16 b) {
  const u32x16 sum = a + b;
  const CompactMask<16> overflow = sum < (a | b); // overflow if (a + b) < (a | b)
  return _mm512_mask_set1_epi32(sum, overflow, -1); // 0xFFFFFFFF if overflow
}

// function sub_saturated: subtract element by element, unsigned with saturation
inline u32x16 sub_saturated(const u32x16 a, const u32x16 b) {
  const u32x16 diff = a - b;
  return _mm512_maskz_mov_epi32(diff <= a, diff); // underflow if diff > a gives zero
}

// function max: a > b ? a : b
inline u32x16 max(const u32x16 a, const u32x16 b) {
  return _mm512_max_epu32(a, b);
}

// function min: a < b ? a : b
inline u32x16 min(const u32x16 a, const u32x16 b) {
  return _mm512_min_epu32(a, b);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U32X16_HPP
