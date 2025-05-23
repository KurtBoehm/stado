#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U16X32_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U16X32_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/mask-32.hpp"
#include "stado/vector/native/operations/i16x32.hpp"
#include "stado/vector/native/types/i16x32.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
namespace stado {
// vector operator + : add element by element
inline u16x32 operator+(const u16x32 a, const u16x32 b) {
  return _mm512_add_epi16(a, b);
}

// vector operator - : subtract element by element
inline u16x32 operator-(const u16x32 a, const u16x32 b) {
  return _mm512_sub_epi16(a, b);
}

// vector operator * : multiply element by element
inline u16x32 operator*(const u16x32 a, const u16x32 b) {
  return _mm512_mullo_epi16(a, b);
}

// vector operator / : divide
// See bottom of file

// vector operator >> : shift right logical all elements
inline u16x32 operator>>(const u16x32 a, u32 b) {
  return _mm512_srl_epi16(a, _mm_cvtsi32_si128(i32(b)));
}
inline u16x32 operator>>(const u16x32 a, int b) {
  return a >> u32(b);
}

// vector operator >>= : shift right logical
inline u16x32& operator>>=(u16x32& a, u32 b) {
  a = a >> b;
  return a;
}

// vector operator >>= : shift right logical (signed b)
inline u16x32& operator>>=(u16x32& a, i32 b) {
  a = a >> u32(b);
  return a;
}

// vector operator << : shift left all elements
inline u16x32 operator<<(const u16x32 a, u32 b) {
  return _mm512_sll_epi16(a, _mm_cvtsi32_si128(i32(b)));
}
inline u16x32 operator<<(const u16x32 a, int b) {
  return a << u32(b);
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline CompactMask<32> operator<(const u16x32 a, const u16x32 b) {
  return _mm512_cmp_epu16_mask(a, b, 1);
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline CompactMask<32> operator>(const u16x32 a, const u16x32 b) {
  return _mm512_cmp_epu16_mask(a, b, 6);
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline CompactMask<32> operator>=(const u16x32 a, const u16x32 b) {
  return _mm512_cmp_epu16_mask(a, b, 5);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline CompactMask<32> operator<=(const u16x32 a, const u16x32 b) {
  return _mm512_cmp_epu16_mask(a, b, 2);
}

// vector operator & : bitwise and
inline u16x32 operator&(const u16x32 a, const u16x32 b) {
  return u16x32(i16x32(a) & i16x32(b));
}

// vector operator | : bitwise or
inline u16x32 operator|(const u16x32 a, const u16x32 b) {
  return u16x32(i16x32(a) | i16x32(b));
}

// vector operator ^ : bitwise xor
inline u16x32 operator^(const u16x32 a, const u16x32 b) {
  return u16x32(i16x32(a) ^ i16x32(b));
}

// vector operator ~ : bitwise not
inline u16x32 operator~(const u16x32 a) {
  return u16x32(~i16x32(a));
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
inline u16x32 select(const CompactMask<32> s, const u16x32 a, const u16x32 b) {
  return u16x32(select(s, i16x32(a), i16x32(b)));
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u16x32 if_add(const CompactMask<32> f, const u16x32 a, const u16x32 b) {
  return _mm512_mask_add_epi16(a, f, a, b);
}

// Conditional subtract
inline u16x32 if_sub(const CompactMask<32> f, const u16x32 a, const u16x32 b) {
  return _mm512_mask_sub_epi16(a, f, a, b);
}

// Conditional multiply
inline u16x32 if_mul(const CompactMask<32> f, const u16x32 a, const u16x32 b) {
  return _mm512_mask_mullo_epi16(a, f, a, b);
}

// function add_saturated: add element by element, unsigned with saturation
inline u16x32 add_saturated(const u16x32 a, const u16x32 b) {
  return _mm512_adds_epu16(a, b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
inline u16x32 sub_saturated(const u16x32 a, const u16x32 b) {
  return _mm512_subs_epu16(a, b);
}

// function max: a > b ? a : b
inline u16x32 max(const u16x32 a, const u16x32 b) {
  return _mm512_max_epu16(a, b);
}

// function min: a < b ? a : b
inline u16x32 min(const u16x32 a, const u16x32 b) {
  return _mm512_min_epu16(a, b);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U16X32_HPP
