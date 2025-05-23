#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U8X64_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U8X64_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/64.hpp"
#include "stado/vector/native/operations/i8x64.hpp"
#include "stado/vector/native/types/i8x64.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
namespace stado {
// vector operator + : add element by element
inline u8x64 operator+(const u8x64 a, const u8x64 b) {
  return _mm512_add_epi8(a, b);
}

// vector operator - : subtract element by element
inline u8x64 operator-(const u8x64 a, const u8x64 b) {
  return _mm512_sub_epi8(a, b);
}

// vector operator ' : multiply element by element
inline u8x64 operator*(const u8x64 a, const u8x64 b) {
  return {i8x64(a) * i8x64(b)};
}

// vector operator / : divide. See bottom of file

// vector operator >> : shift right logical all elements
inline u8x64 operator>>(const u8x64 a, u32 b) {
  const u32 mask = u32{0xFF} << b; // mask to remove bits that are shifted out
  // remove bits that will overflow
  const __m512i am = _mm512_and_si512(a, _mm512_set1_epi8(i8(mask)));
  const __m512i res = _mm512_srl_epi16(am, _mm_cvtsi32_si128(i32(b))); // 16-bit shifts
  return res;
}
inline u8x64 operator>>(const u8x64 a, int b) {
  return a >> u32(b);
}

// vector operator >>= : shift right logical
inline u8x64& operator>>=(u8x64& a, u32 b) {
  a = a >> b;
  return a;
}

// vector operator >>= : shift right logical (signed b)
inline u8x64& operator>>=(u8x64& a, i32 b) {
  a = a >> u32(b);
  return a;
}

// vector operator << : shift left all elements
inline u8x64 operator<<(const u8x64 a, u32 b) {
  return {i8x64(a) << i32(b)};
}
inline u8x64 operator<<(const u8x64 a, int b) {
  return a << u32(b);
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline CompactMask<64> operator<(const u8x64 a, const u8x64 b) {
  return _mm512_cmp_epu8_mask(a, b, 1);
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline CompactMask<64> operator>(const u8x64 a, const u8x64 b) {
  return _mm512_cmp_epu8_mask(a, b, 6);
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline CompactMask<64> operator>=(const u8x64 a, const u8x64 b) {
  return _mm512_cmp_epu8_mask(a, b, 5);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline CompactMask<64> operator<=(const u8x64 a, const u8x64 b) {
  return _mm512_cmp_epu8_mask(a, b, 2);
}

// vector operator & : bitwise and
inline u8x64 operator&(const u8x64 a, const u8x64 b) {
  return {i8x64(a) & i8x64(b)};
}

// vector operator | : bitwise or
inline u8x64 operator|(const u8x64 a, const u8x64 b) {
  return {i8x64(a) | i8x64(b)};
}

// vector operator ^ : bitwise xor
inline u8x64 operator^(const u8x64 a, const u8x64 b) {
  return {i8x64(a) ^ i8x64(b)};
}

// vector operator ~ : bitwise not
inline u8x64 operator~(const u8x64 a) {
  return {~i8x64(a)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
inline u8x64 select(const CompactMask<64> s, const u8x64 a, const u8x64 b) {
  return {select(s, i8x64(a), i8x64(b))};
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u8x64 if_add(const CompactMask<64> f, const u8x64 a, const u8x64 b) {
  return _mm512_mask_add_epi8(a, f, a, b);
}

// Conditional subtract
inline u8x64 if_sub(const CompactMask<64> f, const u8x64 a, const u8x64 b) {
  return _mm512_mask_sub_epi8(a, f, a, b);
}

// Conditional multiply
inline u8x64 if_mul(const CompactMask<64> f, const u8x64 a, const u8x64 b) {
  const u8x64 m = a * b;
  return select(f, m, a);
}

// function add_saturated: add element by element, unsigned with saturation
inline u8x64 add_saturated(const u8x64 a, const u8x64 b) {
  return _mm512_adds_epu8(a, b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
inline u8x64 sub_saturated(const u8x64 a, const u8x64 b) {
  return _mm512_subs_epu8(a, b);
}

// function max: a > b ? a : b
inline u8x64 max(const u8x64 a, const u8x64 b) {
  return _mm512_max_epu8(a, b);
}

// function min: a < b ? a : b
inline u8x64 min(const u8x64 a, const u8x64 b) {
  return _mm512_min_epu8(a, b);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U8X64_HPP
