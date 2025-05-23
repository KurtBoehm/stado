#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U08X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U08X16_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i08x16.hpp"

namespace stado {
// vector operator << : shift left all elements
inline u8x16 operator<<(const u8x16 a, u32 b) {
  const u32 mask = u32{0xFF} >> b; // mask to remove bits that are shifted out
  const __m128i am = _mm_and_si128(a, _mm_set1_epi8(i8(mask))); // remove bits that will overflow
  const __m128i res = _mm_sll_epi16(am, _mm_cvtsi32_si128(i32(b))); // 16-bit shifts
  return res;
}

// vector operator << : shift left all elements
inline u8x16 operator<<(const u8x16 a, i32 b) {
  return a << u32(b);
}

// vector operator >> : shift right logical all elements
inline u8x16 operator>>(const u8x16 a, u32 b) {
  const u32 mask = u32{0xFF} << b; // mask to remove bits that are shifted out
  const __m128i am = _mm_and_si128(a, _mm_set1_epi8(i8(mask))); // remove bits that will overflow
  const __m128i res = _mm_srl_epi16(am, _mm_cvtsi32_si128(i32(b))); // 16-bit shifts
  return res;
}

// vector operator >> : shift right logical all elements
inline u8x16 operator>>(const u8x16 a, i32 b) {
  return a >> u32(b);
}

// vector operator >>= : shift right logical
inline u8x16& operator>>=(u8x16& a, int b) {
  a = a >> b;
  return a;
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline Mask<8, 16> operator>=(const u8x16 a, const u8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epu8_mask(a, b, 5);
#else
  return (Mask<8, 16>)_mm_cmpeq_epi8(_mm_max_epu8(a, b), a); // a == max(a,b)
#endif
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline Mask<8, 16> operator<=(const u8x16 a, const u8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epu8_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline Mask<8, 16> operator>(const u8x16 a, const u8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epu8_mask(a, b, 6);
#else
  return Mask<8, 16>(i8x16(~(b >= a)));
#endif
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline Mask<8, 16> operator<(const u8x16 a, const u8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epu8_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator + : add
inline u8x16 operator+(const u8x16 a, const u8x16 b) {
  return u8x16(i8x16(a) + i8x16(b));
}

// vector operator - : subtract
inline u8x16 operator-(const u8x16 a, const u8x16 b) {
  return u8x16(i8x16(a) - i8x16(b));
}

// vector operator * : multiply
inline u8x16 operator*(const u8x16 a, const u8x16 b) {
  return u8x16(i8x16(a) * i8x16(b));
}

// vector operator & : bitwise and
inline u8x16 operator&(const u8x16 a, const u8x16 b) {
  return u8x16(si128(a) & si128(b));
}
inline u8x16 operator&&(const u8x16 a, const u8x16 b) {
  return a & b;
}

// vector operator | : bitwise or
inline u8x16 operator|(const u8x16 a, const u8x16 b) {
  return u8x16(si128(a) | si128(b));
}
inline u8x16 operator||(const u8x16 a, const u8x16 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
inline u8x16 operator^(const u8x16 a, const u8x16 b) {
  return u8x16(si128(a) ^ si128(b));
}

// vector operator ~ : bitwise not
inline u8x16 operator~(const u8x16 a) {
  return u8x16(~si128(a));
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
inline u8x16 select(const Mask<8, 16> s, const u8x16 a, const u8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mov_epi8(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u8x16 if_add(const Mask<8, 16> f, const u8x16 a, const u8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_add_epi8(a, f, a, b);
#else
  return a + (u8x16(f) & b);
#endif
}

// Conditional sub: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
inline u8x16 if_sub(const Mask<8, 16> f, const u8x16 a, const u8x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_sub_epi8(a, f, a, b);
#else
  return a - (u8x16(f) & b);
#endif
}

// Conditional mul: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
inline u8x16 if_mul(const Mask<8, 16> f, const u8x16 a, const u8x16 b) {
  return select(f, a * b, a);
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
// (Note: horizontal_add_x(u8x16) is slightly faster)
inline u32 horizontal_add(const u8x16 a) {
  const __m128i sum1 = _mm_sad_epu8(a, _mm_setzero_si128());
  const __m128i sum2 = _mm_unpackhi_epi64(sum1, sum1);
  const __m128i sum3 = _mm_add_epi16(sum1, sum2);
  auto sum4 = u16(_mm_cvtsi128_si32(sum3)); // truncate to 16 bits
  return sum4;
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is zero-extended before addition to avoid overflow
inline u32 horizontal_add_x(const u8x16 a) {
  const __m128i sum1 = _mm_sad_epu8(a, _mm_setzero_si128());
  const __m128i sum2 = _mm_unpackhi_epi64(sum1, sum1);
  const __m128i sum3 = _mm_add_epi16(sum1, sum2);
  return u32(_mm_cvtsi128_si32(sum3));
}

// function add_saturated: add element by element, unsigned with saturation
inline u8x16 add_saturated(const u8x16 a, const u8x16 b) {
  return _mm_adds_epu8(a, b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
inline u8x16 sub_saturated(const u8x16 a, const u8x16 b) {
  return _mm_subs_epu8(a, b);
}

// function max: a > b ? a : b
inline u8x16 max(const u8x16 a, const u8x16 b) {
  return _mm_max_epu8(a, b);
}

// function min: a < b ? a : b
inline u8x16 min(const u8x16 a, const u8x16 b) {
  return _mm_min_epu8(a, b);
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U08X16_HPP
