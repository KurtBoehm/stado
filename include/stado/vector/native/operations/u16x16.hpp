#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U16X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U16X16_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/operations/i16x16.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i16x16.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
// vector operator + : add
inline u16x16 operator+(const u16x16 a, const u16x16 b) {
  return {i16x16(a) + i16x16(b)};
}

// vector operator - : subtract
inline u16x16 operator-(const u16x16 a, const u16x16 b) {
  return {i16x16(a) - i16x16(b)};
}

// vector operator * : multiply
inline u16x16 operator*(const u16x16 a, const u16x16 b) {
  return {i16x16(a) * i16x16(b)};
}

// vector operator / : divide
// See bottom of file

// vector operator >> : shift right logical all elements
inline u16x16 operator>>(const u16x16 a, u32 b) {
  return _mm256_srl_epi16(a, _mm_cvtsi32_si128(i32(b)));
}

// vector operator >> : shift right logical all elements
inline u16x16 operator>>(const u16x16 a, i32 b) {
  return a >> u32(b);
}

// vector operator >>= : shift right artihmetic
inline u16x16& operator>>=(u16x16& a, u32 b) {
  a = a >> b;
  return a;
}

// vector operator << : shift left all elements
inline u16x16 operator<<(const u16x16 a, u32 b) {
  return _mm256_sll_epi16(a, _mm_cvtsi32_si128(i32(b)));
}

// vector operator << : shift left all elements
inline u16x16 operator<<(const u16x16 a, i32 b) {
  return a << u32(b);
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline Mask<16, 16> operator>=(const u16x16 a, const u16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu16_mask(a, b, 5);
#else
  __m256i max_ab = _mm256_max_epu16(a, b); // max(a,b), unsigned
  return _mm256_cmpeq_epi16(a, max_ab); // a == max(a,b)
#endif
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline Mask<16, 16> operator<=(const u16x16 a, const u16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu16_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline Mask<16, 16> operator>(const u16x16 a, const u16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu16_mask(a, b, 6);
#else
  return Mask<16, 16>(i16x16(~(b >= a)));
#endif
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline Mask<16, 16> operator<(const u16x16 a, const u16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu16_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator & : bitwise and
inline u16x16 operator&(const u16x16 a, const u16x16 b) {
  return {si256(a) & si256(b)};
}
inline u16x16 operator&&(const u16x16 a, const u16x16 b) {
  return a & b;
}

// vector operator | : bitwise or
inline u16x16 operator|(const u16x16 a, const u16x16 b) {
  return {si256(a) | si256(b)};
}
inline u16x16 operator||(const u16x16 a, const u16x16 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
inline u16x16 operator^(const u16x16 a, const u16x16 b) {
  return {si256(a) ^ si256(b)};
}

// vector operator ~ : bitwise not
inline u16x16 operator~(const u16x16 a) {
  return {~si256(a)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
inline u16x16 select(const Mask<16, 16> s, const u16x16 a, const u16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mov_epi16(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u16x16 if_add(const Mask<16, 16> f, const u16x16 a, const u16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_add_epi16(a, f, a, b);
#else
  return a + (u16x16(f) & b);
#endif
}

// Conditional subtract
inline u16x16 if_sub(const Mask<16, 16> f, const u16x16 a, const u16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_sub_epi16(a, f, a, b);
#else
  return a - (u16x16(f) & b);
#endif
}

// Conditional multiply
inline u16x16 if_mul(const Mask<16, 16> f, const u16x16 a, const u16x16 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mullo_epi16(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline u16 horizontal_add(const u16x16 a) {
  return u16(horizontal_add(i16x16(a)));
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is zero-extended before addition to avoid overflow
inline u32 horizontal_add_x(const u16x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  const __m256i aeven = _mm256_maskz_mov_epi16(__mmask16(0x5555), a);
#else
  __m256i mask = _mm256_set1_epi32(0x0000FFFF); // mask for even positions
  __m256i aeven = _mm256_and_si256(a, mask); // even numbered elements of a
#endif
  const __m256i aodd = _mm256_srli_epi32(a, 16); // zero extend odd numbered elements
  const __m256i sum1 = _mm256_add_epi32(aeven, aodd); // add even and odd elements
  const __m128i sum2 =
    _mm_add_epi32(_mm256_extracti128_si256(sum1, 1), _mm256_castsi256_si128(sum1));
  const __m128i sum3 = _mm_add_epi32(sum2, _mm_unpackhi_epi64(sum2, sum2));
  const __m128i sum4 = _mm_add_epi32(sum3, _mm_shuffle_epi32(sum3, 1));
  return u16(_mm_cvtsi128_si32(sum4)); // truncate to 16 bits
}

// function add_saturated: add element by element, unsigned with saturation
inline u16x16 add_saturated(const u16x16 a, const u16x16 b) {
  return _mm256_adds_epu16(a, b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
inline u16x16 sub_saturated(const u16x16 a, const u16x16 b) {
  return _mm256_subs_epu16(a, b);
}

// function max: a > b ? a : b
inline u16x16 max(const u16x16 a, const u16x16 b) {
  return _mm256_max_epu16(a, b);
}

// function min: a < b ? a : b
inline u16x16 min(const u16x16 a, const u16x16 b) {
  return _mm256_min_epu16(a, b);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U16X16_HPP
