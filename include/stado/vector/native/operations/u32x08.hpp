#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U32X08_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U32X08_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/operations/i32x08.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i32x08.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
// vector operator + : add
inline u32x8 operator+(const u32x8 a, const u32x8 b) {
  return {i32x8(a) + i32x8(b)};
}
// vector operator += : add
inline u32x8& operator+=(u32x8& a, const u32x8 b) {
  a = a + b;
  return a;
}

// vector operator - : subtract
inline u32x8 operator-(const u32x8 a, const u32x8 b) {
  return {i32x8(a) - i32x8(b)};
}

// vector operator * : multiply
inline u32x8 operator*(const u32x8 a, const u32x8 b) {
  return {i32x8(a) * i32x8(b)};
}

// vector operator / : divide
// See bottom of file

// vector operator >> : shift right logical all elements
inline u32x8 operator>>(const u32x8 a, u32 b) {
  return _mm256_srl_epi32(a, _mm_cvtsi32_si128(i32(b)));
}

// vector operator >> : shift right logical all elements
inline u32x8 operator>>(const u32x8 a, i32 b) {
  return a >> u32(b);
}
// vector operator >>= : shift right logical
inline u32x8& operator>>=(u32x8& a, u32 b) {
  a = a >> b;
  return a;
}

// vector operator << : shift left all elements
inline u32x8 operator<<(const u32x8 a, u32 b) {
  return {i32x8(a) << i32(b)};
}
// vector operator << : shift left all elements
inline u32x8 operator<<(const u32x8 a, i32 b) {
  return {i32x8(a) << b};
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline Mask<32, 8> operator>(const u32x8 a, const u32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu32_mask(a, b, 6);
#else
  __m256i signbit = _mm256_set1_epi32(i32(0x80000000));
  __m256i a1 = _mm256_xor_si256(a, signbit);
  __m256i b1 = _mm256_xor_si256(b, signbit);
  return _mm256_cmpgt_epi32(a1, b1); // signed compare
#endif
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline Mask<32, 8> operator<(const u32x8 a, const u32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu32_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline Mask<32, 8> operator>=(const u32x8 a, const u32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu32_mask(a, b, 5);
#else
  __m256i max_ab = _mm256_max_epu32(a, b); // max(a,b), unsigned
  return _mm256_cmpeq_epi32(a, max_ab); // a == max(a,b)
#endif
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline Mask<32, 8> operator<=(const u32x8 a, const u32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu32_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline u32x8 operator&(const u32x8 a, const u32x8 b) {
  return {si256(a) & si256(b)};
}
inline u32x8 operator&&(const u32x8 a, const u32x8 b) {
  return a & b;
}

// vector operator | : bitwise or
inline u32x8 operator|(const u32x8 a, const u32x8 b) {
  return {si256(a) | si256(b)};
}
inline u32x8 operator||(const u32x8 a, const u32x8 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
inline u32x8 operator^(const u32x8 a, const u32x8 b) {
  return {si256(a) ^ si256(b)};
}

// vector operator ~ : bitwise not
inline u32x8 operator~(const u32x8 a) {
  return {~si256(a)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
inline u32x8 select(const Mask<32, 8> s, const u32x8 a, const u32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mov_epi32(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u32x8 if_add(const Mask<32, 8> f, const u32x8 a, const u32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_add_epi32(a, f, a, b);
#else
  return a + (u32x8(f) & b);
#endif
}

// Conditional subtract
inline u32x8 if_sub(const Mask<32, 8> f, const u32x8 a, const u32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_sub_epi32(a, f, a, b);
#else
  return a - (u32x8(f) & b);
#endif
}

// Conditional multiply
inline u32x8 if_mul(const Mask<32, 8> f, const u32x8 a, const u32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mullo_epi32(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline u32 horizontal_add(const u32x8 a) {
  return u32(horizontal_add(i32x8(a)));
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are zero extended before adding to avoid overflow
// inline u64 horizontal_add_x (u32x8 const a); // defined later

// function add_saturated: add element by element, unsigned with saturation
inline u32x8 add_saturated(const u32x8 a, const u32x8 b) {
  const u32x8 sum = a + b;
  const auto aorb = u32x8(a | b);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  const CompactMask<8> overflow = _mm256_cmp_epu32_mask(sum, aorb, 1);
  return _mm256_mask_set1_epi32(sum, overflow, -1);
#else
  u32x8 overflow = u32x8(sum < aorb); // overflow if a + b < (a | b)
  return u32x8(sum | overflow); // return 0xFFFFFFFF if overflow
#endif
}

// function sub_saturated: subtract element by element, unsigned with saturation
inline u32x8 sub_saturated(const u32x8 a, const u32x8 b) {
  const u32x8 diff = a - b;
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  const CompactMask<8> nunderflow =
    _mm256_cmp_epu32_mask(diff, a, 2); // not underflow if a - b <= a
  return _mm256_maskz_mov_epi32(nunderflow, diff); // zero if underflow
#else
  u32x8 underflow = u32x8(diff > a); // underflow if a - b > a
  return _mm256_andnot_si256(underflow, diff); // return 0 if underflow
#endif
}

// function max: a > b ? a : b
inline u32x8 max(const u32x8 a, const u32x8 b) {
  return _mm256_max_epu32(a, b);
}

// function min: a < b ? a : b
inline u32x8 min(const u32x8 a, const u32x8 b) {
  return _mm256_min_epu32(a, b);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U32X08_HPP
