#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U64X04_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U64X04_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/operations/i64x04.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i32x08.hpp"
#include "stado/vector/native/types/i64x04.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
// vector operator + : add
inline u64x4 operator+(const u64x4 a, const u64x4 b) {
  return {i64x4(a) + i64x4(b)};
}
inline u64x4& operator+=(u64x4& a, const u64x4 b) {
  return a = a + b;
}

// vector operator - : subtract
inline u64x4 operator-(const u64x4 a, const u64x4 b) {
  return {i64x4(a) - i64x4(b)};
}

// vector operator * : multiply element by element
inline u64x4 operator*(const u64x4 a, const u64x4 b) {
  return {i64x4(a) * i64x4(b)};
}

// vector operator >> : shift right logical all elements
inline u64x4 operator>>(const u64x4 a, u32 b) {
  return _mm256_srl_epi64(a, _mm_cvtsi32_si128(i32(b)));
}

// vector operator >> : shift right logical all elements
inline u64x4 operator>>(const u64x4 a, i32 b) {
  return a >> u32(b);
}
// vector operator >>= : shift right artihmetic
inline u64x4& operator>>=(u64x4& a, u32 b) {
  a = a >> b;
  return a;
}

// vector operator << : shift left all elements
inline u64x4 operator<<(const u64x4 a, u32 b) {
  return {i64x4(a) << i32(b)};
}
// vector operator << : shift left all elements
inline u64x4 operator<<(const u64x4 a, i32 b) {
  return {i64x4(a) << b};
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline Mask<64, 4> operator>(const u64x4 a, const u64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu64_mask(a, b, 6);
#else
  __m256i sign64 = u64x4(0x8000000000000000);
  __m256i aflip = _mm256_xor_si256(a, sign64);
  __m256i bflip = _mm256_xor_si256(b, sign64);
  i64x4 cmp = _mm256_cmpgt_epi64(aflip, bflip);
  return Mask<64, 4>(cmp);
#endif
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline Mask<64, 4> operator<(const u64x4 a, const u64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu64_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline Mask<64, 4> operator>=(const u64x4 a, const u64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu64_mask(a, b, 5);
#else
  return Mask<64, 4>(i64x4(~(b > a)));
#endif
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline Mask<64, 4> operator<=(const u64x4 a, const u64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu64_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline u64x4 operator&(const u64x4 a, const u64x4 b) {
  return {si256(a) & si256(b)};
}
inline u64x4 operator&&(const u64x4 a, const u64x4 b) {
  return a & b;
}

// vector operator | : bitwise or
inline u64x4 operator|(const u64x4 a, const u64x4 b) {
  return {si256(a) | si256(b)};
}
inline u64x4 operator||(const u64x4 a, const u64x4 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
inline u64x4 operator^(const u64x4 a, const u64x4 b) {
  return {si256(a) ^ si256(b)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
inline u64x4 select(const Mask<64, 4> s, const u64x4 a, const u64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mov_epi64(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u64x4 if_add(const Mask<64, 4> f, const u64x4 a, const u64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_add_epi64(a, f, a, b);
#else
  return a + (u64x4(f) & b);
#endif
}

// Conditional subtract
inline u64x4 if_sub(const Mask<64, 4> f, const u64x4 a, const u64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_sub_epi64(a, f, a, b);
#else
  return a - (u64x4(f) & b);
#endif
}

// Conditional multiply
inline u64x4 if_mul(const Mask<64, 4> f, const u64x4 a, const u64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mullo_epi64(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline u64 horizontal_add(const u64x4 a) {
  return u64(horizontal_add(i64x4(a)));
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are sign/zero extended before adding to avoid overflow
inline i64 horizontal_add_x(const i32x8 a) {
  // sign of all elements
  const __m256i signs = _mm256_srai_epi32(a, 31);
  // sign-extended a0, a1, a4, a5
  const i64x4 a01 = _mm256_unpacklo_epi32(a, signs);
  // sign-extended a2, a3, a6, a7
  const i64x4 a23 = _mm256_unpackhi_epi32(a, signs);
  return horizontal_add(a01 + a23);
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are sign/zero extended before adding to avoid overflow
inline u64 horizontal_add_x(const u32x8 a) {
  // 0
  const __m256i zero = _mm256_setzero_si256();
  // zero-extended a0, a1
  const __m256i a01 = _mm256_unpacklo_epi32(a, zero);
  // zero-extended a2, a3
  const __m256i a23 = _mm256_unpackhi_epi32(a, zero);
  return u64(horizontal_add(i64x4(a01) + i64x4(a23)));
}

// function max: a > b ? a : b
inline u64x4 max(const u64x4 a, const u64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm256_max_epu64(a, b);
#else
  return {select(a > b, a, b)};
#endif
}

// function min: a < b ? a : b
inline u64x4 min(const u64x4 a, const u64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm256_min_epu64(a, b);
#else
  return {select(a > b, b, a)};
#endif
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U64X04_HPP
