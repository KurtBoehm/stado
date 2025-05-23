#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U32X04_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U32X04_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/operations/i32x04.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i32x04.hpp"

namespace stado {
// vector operator + : add
inline u32x4 operator+(const u32x4 a, const u32x4 b) {
  return {i32x4(a) + i32x4(b)};
}
// vector operator += : add
inline u32x4& operator+=(u32x4& a, const u32x4 b) {
  return a = a + b;
}

// vector operator - : subtract
inline u32x4 operator-(const u32x4 a, const u32x4 b) {
  return {i32x4(a) - i32x4(b)};
}

// vector operator * : multiply
inline u32x4 operator*(const u32x4 a, const u32x4 b) {
  return {i32x4(a) * i32x4(b)};
}

// vector operator / : divide. See bottom of file

// vector operator >> : shift right logical all elements
inline u32x4 operator>>(const u32x4 a, u32 b) {
  return _mm_srl_epi32(a, _mm_cvtsi32_si128(i32(b)));
}
// vector operator >> : shift right logical all elements
inline u32x4 operator>>(const u32x4 a, i32 b) {
  return a >> u32(b);
}
// vector operator >>= : shift right logical
inline u32x4& operator>>=(u32x4& a, int b) {
  a = a >> b;
  return a;
}

// vector operator << : shift left all elements
inline u32x4 operator<<(const u32x4 a, u32 b) {
  return {i32x4(a) << i32(b)};
}
// vector operator << : shift left all elements
inline u32x4 operator<<(const u32x4 a, i32 b) {
  return {i32x4(a) << b};
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline Mask<32, 4> operator>(const u32x4 a, const u32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epu32_mask(a, b, 6);
#elif defined(__XOP__) // AMD XOP instruction set
  return (Mask<32, 4>)_mm_comgt_epu32(a, b);
#else // SSE2 instruction set
  __m128i signbit = _mm_set1_epi32(i32(0x80000000));
  __m128i a1 = _mm_xor_si128(a, signbit);
  __m128i b1 = _mm_xor_si128(b, signbit);
  return (Mask<32, 4>)_mm_cmpgt_epi32(a1, b1); // signed compare
#endif
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline Mask<32, 4> operator<(const u32x4 a, const u32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epu32_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline Mask<32, 4> operator>=(const u32x4 a, const u32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epu32_mask(a, b, 5);
#else
#ifdef __XOP__ // AMD XOP instruction set
  return (Mask<32, 4>)_mm_comge_epu32(a, b);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1 // SSE4.1
  __m128i max_ab = _mm_max_epu32(a, b); // max(a,b), unsigned
  return (Mask<32, 4>)_mm_cmpeq_epi32(a, max_ab); // a == max(a,b)
#else // SSE2 instruction set
  return Mask<32, 4>(i32x4(~(b > a)));
#endif
#endif
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline Mask<32, 4> operator<=(const u32x4 a, const u32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epu32_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline u32x4 operator&(const u32x4 a, const u32x4 b) {
  return {si128(a) & si128(b)};
}
inline u32x4 operator&&(const u32x4 a, const u32x4 b) {
  return a & b;
}

// vector operator | : bitwise or
inline u32x4 operator|(const u32x4 a, const u32x4 b) {
  return {si128(a) | si128(b)};
}
inline u32x4 operator||(const u32x4 a, const u32x4 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
inline u32x4 operator^(const u32x4 a, const u32x4 b) {
  return {si128(a) ^ si128(b)};
}

// vector operator ~ : bitwise not
inline u32x4 operator~(const u32x4 a) {
  return {~si128(a)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
inline u32x4 select(const Mask<32, 4> s, const u32x4 a, const u32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mov_epi32(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u32x4 if_add(const Mask<32, 4> f, const u32x4 a, const u32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_add_epi32(a, f, a, b);
#else
  return a + (u32x4(f) & b);
#endif
}

// Conditional sub: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
inline u32x4 if_sub(const Mask<32, 4> f, const u32x4 a, const u32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_sub_epi32(a, f, a, b);
#else
  return a - (u32x4(f) & b);
#endif
}

// Conditional mul: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
inline u32x4 if_mul(const Mask<32, 4> f, const u32x4 a, const u32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mullo_epi32(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline u32 horizontal_add(const u32x4 a) {
  return u32(horizontal_add(i32x4(a)));
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Elements are zero extended before adding to avoid overflow
inline u64 horizontal_add_x(const u32x4 a) {
#ifdef __XOP__ // AMD XOP instruction set
  __m128i sum1 = _mm_haddq_epu32(a);
#else // SSE2
  const __m128i zero = _mm_setzero_si128(); // 0
  const __m128i a01 = _mm_unpacklo_epi32(a, zero); // zero-extended a0, a1
  const __m128i a23 = _mm_unpackhi_epi32(a, zero); // zero-extended a2, a3
  const __m128i sum1 = _mm_add_epi64(a01, a23); // add
#endif
  const __m128i sum2 = _mm_unpackhi_epi64(sum1, sum1); // high qword
  const __m128i sum3 = _mm_add_epi64(sum1, sum2); // add
  return u64(_mm_cvtsi128_si64(sum3));
}

// function add_saturated: add element by element, unsigned with saturation
inline u32x4 add_saturated(const u32x4 a, const u32x4 b) {
  const u32x4 sum = a + b;
  const auto aorb = u32x4(a | b);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  const CompactMask<4> overflow = _mm_cmp_epu32_mask(sum, aorb, 1);
  return _mm_mask_set1_epi32(sum, overflow, -1);
#else
  u32x4 overflow = u32x4(sum < aorb); // overflow if a + b < (a | b)
  return {sum | overflow}; // return 0xFFFFFFFF if overflow
#endif
}

// function sub_saturated: subtract element by element, unsigned with saturation
inline u32x4 sub_saturated(const u32x4 a, const u32x4 b) {
  const u32x4 diff = a - b;
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  const CompactMask<4> nunderflow = _mm_cmp_epu32_mask(diff, a, 2); // not underflow if a - b <= a
  return _mm_maskz_mov_epi32(nunderflow, diff); // zero if underflow
#else
  u32x4 underflow = u32x4(diff > a); // underflow if a - b > a
  return _mm_andnot_si128(underflow, diff); // return 0 if underflow
#endif
}

// function max: a > b ? a : b
inline u32x4 max(const u32x4 a, const u32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1 // SSE4.1
  return _mm_max_epu32(a, b);
#else // SSE2
  return select(a > b, a, b);
#endif
}

// function min: a < b ? a : b
inline u32x4 min(const u32x4 a, const u32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1 // SSE4.1
  return _mm_min_epu32(a, b);
#else // SSE2
  return select(a > b, b, a);
#endif
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U32X04_HPP
