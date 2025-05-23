#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U64X2_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U64X2_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/operations/i64x2.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i64x2.hpp"

namespace stado {
// vector operator + : add
inline u64x2 operator+(const u64x2 a, const u64x2 b) {
  return {i64x2(a) + i64x2(b)};
}

// vector operator - : subtract
inline u64x2 operator-(const u64x2 a, const u64x2 b) {
  return {i64x2(a) - i64x2(b)};
}

// vector operator * : multiply element by element
inline u64x2 operator*(const u64x2 a, const u64x2 b) {
  return {i64x2(a) * i64x2(b)};
}

// vector operator >> : shift right logical all elements
inline u64x2 operator>>(const u64x2 a, u32 b) {
  return _mm_srl_epi64(a, _mm_cvtsi32_si128(i32(b)));
}

// vector operator >> : shift right logical all elements
inline u64x2 operator>>(const u64x2 a, i32 b) {
  return a >> u32(b);
}

// vector operator >>= : shift right logical
inline u64x2& operator>>=(u64x2& a, int b) {
  a = a >> b;
  return a;
}

// vector operator << : shift left all elements
inline u64x2 operator<<(const u64x2 a, u32 b) {
  return {i64x2(a) << i32(b)};
}

// vector operator << : shift left all elements
inline u64x2 operator<<(const u64x2 a, i32 b) {
  return {i64x2(a) << b};
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline Mask<64, 2> operator>(const u64x2 a, const u64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // broad boolean vectors
  return _mm_cmp_epu64_mask(a, b, 6);
#elif defined(__XOP__) // AMD XOP instruction set
  return Mask<64, 2>(_mm_comgt_epu64(a, b));
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_2 // SSE4.2
  __m128i sign64 = _mm_setr_epi32(0, 0x80000000, 0, 0x80000000);
  __m128i aflip = _mm_xor_si128(a, sign64); // flip sign bits to use signed compare
  __m128i bflip = _mm_xor_si128(b, sign64);
  i64x2 cmp = _mm_cmpgt_epi64(aflip, bflip);
  return Mask<64, 2>(cmp);
#else // SSE2 instruction set
  __m128i sign32 = _mm_set1_epi32(0x80000000); // sign bit of each dword
  __m128i aflip = _mm_xor_si128(a, sign32); // a with sign bits flipped to use signed compare
  __m128i bflip = _mm_xor_si128(b, sign32); // b with sign bits flipped to use signed compare
  __m128i equal = _mm_cmpeq_epi32(a, b); // a == b, dwords
  __m128i bigger = _mm_cmpgt_epi32(aflip, bflip); // a > b, dwords
  __m128i biggerl = _mm_shuffle_epi32(bigger, 0xA0); // a > b, low dwords copied to high dwords
  __m128i eqbig = _mm_and_si128(equal, biggerl); // high part equal and low part bigger
  __m128i hibig =
    _mm_or_si128(bigger, eqbig); // high part bigger or high part equal and low part bigger
  __m128i big = _mm_shuffle_epi32(hibig, 0xF5); // result copied to low part
  return Mask<64, 2>(i64x2(big));
#endif
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline Mask<64, 2> operator<(const u64x2 a, const u64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epu64_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline Mask<64, 2> operator>=(const u64x2 a, const u64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // broad boolean vectors
  return _mm_cmp_epu64_mask(a, b, 5);
#elif defined(__XOP__) // AMD XOP instruction set
  return Mask<64, 2>(_mm_comge_epu64(a, b));
#else // SSE2 instruction set
  return Mask<64, 2>(i64x2(~(b > a)));
#endif
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline Mask<64, 2> operator<=(const u64x2 a, const u64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epu64_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline u64x2 operator&(const u64x2 a, const u64x2 b) {
  return {si128(a) & si128(b)};
}
inline u64x2 operator&&(const u64x2 a, const u64x2 b) {
  return a & b;
}

// vector operator | : bitwise or
inline u64x2 operator|(const u64x2 a, const u64x2 b) {
  return {si128(a) | si128(b)};
}
inline u64x2 operator||(const u64x2 a, const u64x2 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
inline u64x2 operator^(const u64x2 a, const u64x2 b) {
  return {si128(a) ^ si128(b)};
}

// vector operator ~ : bitwise not
inline u64x2 operator~(const u64x2 a) {
  return {~si128(a)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 2; i++) result[i] = s[i] ? a[i] : b[i];
inline u64x2 select(const Mask<64, 2> s, const u64x2 a, const u64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mov_epi64(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u64x2 if_add(const Mask<64, 2> f, const u64x2 a, const u64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_add_epi64(a, f, a, b);
#else
  return a + (u64x2(f) & b);
#endif
}

// Conditional sub: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
inline u64x2 if_sub(const Mask<64, 2> f, const u64x2 a, const u64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_sub_epi64(a, f, a, b);
#else
  return a - (u64x2(f) & b);
#endif
}

// Conditional mul: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
inline u64x2 if_mul(const Mask<64, 2> f, const u64x2 a, const u64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mullo_epi64(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline u64 horizontal_add(const u64x2 a) {
  return u64(horizontal_add(i64x2(a)));
}

// function max: a > b ? a : b
inline u64x2 max(const u64x2 a, const u64x2 b) {
  return select(a > b, a, b);
}

// function min: a < b ? a : b
inline u64x2 min(const u64x2 a, const u64x2 b) {
  return select(a > b, b, a);
}
} // namespace stado
#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U64X2_HPP
