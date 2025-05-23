#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I64X4_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I64X4_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i64x4.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
// vector operator + : add element by element
inline i64x4 operator+(const i64x4 a, const i64x4 b) {
  return _mm256_add_epi64(a, b);
}
// vector operator += : add
inline i64x4& operator+=(i64x4& a, const i64x4 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i64x4 operator++(i64x4& a, int) {
  i64x4 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i64x4& operator++(i64x4& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i64x4 operator-(const i64x4 a, const i64x4 b) {
  return _mm256_sub_epi64(a, b);
}
// vector operator - : unary minus
inline i64x4 operator-(const i64x4 a) {
  return _mm256_sub_epi64(_mm256_setzero_si256(), a);
}
// vector operator -= : subtract
inline i64x4& operator-=(i64x4& a, const i64x4 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i64x4 operator--(i64x4& a, int) {
  i64x4 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i64x4& operator--(i64x4& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i64x4 operator*(const i64x4 a, const i64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ AVX512VL
  return _mm256_mullo_epi64(a, b);
#else
  // Split into 32-bit multiplies
  __m256i bswap = _mm256_shuffle_epi32(b, 0xB1); // swap H<->L
  __m256i prodlh = _mm256_mullo_epi32(a, bswap); // 32 bit L*H products
  __m256i zero = _mm256_setzero_si256(); // 0
  __m256i prodlh2 = _mm256_hadd_epi32(prodlh, zero); // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
  __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2, 0x73); // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
  __m256i prodll = _mm256_mul_epu32(a, b); // a0Lb0L,a1Lb1L, 64 bit unsigned products
  __m256i prod =
    _mm256_add_epi64(prodll, prodlh3); // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
  return prod;
#endif
}

// vector operator *= : multiply
inline i64x4& operator*=(i64x4& a, const i64x4 b) {
  a = a * b;
  return a;
}

// vector operator << : shift left
inline i64x4 operator<<(const i64x4 a, i32 b) {
  return _mm256_sll_epi64(a, _mm_cvtsi32_si128(b));
}
// vector operator <<= : shift left
inline i64x4& operator<<=(i64x4& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic
inline i64x4 operator>>(const i64x4 a, i32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm256_sra_epi64(a, _mm_cvtsi32_si128(b));
#else
  __m128i bb;
  __m256i shi, slo, sra2;
  if (b <= 32) {
    bb = _mm_cvtsi32_si128(b); // b
    shi = _mm256_sra_epi32(a, bb); // a >> b signed dwords
    slo = _mm256_srl_epi64(a, bb); // a >> b unsigned qwords
  } else { // b > 32
    bb = _mm_cvtsi32_si128(b - 32); // b - 32
    shi = _mm256_srai_epi32(a, 31); // sign of a
    sra2 = _mm256_sra_epi32(a, bb); // a >> (b-32) signed dwords
    slo = _mm256_srli_epi64(sra2, 32); // a >> (b-32) >> 32 (second shift unsigned qword)
  }
  return _mm256_blend_epi32(slo, shi, 0xAA);
#endif
}
// vector operator >>= : shift right arithmetic
inline i64x4& operator>>=(i64x4& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
template<AnyInt64x4 TVec>
inline Mask<64, 4> operator==(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi64_mask(a, b, 0);
#else
  return _mm256_cmpeq_epi64(a, b);
#endif
}

// vector operator != : returns true for elements for which a != b
template<AnyInt64x4 TVec>
inline Mask<64, 4> operator!=(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi64_mask(a, b, 4);
#else
  return Mask<64, 4>(i64x4(~(a == b)));
#endif
}

// vector operator < : returns true for elements for which a < b
inline Mask<64, 4> operator<(const i64x4 a, const i64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi64_mask(a, b, 1);
#else
  return _mm256_cmpgt_epi64(b, a);
#endif
}

// vector operator > : returns true for elements for which a > b
inline Mask<64, 4> operator>(const i64x4 a, const i64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi64_mask(a, b, 6);
#else
  return b < a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline Mask<64, 4> operator>=(const i64x4 a, const i64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi64_mask(a, b, 5);
#else
  return Mask<64, 4>(i64x4(~(a < b)));
#endif
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline Mask<64, 4> operator<=(const i64x4 a, const i64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi64_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline i64x4 operator&(const i64x4 a, const i64x4 b) {
  return i64x4(si256(a) & si256(b));
}
inline i64x4 operator&&(const i64x4 a, const i64x4 b) {
  return a & b;
}
// vector operator &= : bitwise and
inline i64x4& operator&=(i64x4& a, const i64x4 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i64x4 operator|(const i64x4 a, const i64x4 b) {
  return i64x4(si256(a) | si256(b));
}
inline i64x4 operator||(const i64x4 a, const i64x4 b) {
  return a | b;
}
// vector operator |= : bitwise or
inline i64x4& operator|=(i64x4& a, const i64x4 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i64x4 operator^(const i64x4 a, const i64x4 b) {
  return i64x4(si256(a) ^ si256(b));
}
// vector operator ^= : bitwise xor
inline i64x4& operator^=(i64x4& a, const i64x4 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i64x4 operator~(const i64x4 a) {
  return i64x4(~si256(a));
}

// vector operator ! : logical not, returns true for elements == 0
inline Mask<64, 4> operator!(const i64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi64_mask(a, _mm256_setzero_si256(), 0);
#else
  return a == i64x4(_mm256_setzero_si256());
#endif
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
inline i64x4 select(const Mask<64, 4> s, const i64x4 a, const i64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mov_epi64(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i64x4 if_add(const Mask<64, 4> f, const i64x4 a, const i64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_add_epi64(a, f, a, b);
#else
  return a + (i64x4(f) & b);
#endif
}

// Conditional subtract
inline i64x4 if_sub(const Mask<64, 4> f, const i64x4 a, const i64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_sub_epi64(a, f, a, b);
#else
  return a - (i64x4(f) & b);
#endif
}

// Conditional multiply
inline i64x4 if_mul(const Mask<64, 4> f, const i64x4 a, const i64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mullo_epi64(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i64 horizontal_add(const i64x4 a) {
  const __m256i sum1 = _mm256_shuffle_epi32(a, 0x0E); // high element
  const __m256i sum2 = _mm256_add_epi64(a, sum1); // sum
  const __m128i sum3 = _mm256_extracti128_si256(sum2, 1); // get high part
  const __m128i sum4 = _mm_add_epi64(_mm256_castsi256_si128(sum2), sum3); // add low and high parts
  return _mm_cvtsi128_si64(sum4);
}

// function max: a > b ? a : b
inline i64x4 max(const i64x4 a, const i64x4 b) {
  return select(a > b, a, b);
}

// function min: a < b ? a : b
inline i64x4 min(const i64x4 a, const i64x4 b) {
  return select(a < b, a, b);
}

// function abs: a >= 0 ? a : -a
inline i64x4 abs(const i64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm256_abs_epi64(a);
#else
  __m256i sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a); // 0 > a
  __m256i inv = _mm256_xor_si256(a, sign); // invert bits if negative
  return _mm256_sub_epi64(inv, sign); // add 1
#endif
}

// function abs_saturated: same as abs, saturate if overflow
inline i64x4 abs_saturated(const i64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_min_epu64(abs(a), i64x4(0x7FFFFFFFFFFFFFFF));
#else
  __m256i absa = abs(a); // abs(a)
  __m256i overfl = _mm256_cmpgt_epi64(_mm256_setzero_si256(), absa); // 0 > a
  return _mm256_add_epi64(absa, overfl); // subtract 1 if 0x8000000000000000
#endif
}

// function rotate_left all elements
// Use negative count to rotate right
inline i64x4 rotate_left(const i64x4 a, i32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm256_rolv_epi64(a, _mm256_set1_epi64x(i64(b)));
#else
  __m256i left = _mm256_sll_epi64(a, _mm_cvtsi32_si128(b & 0x3F)); // a << b
  __m256i right = _mm256_srl_epi64(a, _mm_cvtsi32_si128((-b) & 0x3F)); // a >> (64 - b)
  __m256i rot = _mm256_or_si256(left, right); // or
  return rot;
#endif
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I64X4_HPP
