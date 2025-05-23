#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I64X2_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I64X2_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i64x2.hpp"

namespace stado {
// vector operator + : add element by element
inline i64x2 operator+(const i64x2 a, const i64x2 b) {
  return _mm_add_epi64(a, b);
}
// vector operator += : add
inline i64x2& operator+=(i64x2& a, const i64x2 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i64x2 operator++(i64x2& a, int) {
  i64x2 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i64x2& operator++(i64x2& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i64x2 operator-(const i64x2 a, const i64x2 b) {
  return _mm_sub_epi64(a, b);
}
// vector operator - : unary minus
inline i64x2 operator-(const i64x2 a) {
  return _mm_sub_epi64(_mm_setzero_si128(), a);
}
// vector operator -= : subtract
inline i64x2& operator-=(i64x2& a, const i64x2 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i64x2 operator--(i64x2& a, int) {
  i64x2 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i64x2& operator--(i64x2& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i64x2 operator*(const i64x2 a, const i64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ AVX512VL
  return _mm_mullo_epi64(a, b);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
  // Split into 32-bit multiplies
  __m128i bswap = _mm_shuffle_epi32(b, 0xB1); // b0H,b0L,b1H,b1L (swap H<->L)
  __m128i prodlh = _mm_mullo_epi32(a, bswap); // a0Lb0H,a0Hb0L,a1Lb1H,a1Hb1L, 32 bit L*H products
  __m128i zero = _mm_setzero_si128(); // 0
  __m128i prodlh2 = _mm_hadd_epi32(prodlh, zero); // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
  __m128i prodlh3 = _mm_shuffle_epi32(prodlh2, 0x73); // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
  __m128i prodll = _mm_mul_epu32(a, b); // a0Lb0L,a1Lb1L, 64 bit unsigned products
  __m128i prod =
    _mm_add_epi64(prodll, prodlh3); // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
  return prod;
#else // SSE2
  i64 aa[2], bb[2];
  a.store(aa); // split into elements
  b.store(bb);
  return i64x2(aa[0] * bb[0],
               aa[1] * bb[1]); // multiply elements separetely
#endif
}

// vector operator *= : multiply
inline i64x2& operator*=(i64x2& a, const i64x2 b) {
  a = a * b;
  return a;
}

// vector operator << : shift left
inline i64x2 operator<<(const i64x2 a, i32 b) {
  return _mm_sll_epi64(a, _mm_cvtsi32_si128(b));
}

// vector operator <<= : shift left
inline i64x2& operator<<=(i64x2& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic
inline i64x2 operator>>(const i64x2 a, i32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm_sra_epi64(a, _mm_cvtsi32_si128(b));
#else
  __m128i bb, shi, slo, sra2;
  if (b <= 32) {
    bb = _mm_cvtsi32_si128(b); // b
    shi = _mm_sra_epi32(a, bb); // a >> b signed dwords
    slo = _mm_srl_epi64(a, bb); // a >> b unsigned qwords
  } else { // b > 32
    bb = _mm_cvtsi32_si128(b - 32); // b - 32
    shi = _mm_srai_epi32(a, 31); // sign of a
    sra2 = _mm_sra_epi32(a, bb); // a >> (b-32) signed dwords
    slo = _mm_srli_epi64(sra2, 32); // a >> (b-32) >> 32 (second shift unsigned qword)
  }
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_blend_epi16(slo, shi, 0xCC);
#else
  __m128i mask = _mm_setr_epi32(0, -1, 0, -1); // mask for high part containing only sign
  return selectb(mask, shi, slo);
#endif
#endif
}

// vector operator >>= : shift right arithmetic
inline i64x2& operator>>=(i64x2& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
template<AnyInt64x2 TVec>
inline Mask<64, 2> operator==(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // broad boolean vectors
  return _mm_cmp_epi64_mask(a, b, 0);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_cmpeq_epi64(a, b);
#else // SSE2
  // no 64 compare instruction. Do two 32 bit compares
  __m128i com32 = _mm_cmpeq_epi32(a, b); // 32 bit compares
  __m128i com32s = _mm_shuffle_epi32(com32, 0xB1); // swap low and high dwords
  __m128i test = _mm_and_si128(com32, com32s); // low & high
  __m128i teste = _mm_srai_epi32(test, 31); // extend sign bit to 32 bits
  __m128i testee = _mm_shuffle_epi32(teste, 0xF5); // extend sign bit to 64 bits
  return Mask<64, 2>(i64x2(testee));
#endif
}

// vector operator != : returns true for elements for which a != b
template<AnyInt64x2 TVec>
inline Mask<64, 2> operator!=(const TVec a, const TVec b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // broad boolean vectors
  return _mm_cmp_epi64_mask(a, b, 4);
#elif defined(__XOP__) // AMD XOP instruction set
  return Mask<64, 2>(_mm_comneq_epi64(a, b));
#else // SSE2 instruction set
  return Mask<64, 2>(i64x2(~(a == b)));
#endif
}

// vector operator < : returns true for elements for which a < b
inline Mask<64, 2> operator<(const i64x2 a, const i64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // broad boolean vectors
  return _mm_cmp_epi64_mask(a, b, 1);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_2
  return Mask<64, 2>(i64x2(_mm_cmpgt_epi64(b, a)));
#else // SSE2
  // no 64 compare instruction. Subtract
  __m128i s = _mm_sub_epi64(a, b); // a-b
  // a < b if a and b have same sign and s < 0 or (a < 0 and b >= 0)
  // The latter () corrects for overflow
  __m128i axb = _mm_xor_si128(a, b); // a ^ b
  __m128i anb = _mm_andnot_si128(b, a); // a & ~b
  __m128i snaxb = _mm_andnot_si128(axb, s); // s & ~(a ^ b)
  __m128i or1 = _mm_or_si128(anb, snaxb); // (a & ~b) | (s & ~(a ^ b))
  __m128i teste = _mm_srai_epi32(or1, 31); // extend sign bit to 32 bits
  __m128i testee = _mm_shuffle_epi32(teste, 0xF5); // extend sign bit to 64 bits
  return testee;
#endif
}

// vector operator > : returns true for elements for which a > b
inline Mask<64, 2> operator>(const i64x2 a, const i64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi64_mask(a, b, 6);
#else
  return b < a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline Mask<64, 2> operator>=(const i64x2 a, const i64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // broad boolean vectors
  return _mm_cmp_epi64_mask(a, b, 5);
#elif defined(__XOP__) // AMD XOP instruction set
  return Mask<64, 2>(_mm_comge_epi64(a, b));
#else // SSE2 instruction set
  return Mask<64, 2>(i64x2(~(a < b)));
#endif
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline Mask<64, 2> operator<=(const i64x2 a, const i64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_epi64_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline i64x2 operator&(const i64x2 a, const i64x2 b) {
  return i64x2(si128(a) & si128(b));
}
inline i64x2 operator&&(const i64x2 a, const i64x2 b) {
  return a & b;
}
// vector operator &= : bitwise and
inline i64x2& operator&=(i64x2& a, const i64x2 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i64x2 operator|(const i64x2 a, const i64x2 b) {
  return i64x2(si128(a) | si128(b));
}
inline i64x2 operator||(const i64x2 a, const i64x2 b) {
  return a | b;
}
// vector operator |= : bitwise or
inline i64x2& operator|=(i64x2& a, const i64x2 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i64x2 operator^(const i64x2 a, const i64x2 b) {
  return i64x2(si128(a) ^ si128(b));
}
// vector operator ^= : bitwise xor
inline i64x2& operator^=(i64x2& a, const i64x2 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i64x2 operator~(const i64x2 a) {
  return i64x2(~si128(a));
}

// vector operator ! : logical not, returns true for elements == 0
inline Mask<64, 2> operator!(const i64x2 a) {
  return a == i64x2(_mm_setzero_si128());
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
inline i64x2 select(const Mask<64, 2> s, const i64x2 a, const i64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mov_epi64(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i64x2 if_add(const Mask<64, 2> f, const i64x2 a, const i64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_add_epi64(a, f, a, b);
#else
  return a + (i64x2(f) & b);
#endif
}

// Conditional sub: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
inline i64x2 if_sub(const Mask<64, 2> f, const i64x2 a, const i64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_sub_epi64(a, f, a, b);
#else
  return a - (i64x2(f) & b);
#endif
}

// Conditional mul: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
inline i64x2 if_mul(const Mask<64, 2> f, const i64x2 a, const i64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mullo_epi64(a, f, a, b);
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i64 horizontal_add(const i64x2 a) {
  const __m128i sum1 = _mm_unpackhi_epi64(a, a); // high element
  const __m128i sum2 = _mm_add_epi64(a, sum1); // sum
  return _mm_cvtsi128_si64(sum2);
}

// function max: a > b ? a : b
inline i64x2 max(const i64x2 a, const i64x2 b) {
  return select(a > b, a, b);
}

// function min: a < b ? a : b
inline i64x2 min(const i64x2 a, const i64x2 b) {
  return select(a < b, a, b);
}

// function abs: a >= 0 ? a : -a
inline i64x2 abs(const i64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm_abs_epi64(a);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_2
  __m128i sign = _mm_cmpgt_epi64(_mm_setzero_si128(), a); // 0 > a
  __m128i inv = _mm_xor_si128(a, sign); // invert bits if negative
  return _mm_sub_epi64(inv, sign); // add 1
#else // SSE2
  __m128i signh = _mm_srai_epi32(a, 31); // sign in high dword
  __m128i sign = _mm_shuffle_epi32(signh, 0xF5); // copy sign to low dword
  __m128i inv = _mm_xor_si128(a, sign); // invert bits if negative
  return _mm_sub_epi64(inv, sign); // add 1
#endif
}

// function abs_saturated: same as abs, saturate if overflow
inline i64x2 abs_saturated(const i64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_min_epu64(abs(a), i64x2(0x7FFFFFFFFFFFFFFF));
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_2
  __m128i absa = abs(a); // abs(a)
  __m128i overfl = _mm_cmpgt_epi64(_mm_setzero_si128(), absa); // 0 > a
  return _mm_add_epi64(absa, overfl); // subtract 1 if 0x8000000000000000
#else // SSE2
  __m128i absa = abs(a); // abs(a)
  __m128i signh = _mm_srai_epi32(absa, 31); // sign in high dword
  __m128i overfl = _mm_shuffle_epi32(signh, 0xF5); // copy sign to low dword
  return _mm_add_epi64(absa, overfl); // subtract 1 if 0x8000000000000000
#endif
}

// function rotate_left all elements
// Use negative count to rotate right
inline i64x2 rotate_left(const i64x2 a, i32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm_rolv_epi64(a, _mm_set1_epi64x(i64(b)));
#elif defined(__XOP__) // AMD XOP instruction set
  return (i64x2)_mm_rot_epi64(a, i64x2(b));
#else // SSE2 instruction set
  __m128i left = _mm_sll_epi64(a, _mm_cvtsi32_si128(b & 0x3F)); // a << b
  __m128i right = _mm_srl_epi64(a, _mm_cvtsi32_si128((-b) & 0x3F)); // a >> (64 - b)
  __m128i rot = _mm_or_si128(left, right); // or
  return (i64x2)rot;
#endif
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I64X2_HPP
