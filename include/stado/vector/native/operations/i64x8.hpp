#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I64X8_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I64X8_HPP

#include <immintrin.h>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/08.hpp"
#include "stado/vector/native/operations/i64x4.hpp"
#include "stado/vector/native/types/i32x16.hpp"
#include "stado/vector/native/types/i64x8.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
// vector operator + : add element by element
inline i64x8 operator+(const i64x8 a, const i64x8 b) {
  return _mm512_add_epi64(a, b);
}
// vector operator += : add
inline i64x8& operator+=(i64x8& a, const i64x8 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i64x8 operator++(i64x8& a, int) {
  i64x8 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i64x8& operator++(i64x8& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i64x8 operator-(const i64x8 a, const i64x8 b) {
  return _mm512_sub_epi64(a, b);
}
// vector operator - : unary minus
inline i64x8 operator-(const i64x8 a) {
  return _mm512_sub_epi64(_mm512_setzero_epi32(), a);
}
// vector operator -= : subtract
inline i64x8& operator-=(i64x8& a, const i64x8 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i64x8 operator--(i64x8& a, int) {
  i64x8 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i64x8& operator--(i64x8& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i64x8 operator*(const i64x8 a, const i64x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_mullo_epi64(a, b);
#elif defined(__INTEL_COMPILER)
  return _mm512_mullox_epi64(a, b); // _mm512_mullox_epi64 missing in gcc
#else
  // instruction does not exist. Split into 32-bit multiplications
  //__m512i ahigh = _mm512_shuffle_epi32(a, 0xB1);       // swap H<->L
  __m512i ahigh = _mm512_srli_epi64(a, 32); // high 32 bits of each a
  __m512i bhigh = _mm512_srli_epi64(b, 32); // high 32 bits of each b
  __m512i prodahb = _mm512_mul_epu32(ahigh, b); // ahigh*b
  __m512i prodbha = _mm512_mul_epu32(bhigh, a); // bhigh*a
  __m512i prodhl = _mm512_add_epi64(prodahb, prodbha); // sum of high*low products
  __m512i prodhi = _mm512_slli_epi64(prodhl, 32); // same, shifted high
  __m512i prodll = _mm512_mul_epu32(a, b); // alow*blow = 64 bit unsigned products
  __m512i prod = _mm512_add_epi64(prodll, prodhi); // low*low+(high*low)<<32
  return prod;
#endif
}

// vector operator *= : multiply
inline i64x8& operator*=(i64x8& a, const i64x8 b) {
  a = a * b;
  return a;
}

// vector operator << : shift left
inline i64x8 operator<<(const i64x8 a, i32 b) {
  return _mm512_sll_epi64(a, _mm_cvtsi32_si128(b));
}
// vector operator <<= : shift left
inline i64x8& operator<<=(i64x8& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic
inline i64x8 operator>>(const i64x8 a, i32 b) {
  return _mm512_sra_epi64(a, _mm_cvtsi32_si128(b));
}
// vector operator >>= : shift right arithmetic
inline i64x8& operator>>=(i64x8& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
template<AnyInt64x8 TVec>
inline CompactMask<8> operator==(const TVec a, const TVec b) {
  return CompactMask<8>(_mm512_cmpeq_epi64_mask(a, b));
}

// vector operator != : returns true for elements for which a != b
template<AnyInt64x8 TVec>
inline CompactMask<8> operator!=(const TVec a, const TVec b) {
  return CompactMask<8>(_mm512_cmpneq_epi64_mask(a, b));
}

// vector operator < : returns true for elements for which a < b
inline CompactMask<8> operator<(const i64x8 a, const i64x8 b) {
  return _mm512_cmp_epi64_mask(a, b, 1);
}

// vector operator > : returns true for elements for which a > b
inline CompactMask<8> operator>(const i64x8 a, const i64x8 b) {
  return _mm512_cmp_epi64_mask(a, b, 6);
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline CompactMask<8> operator>=(const i64x8 a, const i64x8 b) {
  return _mm512_cmp_epi64_mask(a, b, 5);
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline CompactMask<8> operator<=(const i64x8 a, const i64x8 b) {
  return _mm512_cmp_epi64_mask(a, b, 2);
}

// vector operator & : bitwise and
inline i64x8 operator&(const i64x8 a, const i64x8 b) {
  return _mm512_and_epi32(a, b);
}
// vector operator &= : bitwise and
inline i64x8& operator&=(i64x8& a, const i64x8 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i64x8 operator|(const i64x8 a, const i64x8 b) {
  return _mm512_or_epi32(a, b);
}
// vector operator |= : bitwise or
inline i64x8& operator|=(i64x8& a, const i64x8 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i64x8 operator^(const i64x8 a, const i64x8 b) {
  return _mm512_xor_epi32(a, b);
}
// vector operator ^= : bitwise xor
inline i64x8& operator^=(i64x8& a, const i64x8 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i64x8 operator~(const i64x8 a) {
  return i64x8(~i32x16(a));
  // return _mm512_ternarylogic_epi64(_mm512_undefined_epi32(), _mm512_undefined_epi32(), a, 0x55);
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
inline i64x8 select(const CompactMask<8> s, const i64x8 a, const i64x8 b) {
  // avoid warning in MS compiler if STADO_INSTRUCTION_SET = STADO_AVX512F by casting mask to u8,
  // while __mmask8 is not supported in AVX512F
  return _mm512_mask_mov_epi64(b, u8(s), a);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i64x8 if_add(const CompactMask<8> f, const i64x8 a, const i64x8 b) {
  return _mm512_mask_add_epi64(a, u8(f), a, b);
}

// Conditional subtract
inline i64x8 if_sub(const CompactMask<8> f, const i64x8 a, const i64x8 b) {
  return _mm512_mask_sub_epi64(a, u8(f), a, b);
}

// Conditional multiply
inline i64x8 if_mul(const CompactMask<8> f, const i64x8 a, const i64x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm512_mask_mullo_epi64(a, f, a, b); // AVX512DQ
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i64 horizontal_add(const i64x8 a) {
#if defined(__INTEL_COMPILER)
  return _mm512_reduce_add_epi64(a);
#else
  return horizontal_add(a.get_low() + a.get_high());
#endif
}

// Horizontal add extended: Calculates the sum of all vector elements
// Elements are sign extended before adding to avoid overflow
inline i64 horizontal_add_x(const i32x16 x) {
  const i64x8 a = _mm512_cvtepi32_epi64(x.get_low());
  const i64x8 b = _mm512_cvtepi32_epi64(x.get_high());
  return horizontal_add(a + b);
}

// Horizontal add extended: Calculates the sum of all vector elements
// Elements are zero extended before adding to avoid overflow
inline u64 horizontal_add_x(const u32x16 x) {
  const i64x8 a = _mm512_cvtepu32_epi64(x.get_low());
  const i64x8 b = _mm512_cvtepu32_epi64(x.get_high());
  return u64(horizontal_add(a + b));
}

// function max: a > b ? a : b
inline i64x8 max(const i64x8 a, const i64x8 b) {
  return _mm512_max_epi64(a, b);
}

// function min: a < b ? a : b
inline i64x8 min(const i64x8 a, const i64x8 b) {
  return _mm512_min_epi64(a, b);
}

// function abs: a >= 0 ? a : -a
inline i64x8 abs(const i64x8 a) {
  return _mm512_abs_epi64(a);
}

// function abs_saturated: same as abs, saturate if overflow
inline i64x8 abs_saturated(const i64x8 a) {
  return _mm512_min_epu64(abs(a), i64x8(0x7FFFFFFFFFFFFFFF));
}

// function rotate_left all elements
// Use negative count to rotate right
inline i64x8 rotate_left(const i64x8 a, i32 b) {
  return _mm512_rolv_epi64(a, i64x8(b));
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I64X8_HPP
