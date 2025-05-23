#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F64X08_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F64X08_HPP

#include <immintrin.h>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/08.hpp"
#include "stado/vector/native/operations/f32x04.hpp"
#include "stado/vector/native/operations/f64x04.hpp"
#include "stado/vector/native/operations/i64x08.hpp"
#include "stado/vector/native/operations/u64x08.hpp"
#include "stado/vector/native/types/f32x16.hpp"
#include "stado/vector/native/types/f64x08.hpp"
#include "stado/vector/native/types/i32x08.hpp"
#include "stado/vector/native/types/i32x16.hpp"
#include "stado/vector/native/types/i64x08.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
// vector operator + : add element by element
inline f64x8 operator+(const f64x8 a, const f64x8 b) {
  return _mm512_add_pd(a, b);
}

// vector operator + : add vector and scalar
inline f64x8 operator+(const f64x8 a, f64 b) {
  return a + f64x8(b);
}
inline f64x8 operator+(f64 a, const f64x8 b) {
  return f64x8(a) + b;
}

// vector operator += : add
inline f64x8& operator+=(f64x8& a, const f64x8 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline f64x8 operator++(f64x8& a, int) {
  f64x8 a0 = a;
  a = a + 1.0;
  return a0;
}

// prefix operator ++
inline f64x8& operator++(f64x8& a) {
  a = a + 1.0;
  return a;
}

// vector operator - : subtract element by element
inline f64x8 operator-(const f64x8 a, const f64x8 b) {
  return _mm512_sub_pd(a, b);
}

// vector operator - : subtract vector and scalar
inline f64x8 operator-(const f64x8 a, f64 b) {
  return a - f64x8(b);
}
inline f64x8 operator-(f64 a, const f64x8 b) {
  return f64x8(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
inline f64x8 operator-(const f64x8 a) {
  return _mm512_castsi512_pd(i64x8(_mm512_castpd_si512(a)) ^ i64x8(i64(0x8000000000000000)));
}

// vector operator -= : subtract
inline f64x8& operator-=(f64x8& a, const f64x8 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline f64x8 operator--(f64x8& a, int) {
  f64x8 a0 = a;
  a = a - 1.0;
  return a0;
}

// prefix operator --
inline f64x8& operator--(f64x8& a) {
  a = a - 1.0;
  return a;
}

// vector operator * : multiply element by element
inline f64x8 operator*(const f64x8 a, const f64x8 b) {
  return _mm512_mul_pd(a, b);
}

// vector operator * : multiply vector and scalar
inline f64x8 operator*(const f64x8 a, f64 b) {
  return a * f64x8(b);
}
inline f64x8 operator*(f64 a, const f64x8 b) {
  return f64x8(a) * b;
}

// vector operator *= : multiply
inline f64x8& operator*=(f64x8& a, const f64x8 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer
inline f64x8 operator/(const f64x8 a, const f64x8 b) {
  return _mm512_div_pd(a, b);
}

// vector operator / : divide vector and scalar
inline f64x8 operator/(const f64x8 a, f64 b) {
  return a / f64x8(b);
}
inline f64x8 operator/(f64 a, const f64x8 b) {
  return f64x8(a) / b;
}

// vector operator /= : divide
inline f64x8& operator/=(f64x8& a, const f64x8 b) {
  a = a / b;
  return a;
}

// vector operator == : returns true for elements for which a == b
inline CompactMask<8> operator==(const f64x8 a, const f64x8 b) {
  return _mm512_cmp_pd_mask(a, b, 0);
}

// vector operator != : returns true for elements for which a != b
inline CompactMask<8> operator!=(const f64x8 a, const f64x8 b) {
  return _mm512_cmp_pd_mask(a, b, 4);
}

// vector operator < : returns true for elements for which a < b
inline CompactMask<8> operator<(const f64x8 a, const f64x8 b) {
  return _mm512_cmp_pd_mask(a, b, 1);
}

// vector operator <= : returns true for elements for which a <= b
inline CompactMask<8> operator<=(const f64x8 a, const f64x8 b) {
  return _mm512_cmp_pd_mask(a, b, 2);
}

// vector operator > : returns true for elements for which a > b
inline CompactMask<8> operator>(const f64x8 a, const f64x8 b) {
  return _mm512_cmp_pd_mask(a, b, 6);
}

// vector operator >= : returns true for elements for which a >= b
inline CompactMask<8> operator>=(const f64x8 a, const f64x8 b) {
  return _mm512_cmp_pd_mask(a, b, 5);
}

// Bitwise logical operators

// vector operator & : bitwise and
inline f64x8 operator&(const f64x8 a, const f64x8 b) {
  return _mm512_castsi512_pd(i64x8(_mm512_castpd_si512(a)) & i64x8(_mm512_castpd_si512(b)));
}

// vector operator &= : bitwise and
inline f64x8& operator&=(f64x8& a, const f64x8 b) {
  a = a & b;
  return a;
}

// vector operator & : bitwise and of f64x8 and CompactMask<8>
inline f64x8 operator&(const f64x8 a, const CompactMask<8> b) {
  return _mm512_maskz_mov_pd(__mmask8(b), a);
}

inline f64x8 operator&(const CompactMask<8> a, const f64x8 b) {
  return b & a;
}

// vector operator | : bitwise or
inline f64x8 operator|(const f64x8 a, const f64x8 b) {
  return _mm512_castsi512_pd(i64x8(_mm512_castpd_si512(a)) | i64x8(_mm512_castpd_si512(b)));
}

// vector operator |= : bitwise or
inline f64x8& operator|=(f64x8& a, const f64x8 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline f64x8 operator^(const f64x8 a, const f64x8 b) {
  return _mm512_castsi512_pd(i64x8(_mm512_castpd_si512(a)) ^ i64x8(_mm512_castpd_si512(b)));
}

// vector operator ^= : bitwise xor
inline f64x8& operator^=(f64x8& a, const f64x8 b) {
  a = a ^ b;
  return a;
}

// vector operator ! : logical not. Returns Boolean vector
inline CompactMask<8> operator!(const f64x8 a) {
  return a == f64x8(0.0);
}

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 2; i++) result[i] = s[i] ? a[i] : b[i];
inline f64x8 select(const CompactMask<8> s, const f64x8 a, const f64x8 b) {
  return _mm512_mask_mov_pd(b, __mmask8(s), a);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline f64x8 if_add(const CompactMask<8> f, const f64x8 a, const f64x8 b) {
  return _mm512_mask_add_pd(a, __mmask8(f), a, b);
}

// Conditional subtract
inline f64x8 if_sub(const CompactMask<8> f, const f64x8 a, const f64x8 b) {
  return _mm512_mask_sub_pd(a, __mmask8(f), a, b);
}

// Conditional multiply
inline f64x8 if_mul(const CompactMask<8> f, const f64x8 a, const f64x8 b) {
  return _mm512_mask_mul_pd(a, __mmask8(f), a, b);
}

// Conditional divide
inline f64x8 if_div(const CompactMask<8> f, const f64x8 a, const f64x8 b) {
  return _mm512_mask_div_pd(a, __mmask8(f), a, b);
}

// Sign functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0, -INF and -NAN
inline CompactMask<8> sign_bit(const f64x8 a) {
  const i64x8 t1 = _mm512_castpd_si512(a); // reinterpret as 64-bit integer
  return {t1 < 0};
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
inline f64x8 sign_combine(const f64x8 a, const f64x8 b) {
  // return a ^ (b & f64x8(-0.0));
  return _mm512_castsi512_pd(_mm512_ternarylogic_epi64(
    _mm512_castpd_si512(a), _mm512_castpd_si512(b), (i64x8(i64(0x8000000000000000))), 0x78));
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, denormal or zero,
// false for INF and NAN
inline CompactMask<8> is_finite(const f64x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  const __mmask8 f = _mm512_fpclass_pd_mask(a, 0x99);
  return __mmask8(_mm512_knot(f));
#else
  i64x8 t1 = _mm512_castpd_si512(a); // reinterpret as 64-bit integer
  i64x8 t2 = t1 << 1; // shift out sign bit
  i64x8 t3 = i64(0xFFE0000000000000); // exponent mask
  auto t4 = i64x8(t2 & t3) != t3; // exponent field is not all 1s
  return {t4};
#endif
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
inline CompactMask<8> is_inf(const f64x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_fpclass_pd_mask(a, 0x18);
#else
  i64x8 t1 = _mm512_castpd_si512(a); // reinterpret as 64-bit integer
  i64x8 t2 = t1 << 1; // shift out sign bit
  // exponent is all 1s, fraction is 0
  return {t2 == i64x8(i64(0xFFE0000000000000))};
#endif
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
inline CompactMask<8> is_nan(const f64x8 a) {
  // assume that compiler does not optimize this away with -ffinite-math-only:
  return _mm512_fpclass_pd_mask(a, 0x81);
}
// #elif defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
//__attribute__((optimize("-fno-unsafe-math-optimizations")))
//  inline CompactMask<8> is_nan(f64x8 const a) {
//     return a != a; // not safe with -ffinite-math-only compiler option
// }
#elif (defined(__GNUC__) || defined(__clang__)) && !defined(__INTEL_COMPILER)
inline CompactMask<8> is_nan(const f64x8 a) {
  __m512d aa = a;
  __mmask16 unordered;
  __asm volatile("vcmppd $3, %1, %1, %0" : "=Yk"(unordered) : "v"(aa));
  return {unordered};
}
#else
inline CompactMask<8> is_nan(const f64x8 a) {
  // assume that compiler does not optimize this away with -ffinite-math-only:
  return CompactMask<8>().load_bits(_mm512_cmp_pd_mask(a, a, 3)); // compare unordered
  // return a != a; // This is not safe with -ffinite-math-only, -ffast-math, or /fp:fast compiler
  // option
}
#endif

// Function is_subnormal: gives true for elements that are denormal (subnormal)
// false for finite numbers, zero, NAN and INF
inline CompactMask<8> is_subnormal(const f64x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_fpclass_pd_mask(a, 0x20);
#else
  i64x8 t1 = _mm512_castpd_si512(a); // reinterpret as 64-bit integer
  i64x8 t2 = t1 << 1; // shift out sign bit
  i64x8 t3 = i64(0xFFE0000000000000); // exponent mask
  i64x8 t4 = t2 & t3; // exponent
  i64x8 t5 = _mm512_andnot_si512(t3, t2); // fraction
  return {t4 == i64x8(0) && t5 != i64x8(0)}; // exponent = 0 and fraction != 0
#endif
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
inline CompactMask<8> is_zero_or_subnormal(const f64x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_fpclass_pd_mask(a, 0x26);
#else
  i64x8 t = _mm512_castpd_si512(a); // reinterpret as 32-bit integer
  t &= i64(0x7FF0000000000000); // isolate exponent
  return {t == i64x8(0)}; // exponent = 0
#endif
}

// change signs on vectors f64x8
// Each index i0 - i3 is 1 for changing sign on the corresponding element, 0 for no change
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline f64x8 change_sign(const f64x8 a) {
  const auto m = __mmask8((i0 & 1) | (i1 & 1) << 1 | (i2 & 1) << 2 | (i3 & 1) << 3 | (i4 & 1) << 4 |
                          (i5 & 1) << 5 | (i6 & 1) << 6 | (i7 & 1) << 7);
  if (u8(m) == 0) {
    return a;
  }
#ifdef __x86_64__
  __m512d s = _mm512_castsi512_pd(_mm512_maskz_set1_epi64(m, i64(0x8000000000000000)));
#else // 32 bit mode
  __m512i v = i64x8(0x8000000000000000);
  __m512d s = _mm512_castsi512_pd(_mm512_maskz_mov_epi64(m, v));
#endif
  return a ^ s;
}

// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
inline f64 horizontal_add(const f64x8 a) {
#if defined(__INTEL_COMPILER)
  return _mm512_reduce_add_pd(a);
#else
  return horizontal_add(a.get_low() + a.get_high());
#endif
}

// function max: a > b ? a : b
inline f64x8 max(const f64x8 a, const f64x8 b) {
  return _mm512_max_pd(a, b);
}

// function min: a < b ? a : b
inline f64x8 min(const f64x8 a, const f64x8 b) {
  return _mm512_min_pd(a, b);
}
// NAN-safe versions of maximum and minimum are in vector-convert.hpp

// function abs: absolute value
inline f64x8 abs(const f64x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_range_pd(a, a, 8);
#else
  return a & f64x8(_mm512_castsi512_pd(i64x8(0x7FFFFFFFFFFFFFFF)));
#endif
}

// function sqrt: square root
inline f64x8 sqrt(const f64x8 a) {
  return _mm512_sqrt_pd(a);
}

// function square: a * a
inline f64x8 square(const f64x8 a) {
  return a * a;
}

// The purpose of this template is to prevent implicit conversion of a float
// exponent to int when calling pow(vector, float) and vectormath_exp.h is not included
template<typename TT>
static f64x8 pow(f64x8 x0, TT n); // = delete;

// pow(f64x8, int):
// Raise floating point numbers to integer power n
template<>
inline f64x8 pow<i32>(const f64x8 x0, const i32 n) {
  return pow_template_i<f64x8>(x0, n);
}

// allow conversion from unsigned int
template<>
inline f64x8 pow<u32>(const f64x8 x0, const u32 n) {
  return pow_template_i<f64x8>(x0, i32(n));
}

// Raise floating point numbers to integer power n, where n is a compile-time constant
template<i32 n>
inline f64x8 pow(const f64x8 a, ConstInt<n> /*n*/) {
  return pow_n<f64x8, n>(a);
}

// function round: round to nearest integer (even). (result as f64 vector)
inline f64x8 round(const f64x8 a) {
  return _mm512_roundscale_pd(a, 0);
}

// function truncate: round towards zero. (result as f64 vector)
inline f64x8 truncate(const f64x8 a) {
  return _mm512_roundscale_pd(a, 3);
}

// function floor: round towards minus infinity. (result as f64 vector)
inline f64x8 floor(const f64x8 a) {
  return _mm512_roundscale_pd(a, 1);
}

// function ceil: round towards plus infinity. (result as f64 vector)
inline f64x8 ceil(const f64x8 a) {
  return _mm512_roundscale_pd(a, 2);
}

// function round_to_int32: round to nearest integer (even). (result as integer vector)
inline i32x8 round_to_int32(const f64x8 a) {
  // return _mm512_cvtpd_epi32(a);
  return _mm512_cvt_roundpd_epi32(a, 0 + 8);
}

// function truncate_to_int32: round towards zero. (result as integer vector)
inline i32x8 truncate_to_int32(const f64x8 a) {
  return _mm512_cvttpd_epi32(a);
}

// function truncatei: round towards zero
inline i64x8 truncatei(const f64x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_cvttpd_epi64(a);
#else
  f64 aa[8]; // inefficient
  a.store(aa);
  return {i64(aa[0]), i64(aa[1]), i64(aa[2]), i64(aa[3]),
          i64(aa[4]), i64(aa[5]), i64(aa[6]), i64(aa[7])};
#endif
}

// function roundi: round to nearest or even
inline i64x8 roundi(const f64x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_cvtpd_epi64(a);
#else
  return truncatei(round(a));
#endif
}

// function to_f64: convert integer vector elements to f64 vector
inline f64x8 to_f64(const i64x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_cvtepi64_pd(a);
#else
  i64 aa[8]; // inefficient
  a.store(aa);
  return {f64(aa[0]), f64(aa[1]), f64(aa[2]), f64(aa[3]),
          f64(aa[4]), f64(aa[5]), f64(aa[6]), f64(aa[7])};
#endif
}

inline f64x8 to_f64(const u64x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_cvtepu64_pd(a);
#else
  u64 aa[8]; // inefficient
  a.store(aa);
  return {f64(aa[0]), f64(aa[1]), f64(aa[2]), f64(aa[3]),
          f64(aa[4]), f64(aa[5]), f64(aa[6]), f64(aa[7])};
#endif
}

// function to_f64: convert integer vector to f64 vector
inline f64x8 to_f64(const i32x8 a) {
  return _mm512_cvtepi32_pd(a);
}

// function compress: convert two f64x8 to one f32x16
inline f32x16 compress(const f64x8 low, const f64x8 high) {
  const __m256 t1 = _mm512_cvtpd_ps(low);
  const __m256 t2 = _mm512_cvtpd_ps(high);
  return {t1, t2};
}

// Function extend_low : convert f32x16 vector elements 0 - 3 to
// f64x8
inline f64x8 extend_low(const f32x16 a) {
  return _mm512_cvtps_pd(_mm512_castps512_ps256(a));
}

// Function extend_high : convert f32x16 vector elements 4 - 7 to
// f64x8
inline f64x8 extend_high(const f32x16 a) {
  return _mm512_cvtps_pd(a.get_high());
}

// Fused multiply and add functions

// Multiply and add
inline f64x8 mul_add(const f64x8 a, const f64x8 b, const f64x8 c) {
  return _mm512_fmadd_pd(a, b, c);
}

// Multiply and subtract
inline f64x8 mul_sub(const f64x8 a, const f64x8 b, const f64x8 c) {
  return _mm512_fmsub_pd(a, b, c);
}

// Multiply and inverse subtract
inline f64x8 nmul_add(const f64x8 a, const f64x8 b, const f64x8 c) {
  return _mm512_fnmadd_pd(a, b, c);
}

// Multiply and subtract with extra precision on the intermediate calculations. used internally in
// math functions
inline f64x8 mul_sub_x(const f64x8 a, const f64x8 b, const f64x8 c) {
  return _mm512_fmsub_pd(a, b, c);
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0) = 0, exponent(0.0) = -1023, exponent(INF) = +1024, exponent(NAN) = +1024
inline i64x8 exponent(const f64x8 a) {
  const u64x8 t1 = _mm512_castpd_si512(a); // reinterpret as 64-bit integer
  const u64x8 t2 = t1 << 1; // shift out sign bit
  const u64x8 t3 = t2 >> 53; // shift down logical to position 0
  i64x8 t4 = i64x8(t3) - 0x3FF; // subtract bias from exponent
  return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0) = 1.0, fraction(5.0) = 1.25
inline f64x8 fraction(const f64x8 a) {
  return _mm512_getmant_pd(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
}

// Fast calculation of pow(2,n) with n integer
// n  =     0 gives 1.0
// n >=  1024 gives +INF
// n <= -1023 gives 0.0
// This function will never produce denormals, and never raise exceptions
inline f64x8 exp2(const i64x8 n) {
  const i64x8 t1 = max(n, -0x3FF); // limit to allowed range
  const i64x8 t2 = min(t1, 0x400);
  const i64x8 t3 = t2 + 0x3FF; // add bias
  const i64x8 t4 = t3 << 52; // put exponent into position 52
  return _mm512_castsi512_pd(t4); // reinterpret as f64
}
// static f64x8 exp2(f64x8 const x);    // defined in
// vectormath_exp.h

/*****************************************************************************
 *
 *          Functions for reinterpretation between vector types
 *
 *****************************************************************************/

// AVX512 requires gcc version 4.9 or higher. Apparently the problem with mangling intrinsic vector
// types no longer exists in gcc 4.x

inline __m512i reinterpret_i(const __m512i x) {
  return x;
}

inline __m512i reinterpret_i(const __m512 x) {
  return _mm512_castps_si512(x);
}

inline __m512i reinterpret_i(const __m512d x) {
  return _mm512_castpd_si512(x);
}

inline __m512 reinterpret_f(const __m512i x) {
  return _mm512_castsi512_ps(x);
}

inline __m512 reinterpret_f(const __m512 x) {
  return x;
}

inline __m512 reinterpret_f(const __m512d x) {
  return _mm512_castpd_ps(x);
}

inline __m512d reinterpret_d(const __m512i x) {
  return _mm512_castsi512_pd(x);
}

inline __m512d reinterpret_d(const __m512 x) {
  return _mm512_castps_pd(x);
}

inline __m512d reinterpret_d(const __m512d x) {
  return x;
}

// Function infinite4f: returns a vector where all elements are +INF
inline f32x16 infinite16f() {
  return reinterpret_f(i32x16(0x7F800000));
}

// Function nan4f: returns a vector where all elements are +NAN (quiet)
inline f32x16 nan16f(u32 n = 0x100) {
  return nan_vec<f32x16>(n);
}

// Function infinite2d: returns a vector where all elements are +INF
inline f64x8 infinite8d() {
  return reinterpret_d(i64x8(0x7FF0000000000000));
}

// Function nan8d: returns a vector where all elements are +NAN (quiet NAN)
inline f64x8 nan8d(u32 n = 0x10) {
  return nan_vec<f64x8>(n);
}

inline f64x8 shuffle_up(f64x8 vec) {
  return _mm512_maskz_expand_pd(__mmask8{0xFE}, vec);
}
inline f64x8 shuffle_up(f64x8 vec, f64 first) {
  u64x8 idx_vec{8, 0, 1, 2, 3, 4, 5, 6};
  return _mm512_mask_permutex2var_pd(vec, __mmask8{0xFF}, idx_vec, f64x8::expand_undef(first));
}

inline f64x8 shuffle_down(f64x8 vec) {
  u64x8 idx_vec{1, 2, 3, 4, 5, 6, 7, 15};
  return _mm512_mask_permutex2var_pd(vec, __mmask8{0xFF}, idx_vec, _mm512_setzero_pd());
}
inline f64x8 shuffle_down(f64x8 vec, f64 last) {
  u64x8 idx_vec{1, 2, 3, 4, 5, 6, 7, 15};
  return _mm512_mask_permutex2var_pd(vec, __mmask8{0xFF}, idx_vec, _mm512_set1_pd(last));
}

template<bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7>
inline f64x8 blend(f64x8 a, f64x8 b) {
  return _mm512_mask_blend_pd(__mmask8{unsigned{b0} | (unsigned{b1} << 1U) | (unsigned{b2} << 2U) |
                                       (unsigned{b3} << 3U) | (unsigned{b4} << 4U) |
                                       (unsigned{b5} << 5U) | (unsigned{b6} << 6U) |
                                       (unsigned{b7} << 7U)},
                              b, a);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F64X08_HPP
