#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F64X4_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F64X4_HPP

#include <limits>

#include <immintrin.h>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/operations/f32x4.hpp"
#include "stado/vector/native/operations/f64x2.hpp"
#include "stado/vector/native/operations/i64x4.hpp"
#include "stado/vector/native/operations/u64x4.hpp"
#include "stado/vector/native/types/f32x8.hpp"
#include "stado/vector/native/types/f64x4.hpp"
#include "stado/vector/native/types/i32x4.hpp"
#include "stado/vector/native/types/i64x4.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX
namespace stado {
// vector operator + : add element by element
inline f64x4 operator+(const f64x4 a, const f64x4 b) {
  return _mm256_add_pd(a, b);
}

// vector operator + : add vector and scalar
inline f64x4 operator+(const f64x4 a, f64 b) {
  return a + f64x4(b);
}
inline f64x4 operator+(f64 a, const f64x4 b) {
  return f64x4(a) + b;
}

// vector operator += : add
inline f64x4& operator+=(f64x4& a, const f64x4 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline f64x4 operator++(f64x4& a, int) {
  f64x4 a0 = a;
  a = a + 1.0;
  return a0;
}

// prefix operator ++
inline f64x4& operator++(f64x4& a) {
  a = a + 1.0;
  return a;
}

// vector operator - : subtract element by element
inline f64x4 operator-(const f64x4 a, const f64x4 b) {
  return _mm256_sub_pd(a, b);
}

// vector operator - : subtract vector and scalar
inline f64x4 operator-(const f64x4 a, f64 b) {
  return a - f64x4(b);
}
inline f64x4 operator-(f64 a, const f64x4 b) {
  return f64x4(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
inline f64x4 operator-(const f64x4 a) {
  const __m256d mask = _mm256_castsi256_pd(_mm256_setr_epi32(
    0, i32(0x80000000U), 0, i32(0x80000000U), 0, i32(0x80000000U), 0, i32(0x80000000U)));
  return _mm256_xor_pd(a, mask);
}

// vector operator -= : subtract
inline f64x4& operator-=(f64x4& a, const f64x4 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline f64x4 operator--(f64x4& a, int) {
  f64x4 a0 = a;
  a = a - 1.0;
  return a0;
}

// prefix operator --
inline f64x4& operator--(f64x4& a) {
  a = a - 1.0;
  return a;
}

// vector operator * : multiply element by element
inline f64x4 operator*(const f64x4 a, const f64x4 b) {
  return _mm256_mul_pd(a, b);
}

// vector operator * : multiply vector and scalar
inline f64x4 operator*(const f64x4 a, f64 b) {
  return a * f64x4(b);
}
inline f64x4 operator*(f64 a, const f64x4 b) {
  return f64x4(a) * b;
}

// vector operator *= : multiply
inline f64x4& operator*=(f64x4& a, const f64x4 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer
inline f64x4 operator/(const f64x4 a, const f64x4 b) {
  return _mm256_div_pd(a, b);
}

// vector operator / : divide vector and scalar
inline f64x4 operator/(const f64x4 a, f64 b) {
  return a / f64x4(b);
}
inline f64x4 operator/(f64 a, const f64x4 b) {
  return f64x4(a) / b;
}

// vector operator /= : divide
inline f64x4& operator/=(f64x4& a, const f64x4 b) {
  a = a / b;
  return a;
}

// vector operator == : returns true for elements for which a == b
inline Mask<64, 4> operator==(const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_pd_mask(a, b, 0);
#else
  return _mm256_cmp_pd(a, b, 0);
#endif
}

// vector operator != : returns true for elements for which a != b
inline Mask<64, 4> operator!=(const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_pd_mask(a, b, 4);
#else
  return _mm256_cmp_pd(a, b, 4);
#endif
}

// vector operator < : returns true for elements for which a < b
inline Mask<64, 4> operator<(const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_pd_mask(a, b, 1);
#else
  return _mm256_cmp_pd(a, b, 1);
#endif
}

// vector operator <= : returns true for elements for which a <= b
inline Mask<64, 4> operator<=(const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_pd_mask(a, b, 2);
#else
  return _mm256_cmp_pd(a, b, 2);
#endif
}

// vector operator > : returns true for elements for which a > b
inline Mask<64, 4> operator>(const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_pd_mask(a, b, 6);
#else
  return b < a;
#endif
}

// vector operator >= : returns true for elements for which a >= b
inline Mask<64, 4> operator>=(const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_pd_mask(a, b, 5);
#else
  return b <= a;
#endif
}

// Bitwise logical operators

// vector operator & : bitwise and
inline f64x4 operator&(const f64x4 a, const f64x4 b) {
  return _mm256_and_pd(a, b);
}

// vector operator &= : bitwise and
inline f64x4& operator&=(f64x4& a, const f64x4 b) {
  a = a & b;
  return a;
}

// vector operator & : bitwise and of f64x4 and Mask<64, 4>
inline f64x4 operator&(const f64x4 a, const Mask<64, 4> b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_maskz_mov_pd(b, a);
#else
  return _mm256_and_pd(a, b);
#endif
}
inline f64x4 operator&(const Mask<64, 4> a, const f64x4 b) {
  return b & a;
}

// vector operator | : bitwise or
inline f64x4 operator|(const f64x4 a, const f64x4 b) {
  return _mm256_or_pd(a, b);
}

// vector operator |= : bitwise or
inline f64x4& operator|=(f64x4& a, const f64x4 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline f64x4 operator^(const f64x4 a, const f64x4 b) {
  return _mm256_xor_pd(a, b);
}

// vector operator ^= : bitwise xor
inline f64x4& operator^=(f64x4& a, const f64x4 b) {
  a = a ^ b;
  return a;
}

// vector operator ! : logical not. Returns Boolean vector
inline Mask<64, 4> operator!(const f64x4 a) {
  return a == f64x4(0.0);
}

/*****************************************************************************
 *
 *          Functions for f64x4
 *
 *****************************************************************************/

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 2; i++) result[i] = s[i] ? a[i] : b[i];
inline f64x4 select(const Mask<64, 4> s, const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mov_pd(b, s, a);
#else
  return _mm256_blendv_pd(b, a, s);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline f64x4 if_add(const Mask<64, 4> f, const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_mask_add_pd(a, f, a, b);
#else
  return a + (f64x4(f) & b);
#endif
}

// Conditional subtract
inline f64x4 if_sub(const Mask<64, 4> f, const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_mask_sub_pd(a, f, a, b);
#else
  return a - (f64x4(f) & b);
#endif
}

// Conditional multiply
inline f64x4 if_mul(const Mask<64, 4> f, const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_mask_mul_pd(a, f, a, b);
#else
  return a * select(f, b, 1.);
#endif
}

// Conditional divide
inline f64x4 if_div(const Mask<64, 4> f, const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_mask_div_pd(a, f, a, b);
#else
  return a / select(f, b, 1.);
#endif
}

// sign functions

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
inline f64x4 sign_combine(const f64x4 a, const f64x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_castsi256_pd(_mm256_ternarylogic_epi64(
    _mm256_castpd_si256(a), _mm256_castpd_si256(b), i64x4(i64(0x8000000000000000)), 0x78));
#else
  return a ^ (b & f64x4(-0.0));
#endif
}

// Function is_finite: gives true for elements that are normal, subnormal or zero,
// false for INF and NAN
inline Mask<64, 4> is_finite(const f64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return __mmask8(~_mm256_fpclass_pd_mask(a, 0x99));
#elif STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  i64x4 t1 = _mm256_castpd_si256(a); // reinterpret as 64-bit integer
  i64x4 t2 = t1 << 1; // shift out sign bit
  i64x4 t3 = i64(0xFFE0000000000000); // exponent mask
  Mask<64, 4> t4 = i64x4(t2 & t3) != t3; // exponent field is not all 1s
  return t4;
#else
  return Mask<64, 4>(is_finite(a.get_low()), is_finite(a.get_high()));
#endif
}

// categorization functions

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
inline Mask<64, 4> is_inf(const f64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_fpclass_pd_mask(a, 0x18);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  i64x4 t1 = _mm256_castpd_si256(a); // reinterpret as 64-bit integer
  i64x4 t2 = t1 << 1; // shift out sign bit
  return t2 == i64x4(i64(0xFFE0000000000000)); // exponent is all 1s, fraction is 0
#else
  return Mask<64, 4>(is_inf(a.get_low()), is_inf(a.get_high()));
#endif
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
inline Mask<64, 4> is_nan(const f64x4 a) {
  // assume that compiler does not optimize this away with -ffinite-math-only:
  return _mm256_fpclass_pd_mask(a, 0x81);
}
// #elif defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
//__attribute__((optimize("-fno-unsafe-math-optimizations")))
//  inline Mask<64, 4> is_nan(f64x4 const a) {
//     return a != a; // not safe with -ffinite-math-only compiler option
// }
#elif (defined(__GNUC__) || defined(__clang__)) && !defined(__INTEL_COMPILER)
inline Mask<64, 4> is_nan(const f64x4 a) {
  __m256d aa = a;
  __m256d unordered;
  __asm volatile("vcmppd $3, %1, %1, %0" : "=v"(unordered) : "v"(aa));
  return Mask<64, 4>(unordered);
}
#else
inline Mask<64, 4> is_nan(const f64x4 a) {
  // assume that compiler does not optimize this away with -ffinite-math-only:
  return _mm256_cmp_pd(a, a, 3); // compare unordered
  // return a != a; // This is not safe with -ffinite-math-only, -ffast-math, or /fp:fast compiler
  // option
}
#endif

// Function is_subnormal: gives true for elements that are subnormal (denormal)
// false for finite numbers, zero, NAN and INF
inline Mask<64, 4> is_subnormal(const f64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_fpclass_pd_mask(a, 0x20);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  i64x4 t1 = _mm256_castpd_si256(a); // reinterpret as 64-bit integer
  i64x4 t2 = t1 << 1; // shift out sign bit
  i64x4 t3 = i64(0xFFE0000000000000); // exponent mask
  i64x4 t4 = t2 & t3; // exponent
  i64x4 t5 = _mm256_andnot_si256(t3, t2); // fraction
  return Mask<64, 4>(t4 == i64x4(0) && t5 != i64x4(0)); // exponent = 0 and fraction != 0
#else
  return Mask<64, 4>(is_subnormal(a.get_low()), is_subnormal(a.get_high()));
#endif
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
inline Mask<64, 4> is_zero_or_subnormal(const f64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_fpclass_pd_mask(a, 0x26);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  i64x4 t = _mm256_castpd_si256(a); // reinterpret as 32-bit integer
  t &= i64(0x7FF0000000000000LL); // isolate exponent
  return t == i64x4(0); // exponent = 0
#else
  return Mask<64, 4>(is_zero_or_subnormal(a.get_low()), is_zero_or_subnormal(a.get_high()));
#endif
}

// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
inline f64 horizontal_add(const f64x4 a) {
  return horizontal_add(a.get_low() + a.get_high());
}

// function max: a > b ? a : b
inline f64x4 max(const f64x4 a, const f64x4 b) {
  return _mm256_max_pd(a, b);
}

// function min: a < b ? a : b
inline f64x4 min(const f64x4 a, const f64x4 b) {
  return _mm256_min_pd(a, b);
}
// NAN-safe versions of maximum and minimum are in vector-convert.hpp

// function abs: absolute value
inline f64x4 abs(const f64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm256_range_pd(a, a, 8);
#else
  __m256d mask = _mm256_castsi256_pd(
    _mm256_setr_epi32(i32(0xFFFFFFFF), i32(0x7FFFFFFF), i32(0xFFFFFFFF), i32(0x7FFFFFFF),
                      i32(0xFFFFFFFF), i32(0x7FFFFFFF), i32(0xFFFFFFFF), i32(0x7FFFFFFF)));
  return _mm256_and_pd(a, mask);
#endif
}

// function sqrt: square root
inline f64x4 sqrt(const f64x4 a) {
  return _mm256_sqrt_pd(a);
}

// function square: a * a
inline f64x4 square(const f64x4 a) {
  return a * a;
}

// The purpose of this template is to prevent implicit conversion of a f32
// exponent to int when calling pow(vector, f32) and vectormath_exp.h is not included
template<typename TT>
static f64x4 pow(f64x4 x0, TT n);

// Raise floating point numbers to integer power n
template<>
inline f64x4 pow<i32>(const f64x4 x0, const i32 n) {
  return pow_template_i<f64x4>(x0, n);
}

// allow conversion from unsigned int
template<>
inline f64x4 pow<u32>(const f64x4 x0, const u32 n) {
  return pow_template_i<f64x4>(x0, i32(n));
}

// Raise floating point numbers to integer power n, where n is a compile-time constant
template<i32 n>
inline f64x4 pow(const f64x4 a, ConstInt<n> /*n*/) {
  return pow_n<f64x4, n>(a);
}

// function round: round to nearest integer (even). (result as f64 vector)
inline f64x4 round(const f64x4 a) {
  return _mm256_round_pd(a, 0 + 8);
}

// function truncate: round towards zero. (result as f64 vector)
inline f64x4 truncate(const f64x4 a) {
  return _mm256_round_pd(a, 3 + 8);
}

// function floor: round towards minus infinity. (result as f64 vector)
inline f64x4 floor(const f64x4 a) {
  return _mm256_round_pd(a, 1 + 8);
}

// function ceil: round towards plus infinity. (result as f64 vector)
inline f64x4 ceil(const f64x4 a) {
  return _mm256_round_pd(a, 2 + 8);
}

// function round_to_int32: round to nearest integer (even). (result as integer vector)
inline i32x4 round_to_int32(const f64x4 a) {
  // Note: assume MXCSR control register is set to rounding
  return _mm256_cvtpd_epi32(a);
}

// function truncate_to_int32: round towards zero. (result as integer vector)
inline i32x4 truncate_to_int32(const f64x4 a) {
  return _mm256_cvttpd_epi32(a);
}

#if STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
// function truncatei: round towards zero
inline i64x4 truncatei(const f64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ AVX512VL
  return _mm256_cvttpd_epi64(a);
#else
  f64 aa[4]; // inefficient
  a.store(aa);
  return {i64(aa[0]), i64(aa[1]), i64(aa[2]), i64(aa[3])};
#endif
}

// function roundi: round to nearest or even
inline i64x4 roundi(const f64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ AVX512VL
  return _mm256_cvtpd_epi64(a);
#else
  return truncatei(round(a)); // inefficient
#endif
}

// function to_f64: convert integer vector elements to f64 vector
inline f64x4 to_f64(const i64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ AVX512VL
  return _mm256_maskz_cvtepi64_pd(__mmask16(0xFF), a);
#else
  i64 aa[4]; // inefficient
  a.store(aa);
  return f64x4(f64(aa[0]), f64(aa[1]), f64(aa[2]), f64(aa[3]));
#endif
}

inline f64x4 to_f64(const u64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ AVX512VL
  return _mm256_cvtepu64_pd(a);
#else
  u64 aa[4]; // inefficient
  a.store(aa);
  return {f64(aa[0]), f64(aa[1]), f64(aa[2]), f64(aa[3])};
#endif
}
#endif

// function to_f64: convert integer vector to f64 vector
inline f64x4 to_f64(const i32x4 a) {
  return _mm256_cvtepi32_pd(a);
}

// function compress: convert two f64x4 to one f32x8
inline f32x8 compress(const f64x4 low, const f64x4 high) {
  const __m128 t1 = _mm256_cvtpd_ps(low);
  const __m128 t2 = _mm256_cvtpd_ps(high);
  return f32x8(t1, t2);
}

// Function extend_low : convert f32x8 vector elements 0 - 3 to
// f64x4
inline f64x4 extend_low(const f32x8 a) {
  return _mm256_cvtps_pd(_mm256_castps256_ps128(a));
}

// Function extend_high : convert f32x8 vector elements 4 - 7 to
// f64x4
inline f64x4 extend_high(const f32x8 a) {
  return _mm256_cvtps_pd(_mm256_extractf128_ps(a, 1));
}

// Fused multiply and add functions

// Multiply and add
inline f64x4 mul_add(const f64x4 a, const f64x4 b, const f64x4 c) {
#ifdef __FMA__
  return _mm256_fmadd_pd(a, b, c);
#elif defined(__FMA4__)
  return _mm256_macc_pd(a, b, c);
#else
  return a * b + c;
#endif
}

// Multiply and subtract
inline f64x4 mul_sub(const f64x4 a, const f64x4 b, const f64x4 c) {
#ifdef __FMA__
  return _mm256_fmsub_pd(a, b, c);
#elif defined(__FMA4__)
  return _mm256_msub_pd(a, b, c);
#else
  return a * b - c;
#endif
}

// Multiply and inverse subtract
inline f64x4 nmul_add(const f64x4 a, const f64x4 b, const f64x4 c) {
#ifdef __FMA__
  return _mm256_fnmadd_pd(a, b, c);
#elif defined(__FMA4__)
  return _mm256_nmacc_pd(a, b, c);
#else
  return c - a * b;
#endif
}

// Multiply and subtract with extra precision on the intermediate calculations,
// even if FMA instructions not supported, using Veltkamp-Dekker split.
// This is used in mathematical functions. Do not use it in general code
// because it is inaccurate in certain cases
inline f64x4 mul_sub_x(const f64x4 a, const f64x4 b, const f64x4 c) {
#ifdef __FMA__
  return _mm256_fmsub_pd(a, b, c);
#elif defined(__FMA4__)
  return _mm256_msub_pd(a, b, c);
#else
  // calculate a * b - c with extra precision
  // mask to remove lower 27 bits
  f64x4 upper_mask = _mm256_castsi256_pd(
    _mm256_setr_epi32(i32(0xF8000000), i32(0xFFFFFFFF), i32(0xF8000000), i32(0xFFFFFFFF),
                      i32(0xF8000000), i32(0xFFFFFFFF), i32(0xF8000000), i32(0xFFFFFFFF)));
  f64x4 a_high = a & upper_mask; // split into high and low parts
  f64x4 b_high = b & upper_mask;
  f64x4 a_low = a - a_high;
  f64x4 b_low = b - b_high;
  f64x4 r1 = a_high * b_high; // this product is exact
  f64x4 r2 = r1 - c; // subtract c from high product
  f64x4 r3 = r2 + (a_high * b_low + b_high * a_low) + a_low * b_low; // add rest of product
  return r3; // + ((r2 - r1) + c);
#endif
}

// Math functions using fast bit manipulation

#if STADO_INSTRUCTION_SET >= STADO_AVX2
// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0) = 0, exponent(0.0) = -1023, exponent(INF) = +1024, exponent(NAN) = +1024
inline i64x4 exponent(const f64x4 a) {
  const u64x4 t1 = _mm256_castpd_si256(a); // reinterpret as 64-bit integer
  const u64x4 t2 = t1 << 1; // shift out sign bit
  const u64x4 t3 = t2 >> 53; // shift down logical to position 0
  i64x4 t4 = i64x4(t3) - 0x3FF; // subtract bias from exponent
  return t4;
}
#endif

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0) = 1.0, fraction(5.0) = 1.25
inline f64x4 fraction(const f64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_getmant_pd(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  u64x4 t1 = _mm256_castpd_si256(a); // reinterpret as 64-bit integer
  u64x4 t2 = (t1 & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000; // set exponent to 0 + bias
  return _mm256_castsi256_pd(t2);
#else
  return f64x4(fraction(a.get_low()), fraction(a.get_high()));
#endif
}

// Fast calculation of pow(2,n) with n integer
// n  =     0 gives 1.0
// n >=  1024 gives +INF
// n <= -1023 gives 0.0
// This function will never produce subnormals, and never raise exceptions
inline f64x4 exp2(const i64x4 n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  const i64x4 t1 = max(n, -0x3FF); // limit to allowed range
  const i64x4 t2 = min(t1, 0x400);
  const i64x4 t3 = t2 + 0x3FF; // add bias
  const i64x4 t4 = t3 << 52; // put exponent into position 52
  return _mm256_castsi256_pd(t4); // reinterpret as f64
#else
  return f64x4(exp2(n.get_low()), exp2(n.get_high()));
#endif
}
// inline f64x4 exp2(f64x4 const x); // defined in
// vectormath_exp.h

// Categorization functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0, -INF and -NAN
// Note that sign_bit(f64x4(-0.0)) gives true, while f64x4(-0.0)
// < f64x4(0.0) gives false
inline Mask<64, 4> sign_bit(const f64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  const i64x4 t1 = _mm256_castpd_si256(a); // reinterpret as 64-bit integer
  const i64x4 t2 = t1 >> 63; // extend sign bit
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return t2 != i64x4(0);
#else
  return _mm256_castsi256_pd(t2); // reinterpret as 64-bit Boolean
#endif
#else
  return Mask<64, 4>(sign_bit(a.get_low()), sign_bit(a.get_high()));
#endif
}

// change signs on vectors f64x4
// Each index i0 - i3 is 1 for changing sign on the corresponding element, 0 for no change
template<int i0, int i1, int i2, int i3>
inline f64x4 change_sign(const f64x4 a) {
  if ((i0 | i1 | i2 | i3) == 0) {
    return a;
  }
  __m256d mask =
    _mm256_castsi256_pd(_mm256_setr_epi32(0, (i0 ? 0x80000000 : 0), 0, (i1 ? 0x80000000 : 0), 0,
                                          (i2 ? 0x80000000U : 0), 0, (i3 ? 0x80000000 : 0)));
  return _mm256_xor_pd(a, mask);
}

#if STADO_INSTRUCTION_SET >= STADO_AVX2
inline __m256i reinterpret_i(const __m256i x) {
  return x;
}

inline __m256i reinterpret_i(const __m256 x) {
  return _mm256_castps_si256(x);
}

inline __m256i reinterpret_i(const __m256d x) {
  return _mm256_castpd_si256(x);
}

inline __m256 reinterpret_f(const __m256i x) {
  return _mm256_castsi256_ps(x);
}

inline __m256 reinterpret_f(const __m256 x) {
  return x;
}

inline __m256 reinterpret_f(const __m256d x) {
  return _mm256_castpd_ps(x);
}

inline __m256d reinterpret_d(const __m256i x) {
  return _mm256_castsi256_pd(x);
}

inline __m256d reinterpret_d(const __m256 x) {
  return _mm256_castps_pd(x);
}

inline __m256d reinterpret_d(const __m256d x) {
  return x;
}
#else
inline __m256 reinterpret_f(const __m256 x) {
  return x;
}

inline __m256 reinterpret_f(const __m256d x) {
  return _mm256_castpd_ps(x);
}

inline __m256d reinterpret_d(const __m256 x) {
  return _mm256_castps_pd(x);
}

inline __m256d reinterpret_d(const __m256d x) {
  return x;
}
#endif

// Function infinite4f: returns a vector where all elements are +INF
inline f32x8 infinite8f() {
  return {std::numeric_limits<f32>::infinity()};
}

// Function nan8f: returns a vector where all elements are +NAN (quiet)
inline f32x8 nan8f(i32 n = 0x10) {
  return nan_vec<f32x8>(n);
}

// Function infinite2d: returns a vector where all elements are +INF
inline f64x4 infinite4d() {
  return {std::numeric_limits<f64>::infinity()};
}

// Function nan4d: returns a vector where all elements are +NAN (quiet)
inline f64x4 nan4d(i32 n = 0x10) {
  return nan_vec<f64x4>(n);
}

inline f64x4 shuffle_up(f64x4 vec) {
  auto x = _mm256_insertf128_pd(_mm256_setzero_pd(), _mm256_castpd256_pd128(vec), 1);
  return _mm256_shuffle_pd(x, vec, 4);
}
inline f64x4 shuffle_up(f64x4 vec, f64 first) {
  auto x = _mm256_insertf128_pd(f64x4::expand_undef(first), _mm256_castpd256_pd128(vec), 1);
  return _mm256_shuffle_pd(x, vec, 4);
}

inline f64x4 shuffle_down(f64x4 vec) {
  auto tmp = _mm256_permute4x64_pd(vec, 0xF9);
  return _mm256_castps_pd(_mm256_blend_ps(_mm256_castpd_ps(tmp), _mm256_setzero_ps(), 0xC0));
}
inline f64x4 shuffle_down(f64x4 vec, f64 last) {
  auto bc = _mm256_set1_pd(last);
  auto tmp = _mm256_permute4x64_pd(vec, 0xF9);
  return _mm256_castps_pd(_mm256_blend_ps(_mm256_castpd_ps(tmp), _mm256_castpd_ps(bc), 0xC0));
}

template<bool b0, bool b1, bool b2, bool b3>
inline f64x4 blend(f64x4 a, f64x4 b) {
  return _mm256_blend_pd(
    b, a, unsigned{b0} | (unsigned{b1} << 1U) | (unsigned{b2} << 2U) | (unsigned{b3} << 3U));
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F64X4_HPP
