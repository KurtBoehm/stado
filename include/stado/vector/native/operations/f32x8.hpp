#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F32X8_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F32X8_HPP

#include <limits>

#include <immintrin.h>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/operations/f32x4.hpp"
#include "stado/vector/native/operations/i32x8.hpp"
#include "stado/vector/native/operations/u32x8.hpp"
#include "stado/vector/native/types/f32x8.hpp"
#include "stado/vector/native/types/i32x8.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX
namespace stado {
// vector operator + : add element by element
inline f32x8 operator+(const f32x8 a, const f32x8 b) {
  return _mm256_add_ps(a, b);
}

// vector operator + : add vector and scalar
inline f32x8 operator+(const f32x8 a, f32 b) {
  return a + f32x8(b);
}
inline f32x8 operator+(f32 a, const f32x8 b) {
  return f32x8(a) + b;
}

// vector operator += : add
inline f32x8& operator+=(f32x8& a, const f32x8 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline f32x8 operator++(f32x8& a, int) {
  f32x8 a0 = a;
  a = a + 1.0F;
  return a0;
}

// prefix operator ++
inline f32x8& operator++(f32x8& a) {
  a = a + 1.0F;
  return a;
}

// vector operator - : subtract element by element
inline f32x8 operator-(const f32x8 a, const f32x8 b) {
  return _mm256_sub_ps(a, b);
}

// vector operator - : subtract vector and scalar
inline f32x8 operator-(const f32x8 a, f32 b) {
  return a - f32x8(b);
}
inline f32x8 operator-(f32 a, const f32x8 b) {
  return f32x8(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
inline f32x8 operator-(const f32x8 a) {
  return _mm256_xor_ps(a, f32x8(-0.0F));
}

// vector operator -= : subtract
inline f32x8& operator-=(f32x8& a, const f32x8 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline f32x8 operator--(f32x8& a, int) {
  f32x8 a0 = a;
  a = a - 1.0F;
  return a0;
}

// prefix operator --
inline f32x8& operator--(f32x8& a) {
  a = a - 1.0F;
  return a;
}

// vector operator * : multiply element by element
inline f32x8 operator*(const f32x8 a, const f32x8 b) {
  return _mm256_mul_ps(a, b);
}

// vector operator * : multiply vector and scalar
inline f32x8 operator*(const f32x8 a, f32 b) {
  return a * f32x8(b);
}
inline f32x8 operator*(f32 a, const f32x8 b) {
  return f32x8(a) * b;
}

// vector operator *= : multiply
inline f32x8& operator*=(f32x8& a, const f32x8 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer
inline f32x8 operator/(const f32x8 a, const f32x8 b) {
  return _mm256_div_ps(a, b);
}

// vector operator / : divide vector and scalar
inline f32x8 operator/(const f32x8 a, f32 b) {
  return a / f32x8(b);
}
inline f32x8 operator/(f32 a, const f32x8 b) {
  return f32x8(a) / b;
}

// vector operator /= : divide
inline f32x8& operator/=(f32x8& a, const f32x8 b) {
  a = a / b;
  return a;
}

// vector operator == : returns true for elements for which a == b
inline Mask<32, 8> operator==(const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_ps_mask(a, b, 0);
#else
  return _mm256_cmp_ps(a, b, 0);
#endif
}

// vector operator != : returns true for elements for which a != b
inline Mask<32, 8> operator!=(const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_ps_mask(a, b, 4);
#else
  return _mm256_cmp_ps(a, b, 4);
#endif
}

// vector operator < : returns true for elements for which a < b
inline Mask<32, 8> operator<(const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_ps_mask(a, b, 1);
#else
  return _mm256_cmp_ps(a, b, 1);
#endif
}

// vector operator <= : returns true for elements for which a <= b
inline Mask<32, 8> operator<=(const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_ps_mask(a, b, 2);
#else
  return _mm256_cmp_ps(a, b, 2);
#endif
}

// vector operator > : returns true for elements for which a > b
inline Mask<32, 8> operator>(const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_ps_mask(a, b, 6);
#else
  return b < a;
#endif
}

// vector operator >= : returns true for elements for which a >= b
inline Mask<32, 8> operator>=(const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_ps_mask(a, b, 5);
#else
  return b <= a;
#endif
}

// Bitwise logical operators

// vector operator & : bitwise and
inline f32x8 operator&(const f32x8 a, const f32x8 b) {
  return _mm256_and_ps(a, b);
}

// vector operator &= : bitwise and
inline f32x8& operator&=(f32x8& a, const f32x8 b) {
  a = a & b;
  return a;
}

// vector operator & : bitwise and of f32x8 and Mask<32, 8>
inline f32x8 operator&(const f32x8 a, const Mask<32, 8> b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_maskz_mov_ps(b, a);
#else
  return _mm256_and_ps(a, b);
#endif
}
inline f32x8 operator&(const Mask<32, 8> a, const f32x8 b) {
  return b & a;
}

// vector operator | : bitwise or
inline f32x8 operator|(const f32x8 a, const f32x8 b) {
  return _mm256_or_ps(a, b);
}

// vector operator |= : bitwise or
inline f32x8& operator|=(f32x8& a, const f32x8 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline f32x8 operator^(const f32x8 a, const f32x8 b) {
  return _mm256_xor_ps(a, b);
}

// vector operator ^= : bitwise xor
inline f32x8& operator^=(f32x8& a, const f32x8 b) {
  a = a ^ b;
  return a;
}

// vector operator ! : logical not. Returns Boolean vector
inline Mask<32, 8> operator!(const f32x8 a) {
  return a == f32x8(0.0F);
}

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
inline f32x8 select(const Mask<32, 8> s, const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mov_ps(b, s, a);
#else
  return _mm256_blendv_ps(b, a, s);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline f32x8 if_add(const Mask<32, 8> f, const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_mask_add_ps(a, f, a, b);
#else
  return a + (f32x8(f) & b);
#endif
}

// Conditional subtract
inline f32x8 if_sub(const Mask<32, 8> f, const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_mask_sub_ps(a, f, a, b);
#else
  return a - (f32x8(f) & b);
#endif
}

// Conditional multiply
inline f32x8 if_mul(const Mask<32, 8> f, const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_mask_mul_ps(a, f, a, b);
#else
  return a * select(f, b, 1.F);
#endif
}

// Conditional divide
inline f32x8 if_div(const Mask<32, 8> f, const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_mask_div_ps(a, f, a, b);
#else
  return a / select(f, b, 1.F);
#endif
}

// Sign functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
inline Mask<32, 8> sign_bit(const f32x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  const i32x8 t1 = _mm256_castps_si256(a); // reinterpret as 32-bit integer
  const i32x8 t2 = t1 >> 31; // extend sign bit
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return t2 != i32x8(0);
#else
  return _mm256_castsi256_ps(t2); // reinterpret as 32-bit Boolean
#endif
#else
  return {sign_bit(a.get_low()), sign_bit(a.get_high())};
#endif
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
inline f32x8 sign_combine(const f32x8 a, const f32x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_castsi256_ps(
    _mm256_ternarylogic_epi32(_mm256_castps_si256(a), _mm256_castps_si256(b),
                              (i32x8(std::numeric_limits<i32>::min())), 0x78));
#else
  return a ^ (b & f32x8(-0.0F));
#endif
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, subnormal or zero,
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
inline Mask<32, 8> is_finite(const f32x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return __mmask8(~_mm256_fpclass_ps_mask(a, 0x99));
#elif STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  i32x8 t1 = _mm256_castps_si256(a); // reinterpret as 32-bit integer
  i32x8 t2 = t1 << 1; // shift out sign bit
  // exponent field is not all 1s
  Mask<32, 8> t3 = i32x8(t2 & i32(0xFF000000)) != i32x8(i32(0xFF000000));
  return t3;
#else
  return {is_finite(a.get_low()), is_finite(a.get_high())};
#endif
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
inline Mask<32, 8> is_inf(const f32x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_fpclass_ps_mask(a, 0x18);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2 //  256 bit integer vectors are available
  i32x8 t1 = _mm256_castps_si256(a); // reinterpret as 32-bit integer
  i32x8 t2 = t1 << 1; // shift out sign bit
  return t2 == i32x8(i32(0xFF000000)); // exponent is all 1s, fraction is 0
#else
  return {is_inf(a.get_low()), is_inf(a.get_high())};
#endif
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
inline Mask<32, 8> is_nan(const f32x8 a) {
  // assume that compiler does not optimize this away with -ffinite-math-only:
  return _mm256_fpclass_ps_mask(a, 0x81);
}
// #elif defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
//__attribute__((optimize("-fno-unsafe-math-optimizations")))
//  inline Mask<32, 8> is_nan(f32x8 const a) {
//     return a != a; // not safe with -ffinite-math-only compiler option
// }
#elif (defined(__GNUC__) || defined(__clang__)) && !defined(__INTEL_COMPILER)
inline Mask<32, 8> is_nan(const f32x8 a) {
  __m256 aa = a;
  __m256 unordered;
  __asm volatile("vcmpps $3, %1, %1, %0" : "=v"(unordered) : "v"(aa));
  return {unordered};
}
#else
inline Mask<32, 8> is_nan(const f32x8 a) {
  // assume that compiler does not optimize this away with -ffinite-math-only:
  return _mm256_cmp_ps(a, a, 3); // compare unordered
  // return a != a; // This is not safe with -ffinite-math-only, -ffast-math, or /fp:fast compiler
  // option
}
#endif

// Function is_subnormal: gives true for elements that are subnormal (denormal)
// false for finite numbers, zero, NAN and INF
inline Mask<32, 8> is_subnormal(const f32x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_fpclass_ps_mask(a, 0x20);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  i32x8 t1 = _mm256_castps_si256(a); // reinterpret as 32-bit integer
  i32x8 t2 = t1 << 1; // shift out sign bit
  i32x8 t3 = i32(0xFF000000); // exponent mask
  i32x8 t4 = t2 & t3; // exponent
  i32x8 t5 = _mm256_andnot_si256(t3, t2); // fraction
  return {t4 == i32x8(0) && t5 != i32x8(0)}; // exponent = 0 and fraction != 0
#else
  return {is_subnormal(a.get_low()), is_subnormal(a.get_high())};
#endif
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
inline Mask<32, 8> is_zero_or_subnormal(const f32x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_fpclass_ps_mask(a, 0x26);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  i32x8 t = _mm256_castps_si256(a); // reinterpret as 32-bit integer
  t &= 0x7F800000; // isolate exponent
  return t == i32x8(0); // exponent = 0
#else
  return {is_zero_or_subnormal(a.get_low()), is_zero_or_subnormal(a.get_high())};
#endif
}

// change signs on vectors f32x8
// Each index i0 - i7 is 1 for changing sign on the corresponding element, 0 for no change
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline f32x8 change_sign(const f32x8 a) {
  if ((i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7) == 0) {
    return a;
  }
  __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(
    i0 ? 0x80000000U : 0U, i1 ? 0x80000000U : 0U, i2 ? 0x80000000U : 0U, i3 ? 0x80000000U : 0U,
    i4 ? 0x80000000U : 0U, i5 ? 0x80000000U : 0U, i6 ? 0x80000000U : 0U, i7 ? 0x80000000U : 0U));
  return _mm256_xor_ps(a, mask);
}

// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
inline f32 horizontal_add(const f32x8 a) {
  return horizontal_add(a.get_low() + a.get_high());
}

// function max: a > b ? a : b
inline f32x8 max(const f32x8 a, const f32x8 b) {
  return _mm256_max_ps(a, b);
}

// function min: a < b ? a : b
inline f32x8 min(const f32x8 a, const f32x8 b) {
  return _mm256_min_ps(a, b);
}
// NAN-safe versions of maximum and minimum are in vector-convert.hpp

// function abs: absolute value
inline f32x8 abs(const f32x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm256_range_ps(a, a, 8);
#else
  __m256 mask = _mm256_castsi256_ps(
    _mm256_setr_epi32(i32(0x7FFFFFFF), i32(0x7FFFFFFF), i32(0x7FFFFFFF), i32(0x7FFFFFFF),
                      i32(0x7FFFFFFF), i32(0x7FFFFFFF), i32(0x7FFFFFFF), i32(0x7FFFFFFF)));
  return _mm256_and_ps(a, mask);
#endif
}

// function sqrt: square root
inline f32x8 sqrt(const f32x8 a) {
  return _mm256_sqrt_ps(a);
}

// function square: a * a
inline f32x8 square(const f32x8 a) {
  return a * a;
}

// The purpose of this template is to prevent implicit conversion of a f32
// exponent to int when calling pow(vector, f32) and vectormath_exp.h is not included
template<typename TT>
static f32x8 pow(f32x8 x0, TT n);

// Raise floating point numbers to integer power n
template<>
inline f32x8 pow<i32>(const f32x8 x0, const i32 n) {
  return pow_template_i<f32x8>(x0, n);
}

// allow conversion from unsigned int
template<>
inline f32x8 pow<u32>(const f32x8 x0, const u32 n) {
  return pow_template_i<f32x8>(x0, i32(n));
}

// Raise floating point numbers to integer power n, where n is a compile-time constant
template<i32 n>
inline f32x8 pow(const f32x8 a, ConstInt<n> /*n*/) {
  return pow_n<f32x8, n>(a);
}

// function round: round to nearest integer (even). (result as f32 vector)
inline f32x8 round(const f32x8 a) {
  return _mm256_round_ps(a, 0 + 8);
}

// function truncate: round towards zero. (result as f32 vector)
inline f32x8 truncate(const f32x8 a) {
  return _mm256_round_ps(a, 3 + 8);
}

// function floor: round towards minus infinity. (result as f32 vector)
inline f32x8 floor(const f32x8 a) {
  return _mm256_round_ps(a, 1 + 8);
}

// function ceil: round towards plus infinity. (result as f32 vector)
inline f32x8 ceil(const f32x8 a) {
  return _mm256_round_ps(a, 2 + 8);
}

#if STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
// function roundi: round to nearest integer (even). (result as integer vector)
inline i32x8 roundi(const f32x8 a) {
  // Note: assume MXCSR control register is set to rounding
  return _mm256_cvtps_epi32(a);
}

// function truncatei: round towards zero. (result as integer vector)
inline i32x8 truncatei(const f32x8 a) {
  return _mm256_cvttps_epi32(a);
}

// function to_f32: convert integer vector to f32 vector
inline f32x8 to_f32(const i32x8 a) {
  return _mm256_cvtepi32_ps(a);
}

// function to_f32: convert unsigned integer vector to f32 vector
inline f32x8 to_f32(const u32x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && !defined(_MSC_VER)
  // _mm256_cvtepu32_ps missing in VS2019
  return _mm256_cvtepu32_ps(a);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  return _mm512_castps512_ps256(_mm512_cvtepu32_ps(_mm512_castsi256_si512(a)));
#else
  f32x8 b = to_f32(i32x8(a & 0xFFFFF)); // 20 bits
  f32x8 c = to_f32(i32x8(a >> 20)); // remaining bits
  f32x8 d = b + c * 1048576.f; // 2^20
  return d;
#endif
}
#endif

// Fused multiply and add functions

// Multiply and add
inline f32x8 mul_add(const f32x8 a, const f32x8 b, const f32x8 c) {
#ifdef __FMA__
  return _mm256_fmadd_ps(a, b, c);
#elif defined(__FMA4__)
  return _mm256_macc_ps(a, b, c);
#else
  return a * b + c;
#endif
}

// Multiply and subtract
inline f32x8 mul_sub(const f32x8 a, const f32x8 b, const f32x8 c) {
#ifdef __FMA__
  return _mm256_fmsub_ps(a, b, c);
#elif defined(__FMA4__)
  return _mm256_msub_ps(a, b, c);
#else
  return a * b - c;
#endif
}

// Multiply and inverse subtract
inline f32x8 nmul_add(const f32x8 a, const f32x8 b, const f32x8 c) {
#ifdef __FMA__
  return _mm256_fnmadd_ps(a, b, c);
#elif defined(__FMA4__)
  return _mm256_nmacc_ps(a, b, c);
#else
  return c - a * b;
#endif
}

// Multiply and subtract with extra precision on the intermediate calculations,
// even if FMA instructions not supported, using Veltkamp-Dekker split
// This is used in mathematical functions. Do not use it in general code
// because it is inaccurate in certain cases
inline f32x8 mul_sub_x(const f32x8 a, const f32x8 b, const f32x8 c) {
#ifdef __FMA__
  return _mm256_fmsub_ps(a, b, c);
#elif defined(__FMA4__)
  return _mm256_msub_ps(a, b, c);
#else
  // calculate a * b - c with extra precision
  const u32 b12 = u32(-(1 << 12)); // mask to remove lower 12 bits
  f32x8 upper_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32(b12)));
  f32x8 a_high = a & upper_mask; // split into high and low parts
  f32x8 b_high = b & upper_mask;
  f32x8 a_low = a - a_high;
  f32x8 b_low = b - b_high;
  f32x8 r1 = a_high * b_high; // this product is exact
  f32x8 r2 = r1 - c; // subtract c from high product
  f32x8 r3 = r2 + (a_high * b_low + b_high * a_low) + a_low * b_low; // add rest of product
  return r3; // + ((r2 - r1) + c);
#endif
}

// Approximate math functions

// approximate reciprocal (Faster than 1.f / a. relative accuracy better than 2^-11)
inline f32x8 approx_recipr(const f32x8 a) {
#ifdef __AVX512ER__ // AVX512ER: full precision
  // todo: if future processors have both AVX512ER and AVX512VL:
  // _mm256_rcp28_round_ps(a, _MM_FROUND_NO_EXC);
  return _mm512_castps512_ps256(
    _mm512_rcp28_round_ps(_mm512_castps256_ps512(a), _MM_FROUND_NO_EXC));
#elif STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL: 14 bit precision
  return _mm256_rcp14_ps(a);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F // AVX512F: 14 bit precision
  return _mm512_castps512_ps256(_mm512_rcp14_ps(_mm512_castps256_ps512(a)));
#else // AVX: 11 bit precision
  return _mm256_rcp_ps(a);
#endif
}

// approximate reciprocal squareroot (Faster than 1.f / sqrt(a). Relative accuracy better than
// 2^-11)
inline f32x8 approx_rsqrt(const f32x8 a) {
// use more accurate version if available. (none of these will raise exceptions on zero)
#ifdef __AVX512ER__ // AVX512ER: full precision
  // todo: if future processors have both AVX512ER and AVX521VL:
  // _mm256_rsqrt28_round_ps(a, _MM_FROUND_NO_EXC);
  return _mm512_castps512_ps256(
    _mm512_rsqrt28_round_ps(_mm512_castps256_ps512(a), _MM_FROUND_NO_EXC));
#elif STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(_mm256_rsqrt14_ps) // missing in VS2019
  return _mm256_rsqrt14_ps(a);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F // AVX512F: 14 bit precision
  return _mm512_castps512_ps256(_mm512_rsqrt14_ps(_mm512_castps256_ps512(a)));
#else // AVX: 11 bit precision
  return _mm256_rsqrt_ps(a);
#endif
}

// Math functions using fast bit manipulation

#if STADO_INSTRUCTION_SET >= STADO_AVX2
// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
inline i32x8 exponent(const f32x8 a) {
  const u32x8 t1 = _mm256_castps_si256(a); // reinterpret as 32-bit integer
  const u32x8 t2 = t1 << 1; // shift out sign bit
  const u32x8 t3 = t2 >> 24; // shift down logical to position 0
  i32x8 t4 = i32x8(t3) - 0x7F; // subtract bias from exponent
  return t4;
}
#endif

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0f) = 1.0f, fraction(5.0f) = 1.25f
inline f32x8 fraction(const f32x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_getmant_ps(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available
  u32x8 t1 = _mm256_castps_si256(a); // reinterpret as 32-bit integer
  u32x8 t2 = (t1 & 0x007FFFFF) | 0x3F800000; // set exponent to 0 + bias
  return _mm256_castsi256_ps(t2);
#else
  return f32x8(fraction(a.get_low()), fraction(a.get_high()));
#endif
}

#if STADO_INSTRUCTION_SET >= STADO_AVX2
// Fast calculation of pow(2,n) with n integer
// n  =    0 gives 1.0f
// n >=  128 gives +INF
// n <= -127 gives 0.0f
// This function will never produce subnormals, and never raise exceptions
inline f32x8 exp2(const i32x8 n) {
  const i32x8 t1 = max(n, -0x7F); // limit to allowed range
  const i32x8 t2 = min(t1, 0x80);
  const i32x8 t3 = t2 + 0x7F; // add bias
  const i32x8 t4 = t3 << 23; // put exponent into position 23
  return _mm256_castsi256_ps(t4); // reinterpret as f32
}
// inline f32x8 exp2(f32x8 const x); // defined in vectormath_exp.h

inline f32x8 shuffle_up(f32x8 vec) {
  const __m256i idxs = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
  auto perm = _mm256_permutevar8x32_ps(vec, idxs);
  return _mm256_blend_ps(perm, _mm256_setzero_ps(), 0x1);
}
inline f32x8 shuffle_up(f32x8 vec, f32 first) {
  const __m256i idxs = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
  auto perm = _mm256_permutevar8x32_ps(vec, idxs);
  return _mm256_blend_ps(perm, f32x8::expand_undef(first), 0x1);
}

inline f32x8 shuffle_down(f32x8 vec) {
  const __m256i idxs = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
  auto perm = _mm256_permutevar8x32_ps(vec, idxs);
  return _mm256_blend_ps(perm, _mm256_setzero_ps(), 0x80);
}
inline f32x8 shuffle_down(f32x8 vec, f32 last) {
  const __m256i idxs = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
  auto perm = _mm256_permutevar8x32_ps(vec, idxs);
  return _mm256_blend_ps(perm, _mm256_set1_ps(last), 0x80);
}
#endif

template<bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7>
inline f32x8 blend(f32x8 a, f32x8 b) {
  return _mm256_blend_ps(b, a,
                         unsigned{b0} | (unsigned{b1} << 1U) | (unsigned{b2} << 2U) |
                           (unsigned{b3} << 3U) | (unsigned{b4} << 4U) | (unsigned{b5} << 5U) |
                           (unsigned{b6} << 6U) | (unsigned{b7} << 7U));
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F32X8_HPP
