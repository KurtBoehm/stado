#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F64X2_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F64X2_HPP

#include <limits>

#include <emmintrin.h>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/operations/f32x4.hpp"
#include "stado/vector/native/operations/i64x2.hpp"
#include "stado/vector/native/operations/u64x2.hpp"
#include "stado/vector/native/types/f32x4.hpp"
#include "stado/vector/native/types/f64x2.hpp"
#include "stado/vector/native/types/i32x4.hpp"
#include "stado/vector/native/types/i64x2.hpp"

namespace stado {
// vector operator + : add element by element
inline f64x2 operator+(const f64x2 a, const f64x2 b) {
  return _mm_add_pd(a, b);
}

// vector operator + : add vector and scalar
inline f64x2 operator+(const f64x2 a, f64 b) {
  return a + f64x2(b);
}
inline f64x2 operator+(f64 a, const f64x2 b) {
  return f64x2(a) + b;
}

// vector operator += : add
inline f64x2& operator+=(f64x2& a, const f64x2 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline f64x2 operator++(f64x2& a, int) {
  f64x2 a0 = a;
  a = a + 1.0;
  return a0;
}

// prefix operator ++
inline f64x2& operator++(f64x2& a) {
  a = a + 1.0;
  return a;
}

// vector operator - : subtract element by element
inline f64x2 operator-(const f64x2 a, const f64x2 b) {
  return _mm_sub_pd(a, b);
}

// vector operator - : subtract vector and scalar
inline f64x2 operator-(const f64x2 a, f64 b) {
  return a - f64x2(b);
}
inline f64x2 operator-(f64 a, const f64x2 b) {
  return f64x2(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
inline f64x2 operator-(const f64x2 a) {
  return _mm_xor_pd(a, _mm_castsi128_pd(_mm_setr_epi32(0, std::numeric_limits<i32>::min(), 0,
                                                       std::numeric_limits<i32>::min())));
}

// vector operator -= : subtract
inline f64x2& operator-=(f64x2& a, const f64x2 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline f64x2 operator--(f64x2& a, int) {
  f64x2 a0 = a;
  a = a - 1.0;
  return a0;
}

// prefix operator --
inline f64x2& operator--(f64x2& a) {
  a = a - 1.0;
  return a;
}

// vector operator * : multiply element by element
inline f64x2 operator*(const f64x2 a, const f64x2 b) {
  return _mm_mul_pd(a, b);
}

// vector operator * : multiply vector and scalar
inline f64x2 operator*(const f64x2 a, f64 b) {
  return a * f64x2(b);
}
inline f64x2 operator*(f64 a, const f64x2 b) {
  return f64x2(a) * b;
}

// vector operator *= : multiply
inline f64x2& operator*=(f64x2& a, const f64x2 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer
inline f64x2 operator/(const f64x2 a, const f64x2 b) {
  return _mm_div_pd(a, b);
}

// vector operator / : divide vector and scalar
inline f64x2 operator/(const f64x2 a, f64 b) {
  return a / f64x2(b);
}
inline f64x2 operator/(f64 a, const f64x2 b) {
  return f64x2(a) / b;
}

// vector operator /= : divide
inline f64x2& operator/=(f64x2& a, const f64x2 b) {
  a = a / b;
  return a;
}

// vector operator == : returns true for elements for which a == b
inline Mask<64, 2> operator==(const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_pd_mask(a, b, 0);
#else
  return _mm_cmpeq_pd(a, b);
#endif
}

// vector operator != : returns true for elements for which a != b
inline Mask<64, 2> operator!=(const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_pd_mask(a, b, 4);
#else
  return _mm_cmpneq_pd(a, b);
#endif
}

// vector operator < : returns true for elements for which a < b
inline Mask<64, 2> operator<(const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_pd_mask(a, b, 1);
#else
  return _mm_cmplt_pd(a, b);
#endif
}

// vector operator <= : returns true for elements for which a <= b
inline Mask<64, 2> operator<=(const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_pd_mask(a, b, 2);
#else
  return _mm_cmple_pd(a, b);
#endif
}

// vector operator > : returns true for elements for which a > b
inline Mask<64, 2> operator>(const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_pd_mask(a, b, 6);
#else
  return b < a;
#endif
}

// vector operator >= : returns true for elements for which a >= b
inline Mask<64, 2> operator>=(const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_pd_mask(a, b, 5);
#else
  return b <= a;
#endif
}

// Bitwise logical operators

// vector operator & : bitwise and
inline f64x2 operator&(const f64x2 a, const f64x2 b) {
  return _mm_and_pd(a, b);
}

// vector operator &= : bitwise and
inline f64x2& operator&=(f64x2& a, const f64x2 b) {
  a = a & b;
  return a;
}

// vector operator & : bitwise and of f64x2 and Mask<64, 2>
inline f64x2 operator&(const f64x2 a, const Mask<64, 2> b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_maskz_mov_pd(b, a);
#else
  return _mm_and_pd(a, b);
#endif
}
inline f64x2 operator&(const Mask<64, 2> a, const f64x2 b) {
  return b & a;
}

// vector operator | : bitwise or
inline f64x2 operator|(const f64x2 a, const f64x2 b) {
  return _mm_or_pd(a, b);
}

// vector operator |= : bitwise or
inline f64x2& operator|=(f64x2& a, const f64x2 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline f64x2 operator^(const f64x2 a, const f64x2 b) {
  return _mm_xor_pd(a, b);
}

// vector operator ^= : bitwise xor
inline f64x2& operator^=(f64x2& a, const f64x2 b) {
  a = a ^ b;
  return a;
}

// vector operator ! : logical not. Returns Boolean vector
inline Mask<64, 2> operator!(const f64x2 a) {
  return a == f64x2(0.0);
}

// Select between two __m128d sources, element by element, with broad boolean vector.
// Corresponds to this pseudocode:
// for (int i = 0; i < 2; i++) result[i] = s[i] ? a[i] : b[i];
// Each element in s must be either 0 (false) or 0xFFFFFFFFFFFFFFFF (true). No other
// No other values are allowed for broad boolean vectors.
// The implementation depends on the instruction set:
// If SSE4.1 is supported then only bit 63 in each dword of s is checked,
// otherwise all bits in s are used.
inline __m128d selectd(const __m128d s, const __m128d a, const __m128d b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_blendv_pd(b, a, s);
#else
  return _mm_or_pd(_mm_and_pd(s, a), _mm_andnot_pd(s, b));
#endif
}

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 2; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFFFFFFFFFFFFFFFF (true).
// No other values are allowed.
inline f64x2 select(const Mask<64, 2> s, const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mov_pd(b, s, a);
#else
  return selectd(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline f64x2 if_add(const Mask<64, 2> f, const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_mask_add_pd(a, f, a, b);
#else
  return a + (f64x2(f) & b);
#endif
}

// Conditional subtract
inline f64x2 if_sub(const Mask<64, 2> f, const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_mask_sub_pd(a, f, a, b);
#else
  return a - (f64x2(f) & b);
#endif
}

// Conditional multiply
inline f64x2 if_mul(const Mask<64, 2> f, const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_mask_mul_pd(a, f, a, b);
#else
  return a * select(f, b, 1.);
#endif
}

// Conditional divide
inline f64x2 if_div(const Mask<64, 2> f, const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_mask_div_pd(a, f, a, b);
#else
  return a / select(f, b, 1.);
#endif
}

// Sign functions

// change signs on vectors f64x2
// Each index i0 - i1 is 1 for changing sign on the corresponding element, 0 for no change
template<int i0, int i1>
inline f64x2 change_sign(const f64x2 a) {
  if ((i0 | i1) == 0) {
    return a;
  }
  __m128i mask = _mm_setr_epi32(0, i0 ? 0x80000000 : 0, 0, i1 ? 0x80000000 : 0);
  return _mm_xor_pd(a, _mm_castsi128_pd(mask)); // flip sign bits
}

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0, -INF and -NAN
// Note that sign_bit(f64x2(-0.0)) gives true, while f64x2(-0.0)
// < f64x2(0.0) gives false
inline Mask<64, 2> sign_bit(const f64x2 a) {
  const i64x2 t1 = _mm_castpd_si128(a); // reinterpret as 64-bit integer
  const i64x2 t2 = t1 >> 63; // extend sign bit
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return t2 != i64x2(0);
#else
  return _mm_castsi128_pd(t2); // reinterpret as 64-bit Boolean
#endif
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
inline f64x2 sign_combine(const f64x2 a, const f64x2 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_castsi128_pd(_mm_ternarylogic_epi64(_mm_castpd_si128(a), _mm_castpd_si128(b),
                                                 (i64x2(i64(0x8000000000000000))), 0x78));
#else
  return a ^ (b & f64x2(-0.0));
#endif
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, denormal or zero,
// false for INF and NAN
inline Mask<64, 2> is_finite(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return __mmask8(_mm_fpclass_pd_mask(a, 0x99) ^ 0x03);
#else
  i64x2 t1 = _mm_castpd_si128(a); // reinterpret as integer
  i64x2 t2 = t1 << 1; // shift out sign bit
  i64x2 t3 = i64(0xFFE0000000000000LL); // exponent mask
  Mask<64, 2> t4 = i64x2(t2 & t3) != t3; // exponent field is not all 1s
  return t4;
#endif
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
inline Mask<64, 2> is_inf(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_fpclass_pd_mask(a, 0x18);
#else
  i64x2 t1 = _mm_castpd_si128(a); // reinterpret as integer
  i64x2 t2 = t1 << 1; // shift out sign bit
  return t2 == i64x2(i64(0xFFE0000000000000LL)); // exponent is all 1s, fraction is 0
#endif
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
inline Mask<64, 2> is_nan(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  // assume that compiler does not optimize this away with -ffinite-math-only:
  return Mask<64, 2>(_mm_fpclass_pd_mask(a, 0x81));

  // #elif defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
  //__attribute__((optimize("-fno-unsafe-math-optimizations")))
  //  inline Mask<32, 4> is_nan(f32x4 const a) {
  //     return a != a; // not safe with -ffinite-math-only compiler option
  // }

#elif STADO_INSTRUCTION_SET >= STADO_AVX

#if (defined(__GNUC__) || defined(__clang__)) && !defined(__INTEL_COMPILER)
  // use assembly to avoid optimizing away with -ffinite-math-only and similar options
  __m128d aa = a;
  __m128i unordered;
  __asm volatile("vcmppd $3,  %1, %1, %0" : "=x"(unordered) : "x"(aa));
  return Mask<64, 2>(unordered);
#else
  return _mm_cmp_pd(a, a, 3); // compare unordered
#endif
#else
  return a !=
         a; // This is not safe with -ffinite-math-only, -ffast-math, or /fp:fast compiler option
#endif
}

// Function is_subnormal: gives true for elements that are subnormal (denormal)
// false for finite numbers, zero, NAN and INF
inline Mask<64, 2> is_subnormal(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_fpclass_pd_mask(a, 0x20);
#else
  i64x2 t1 = _mm_castpd_si128(a); // reinterpret as 32-bit integer
  i64x2 t2 = t1 << 1; // shift out sign bit
  i64x2 t3 = i64(0xFFE0000000000000LL); // exponent mask
  i64x2 t4 = t2 & t3; // exponent
  i64x2 t5 = _mm_andnot_si128(t3, t2); // fraction
  return Mask<64, 2>((t4 == i64x2(0)) & (t5 != i64x2(0))); // exponent = 0 and fraction != 0
#endif
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
inline Mask<64, 2> is_zero_or_subnormal(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_fpclass_pd_mask(a, 0x26);
#else
  i64x2 t = _mm_castpd_si128(a); // reinterpret as 32-bit integer
  t &= i64x2(i64(0x7FF0000000000000LL)); // isolate exponent
  return t == i64x2(0); // exponent = 0
#endif
}

// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
inline f64 horizontal_add(const f64x2 a) {
  // This version is OK
  const __m128d t1 = _mm_unpackhi_pd(a, a);
  const __m128d t2 = _mm_add_pd(a, t1);
  return _mm_cvtsd_f64(t2);
}

// function max: a > b ? a : b
inline f64x2 max(const f64x2 a, const f64x2 b) {
  return _mm_max_pd(a, b);
}

// function min: a < b ? a : b
inline f64x2 min(const f64x2 a, const f64x2 b) {
  return _mm_min_pd(a, b);
}
// NAN-safe versions of maximum and minimum are in vector-convert.hpp

// function abs: absolute value
inline f64x2 abs(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm_range_pd(a, a, 8);
#else
  __m128d mask = _mm_castsi128_pd(_mm_setr_epi32(-1, 0x7FFFFFFF, -1, 0x7FFFFFFF));
  return _mm_and_pd(a, mask);
#endif
}

// function sqrt: square root
inline f64x2 sqrt(const f64x2 a) {
  return _mm_sqrt_pd(a);
}

// function square: a * a
inline f64x2 square(const f64x2 a) {
  return a * a;
}

// pow(f64x2, int):
// The purpose of this template is to prevent implicit conversion of a f32
// exponent to int when calling pow(vector, f32) and vectormath_exp.h is not included
template<typename TT>
static f64x2 pow(f64x2 x0, TT n);

// Raise floating point numbers to integer power n
template<>
inline f64x2 pow<i32>(const f64x2 x0, const i32 n) {
  return pow_template_i<f64x2>(x0, n);
}

// allow conversion from unsigned int
template<>
inline f64x2 pow<u32>(const f64x2 x0, const u32 n) {
  return pow_template_i<f64x2>(x0, i32(n));
}

// Raise floating point numbers to integer power n, where n is a compile-time constant
template<i32 n>
inline f64x2 pow(const f64x2 a, ConstInt<n> /*n*/) {
  return pow_n<f64x2, n>(a);
}

// function round: round to nearest integer (even). (result as f64 vector)
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
inline f64x2 round(const f64x2 a) {
  return _mm_round_pd(a, 0 + 8);
}
#else
// avoid unsafe optimization in function round
#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__) && \
  STADO_INSTRUCTION_SET < STADO_SSE4_1
inline f64x2 round(const f64x2 a) __attribute__((optimize("-fno-unsafe-math-optimizations")));
#elif defined(__clang__) && STADO_INSTRUCTION_SET < STADO_SSE4_1
inline f64x2 round(const f64x2 a) __attribute__((optnone));
#elif defined(FLOAT_CONTROL_PRECISE_FOR_ROUND)
#pragma float_control(push)
#pragma float_control(precise, on)
#endif
// function round: round to nearest integer (even). (result as f64 vector)
inline f64x2 round(const f64x2 a) {
  // Note: assume MXCSR control register is set to rounding
  // (don't use conversion to int, it will limit the value to +/- 2^31)
  f64x2 signmask = _mm_castsi128_pd(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000)); // -0.0
  f64x2 magic =
    _mm_castsi128_pd(_mm_setr_epi32(0, 0x43300000, 0, 0x43300000)); // magic number = 2^52
  f64x2 sign = _mm_and_pd(a, signmask); // signbit of a
  f64x2 signedmagic = _mm_or_pd(magic, sign); // magic number with sign of a
  f64x2 y = a + signedmagic - signedmagic; // round by adding magic number
#ifdef SIGNED_ZERO
  y |= (a & f64x2(-0.0)); // sign of zero
#endif
  return y;
}
#if defined(FLOAT_CONTROL_PRECISE_FOR_ROUND)
#pragma float_control(pop)
#endif
#endif

// function truncate: round towards zero. (result as f64 vector)
inline f64x2 truncate(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_round_pd(a, 3 + 8);
#else // SSE2
  f64x2 a1 = abs(a); // abs
  f64x2 y1 = round(a1); // round
  f64x2 y2 = y1 - (f64x2(1.0) & (y1 > a1)); // subtract 1 if bigger
  f64x2 y3 = y2 | (a & f64x2(-0.)); // put the sign back in
  return y3;
#endif
}

// function floor: round towards minus infinity. (result as f64 vector)
inline f64x2 floor(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_round_pd(a, 1 + 8);
#else // SSE2
  f64x2 y = round(a); // round
  y -= f64x2(1.0) & (y > a); // subtract 1 if bigger
#ifdef SIGNED_ZERO
  y |= (a & f64x2(-0.0)); // sign of zero
#endif
  return y;
#endif
}

// function ceil: round towards plus infinity. (result as f64 vector)
inline f64x2 ceil(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_round_pd(a, 2 + 8);
#else // SSE2
  f64x2 y = round(a); // round
  y += f64x2(1.0) & (y < a); // add 1 if smaller
#ifdef SIGNED_ZERO
  y |= (a & f64x2(-0.0)); // sign of zero
#endif
  return y;
#endif
}

// function truncate_to_int32: round towards zero.
inline i32x4 truncate_to_int32(const f64x2 a, const f64x2 b) {
  const i32x4 t1 = _mm_cvttpd_epi32(a);
  const i32x4 t2 = _mm_cvttpd_epi32(b);
  return _mm_unpacklo_epi64(t1, t2);
}

// function truncate_to_int32: round towards zero.
inline i32x4 truncate_to_int32(const f64x2 a) {
  return _mm_cvttpd_epi32(a);
}

// function truncatei: round towards zero. (inefficient for lower instruction sets)
inline i64x2 truncatei(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ AVX512VL
  // return _mm_maskz_cvttpd_epi64( __mmask8(0xFF), a);
  return _mm_cvttpd_epi64(a);
#else
  f64 aa[2];
  a.store(aa);
  return {i64(aa[0]), i64(aa[1])};
#endif
}

// function round_to_int: round to nearest integer (even).
// result as 32-bit integer vector
inline i32x4 round_to_int32(const f64x2 a, const f64x2 b) {
  // Note: assume MXCSR control register is set to rounding
  const i32x4 t1 = _mm_cvtpd_epi32(a);
  const i32x4 t2 = _mm_cvtpd_epi32(b);
  return _mm_unpacklo_epi64(t1, t2);
}

// function round_to_int: round to nearest integer (even).
// result as 32-bit integer vector. Upper two values of result are 0
inline i32x4 round_to_int32(const f64x2 a) {
  i32x4 t1 = _mm_cvtpd_epi32(a);
  return t1;
}

// function round_to_int64: round to nearest or even. (inefficient for lower instruction sets)
inline i64x2 roundi(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ AVX512VL
  return _mm_cvtpd_epi64(a);
#else
  return truncatei(round(a));
#endif
}

// function to_f64: convert integer vector elements to f64 vector (inefficient for lower
// instruction sets)
inline f64x2 to_f64(const i64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ AVX512VL
  return _mm_maskz_cvtepi64_pd(__mmask8(0xFF), a);
#else
  i64 aa[2];
  a.store(aa);
  return f64x2(f64(aa[0]), f64(aa[1]));
#endif
}

inline f64x2 to_f64(const u64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ AVX512VL
  return _mm_cvtepu64_pd(a);
#else
  u64 aa[2]; // inefficient
  a.store(aa);
  return f64x2(f64(aa[0]), f64(aa[1]));
#endif
}

// function to_f64_low: convert integer vector elements [0] and [1] to f64 vector
inline f64x2 to_f64_low(const i32x4 a) {
  return _mm_cvtepi32_pd(a);
}

// function to_f64_high: convert integer vector elements [2] and [3] to f64 vector
inline f64x2 to_f64_high(const i32x4 a) {
  return to_f64_low(_mm_srli_si128(a, 8));
}

// function compress: convert two f64x2 to one f32x4
inline f32x4 compress(const f64x2 low, const f64x2 high) {
  const f32x4 t1 = _mm_cvtpd_ps(low);
  const f32x4 t2 = _mm_cvtpd_ps(high);
  return _mm_shuffle_ps(t1, t2, 0x44);
}

// Function extend_low : convert f32x4 vector elements [0] and [1] to
// f64x2
inline f64x2 extend_low(const f32x4 a) {
  return _mm_cvtps_pd(a);
}

// Function extend_high : convert f32x4 vector elements [2] and [3] to
// f64x2
inline f64x2 extend_high(const f32x4 a) {
  return _mm_cvtps_pd(_mm_movehl_ps(a, a));
}

// Fused multiply and add functions

// Multiply and add
inline f64x2 mul_add(const f64x2 a, const f64x2 b, const f64x2 c) {
#ifdef __FMA__
  return _mm_fmadd_pd(a, b, c);
#elif defined(__FMA4__)
  return _mm_macc_pd(a, b, c);
#else
  return a * b + c;
#endif
}

// Multiply and subtract
inline f64x2 mul_sub(const f64x2 a, const f64x2 b, const f64x2 c) {
#ifdef __FMA__
  return _mm_fmsub_pd(a, b, c);
#elif defined(__FMA4__)
  return _mm_msub_pd(a, b, c);
#else
  return a * b - c;
#endif
}

// Multiply and inverse subtract
inline f64x2 nmul_add(const f64x2 a, const f64x2 b, const f64x2 c) {
#ifdef __FMA__
  return _mm_fnmadd_pd(a, b, c);
#elif defined(__FMA4__)
  return _mm_nmacc_pd(a, b, c);
#else
  return c - a * b;
#endif
}

// Multiply and subtract with extra precision on the intermediate calculations,
// even if FMA instructions not supported, using Veltkamp-Dekker split.
// This is used in mathematical functions. Do not use it in general code
// because it is inaccurate in certain cases
inline f64x2 mul_sub_x(const f64x2 a, const f64x2 b, const f64x2 c) {
#ifdef __FMA__
  return _mm_fmsub_pd(a, b, c);
#elif defined(__FMA4__)
  return _mm_msub_pd(a, b, c);
#else
  // calculate a * b - c with extra precision
  i64x2 upper_mask = -(1LL << 27); // mask to remove lower 27 bits
  f64x2 a_high = a & f64x2(_mm_castsi128_pd(upper_mask)); // split into high and low parts
  f64x2 b_high = b & f64x2(_mm_castsi128_pd(upper_mask));
  f64x2 a_low = a - a_high;
  f64x2 b_low = b - b_high;
  f64x2 r1 = a_high * b_high; // this product is exact
  f64x2 r2 = r1 - c; // subtract c from high product
  f64x2 r3 = r2 + (a_high * b_low + b_high * a_low) + a_low * b_low; // add rest of product
  return r3; // + ((r2 - r1) + c);
#endif
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0) = 0, exponent(0.0) = -1023, exponent(INF) = +1024, exponent(NAN) = +1024
inline i64x2 exponent(const f64x2 a) {
  const u64x2 t1 = _mm_castpd_si128(a); // reinterpret as 64-bit integer
  const u64x2 t2 = t1 << 1; // shift out sign bit
  const u64x2 t3 = t2 >> 53; // shift down logical to position 0
  i64x2 t4 = i64x2(t3) - 0x3FF; // subtract bias from exponent
  return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0) = 1.0, fraction(5.0) = 1.25
// NOTE: The name fraction clashes with an ENUM in MAC XCode CarbonCore script.h !
inline f64x2 fraction(const f64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_getmant_pd(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
#else
  u64x2 t1 = _mm_castpd_si128(a); // reinterpret as 64-bit integer
  u64x2 t2 = (t1 & 0x000FFFFFFFFFFFFFLL) | 0x3FF0000000000000LL; // set exponent to 0 + bias
  return _mm_castsi128_pd(t2);
#endif
}

// Fast calculation of pow(2,n) with n integer
// n  =     0 gives 1.0
// n >=  1024 gives +INF
// n <= -1023 gives 0.0
// This function will never produce denormals, and never raise exceptions
inline f64x2 exp2(const i64x2 n) {
  const i64x2 t1 = max(n, -0x3FF); // limit to allowed range
  const i64x2 t2 = min(t1, 0x400);
  const i64x2 t3 = t2 + 0x3FF; // add bias
  const i64x2 t4 = t3 << 52; // put exponent into position 52
  return _mm_castsi128_pd(t4); // reinterpret as f64
}
// static f64x2 exp2(f64x2 const x); // defined in
// vectormath_exp.h

/*****************************************************************************
 *
 *          Functions for reinterpretation between vector types
 *
 *****************************************************************************/

inline __m128i reinterpret_i(const __m128i x) {
  return x;
}

inline __m128i reinterpret_i(const __m128 x) {
  return _mm_castps_si128(x);
}

inline __m128i reinterpret_i(const __m128d x) {
  return _mm_castpd_si128(x);
}

inline __m128 reinterpret_f(const __m128i x) {
  return _mm_castsi128_ps(x);
}

inline __m128 reinterpret_f(const __m128 x) {
  return x;
}

inline __m128 reinterpret_f(const __m128d x) {
  return _mm_castpd_ps(x);
}

inline __m128d reinterpret_d(const __m128i x) {
  return _mm_castsi128_pd(x);
}

inline __m128d reinterpret_d(const __m128 x) {
  return _mm_castps_pd(x);
}

inline __m128d reinterpret_d(const __m128d x) {
  return x;
}

// Function infinite2d: returns a vector where all elements are +INF
inline f64x2 infinite2d() {
  return reinterpret_d(i64x2(0x7FF0000000000000));
}

// Function nan2d: returns a vector where all elements are +NAN (quiet)
inline f64x2 nan2d(u32 n = 0x10) {
  return nan_vec<f64x2>(n);
}

inline f64x2 shuffle_up(f64x2 vec) {
  return _mm_castsi128_pd(_mm_bslli_si128(_mm_castpd_si128(vec), 8));
}
inline f64x2 shuffle_up(f64x2 vec, f64 first) {
  return _mm_castps_pd(
    _mm_movelh_ps(_mm_castpd_ps(f64x2::expand_undef(first)), _mm_castpd_ps(vec)));
}

inline f64x2 shuffle_down(f64x2 last) {
  return _mm_castsi128_pd(_mm_bsrli_si128(_mm_castpd_si128(last), 8));
}
inline f64x2 shuffle_down(f64x2 vec, f64 first) {
  return _mm_castps_pd(
    _mm_shuffle_ps(_mm_castpd_ps(vec), _mm_castpd_ps(f64x2::expand_undef(first)), 0x4E));
}

template<bool b0, bool b1>
inline f64x2 blend(f64x2 a, f64x2 b) {
  return _mm_blend_pd(b, a, unsigned{b0} | (unsigned{b1} << 1U));
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F64X2_HPP
