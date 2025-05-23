#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F32X16_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F32X16_HPP

#include <limits>

#include <immintrin.h>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/16.hpp"
#include "stado/vector/native/operations/f32x04.hpp"
#include "stado/vector/native/operations/f32x08.hpp"
#include "stado/vector/native/operations/i32x16.hpp"
#include "stado/vector/native/operations/u32x16.hpp"
#include "stado/vector/native/types/f32x16.hpp"
#include "stado/vector/native/types/i32x16.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
// vector operator + : add element by element
inline f32x16 operator+(const f32x16 a, const f32x16 b) {
  return _mm512_add_ps(a, b);
}

// vector operator + : add vector and scalar
inline f32x16 operator+(const f32x16 a, f32 b) {
  return a + f32x16(b);
}
inline f32x16 operator+(f32 a, const f32x16 b) {
  return f32x16(a) + b;
}

// vector operator += : add
inline f32x16& operator+=(f32x16& a, const f32x16 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline f32x16 operator++(f32x16& a, int) {
  f32x16 a0 = a;
  a = a + 1.0F;
  return a0;
}

// prefix operator ++
inline f32x16& operator++(f32x16& a) {
  a = a + 1.0F;
  return a;
}

// vector operator - : subtract element by element
inline f32x16 operator-(const f32x16 a, const f32x16 b) {
  return _mm512_sub_ps(a, b);
}

// vector operator - : subtract vector and scalar
inline f32x16 operator-(const f32x16 a, f32 b) {
  return a - f32x16(b);
}
inline f32x16 operator-(f32 a, const f32x16 b) {
  return f32x16(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
inline f32x16 operator-(const f32x16 a) {
  return _mm512_castsi512_ps(i32x16(_mm512_castps_si512(a)) ^ std::numeric_limits<i32>::min());
}

// vector operator -= : subtract
inline f32x16& operator-=(f32x16& a, const f32x16 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline f32x16 operator--(f32x16& a, int) {
  f32x16 a0 = a;
  a = a - 1.0F;
  return a0;
}

// prefix operator --
inline f32x16& operator--(f32x16& a) {
  a = a - 1.0F;
  return a;
}

// vector operator * : multiply element by element
inline f32x16 operator*(const f32x16 a, const f32x16 b) {
  return _mm512_mul_ps(a, b);
}

// vector operator * : multiply vector and scalar
inline f32x16 operator*(const f32x16 a, f32 b) {
  return a * f32x16(b);
}
inline f32x16 operator*(f32 a, const f32x16 b) {
  return f32x16(a) * b;
}

// vector operator *= : multiply
inline f32x16& operator*=(f32x16& a, const f32x16 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer
inline f32x16 operator/(const f32x16 a, const f32x16 b) {
  return _mm512_div_ps(a, b);
}

// vector operator / : divide vector and scalar
inline f32x16 operator/(const f32x16 a, f32 b) {
  return a / f32x16(b);
}
inline f32x16 operator/(f32 a, const f32x16 b) {
  return f32x16(a) / b;
}

// vector operator /= : divide
inline f32x16& operator/=(f32x16& a, const f32x16 b) {
  a = a / b;
  return a;
}

// vector operator == : returns true for elements for which a == b
inline CompactMask<16> operator==(const f32x16 a, const f32x16 b) {
  //    return _mm512_cmpeq_ps_mask(a, b);
  return _mm512_cmp_ps_mask(a, b, 0);
}

// vector operator != : returns true for elements for which a != b
inline CompactMask<16> operator!=(const f32x16 a, const f32x16 b) {
  //    return _mm512_cmpneq_ps_mask(a, b);
  return _mm512_cmp_ps_mask(a, b, 4);
}

// vector operator < : returns true for elements for which a < b
inline CompactMask<16> operator<(const f32x16 a, const f32x16 b) {
  //    return _mm512_cmplt_ps_mask(a, b);
  return _mm512_cmp_ps_mask(a, b, 1);
}

// vector operator <= : returns true for elements for which a <= b
inline CompactMask<16> operator<=(const f32x16 a, const f32x16 b) {
  //    return _mm512_cmple_ps_mask(a, b);
  return _mm512_cmp_ps_mask(a, b, 2);
}

// vector operator > : returns true for elements for which a > b
inline CompactMask<16> operator>(const f32x16 a, const f32x16 b) {
  return _mm512_cmp_ps_mask(a, b, 6);
}

// vector operator >= : returns true for elements for which a >= b
inline CompactMask<16> operator>=(const f32x16 a, const f32x16 b) {
  return _mm512_cmp_ps_mask(a, b, 5);
}

// Bitwise logical operators

// vector operator & : bitwise and
inline f32x16 operator&(const f32x16 a, const f32x16 b) {
  return _mm512_castsi512_ps(i32x16(_mm512_castps_si512(a)) & i32x16(_mm512_castps_si512(b)));
}

// vector operator &= : bitwise and
inline f32x16& operator&=(f32x16& a, const f32x16 b) {
  a = a & b;
  return a;
}

// vector operator & : bitwise and of f32x16 and CompactMask<16>
inline f32x16 operator&(const f32x16 a, const CompactMask<16> b) {
  return _mm512_maskz_mov_ps(b, a);
}
inline f32x16 operator&(const CompactMask<16> a, const f32x16 b) {
  return b & a;
}

// vector operator | : bitwise or
inline f32x16 operator|(const f32x16 a, const f32x16 b) {
  return _mm512_castsi512_ps(i32x16(_mm512_castps_si512(a)) | i32x16(_mm512_castps_si512(b)));
}

// vector operator |= : bitwise or
inline f32x16& operator|=(f32x16& a, const f32x16 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline f32x16 operator^(const f32x16 a, const f32x16 b) {
  return _mm512_castsi512_ps(i32x16(_mm512_castps_si512(a)) ^ i32x16(_mm512_castps_si512(b)));
}

// vector operator ^= : bitwise xor
inline f32x16& operator^=(f32x16& a, const f32x16 b) {
  a = a ^ b;
  return a;
}

// vector operator ! : logical not. Returns Boolean vector
inline CompactMask<16> operator!(const f32x16 a) {
  return a == f32x16(0.0F);
}

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 8; i++) result[i] = s[i] ? a[i] : b[i];
inline f32x16 select(const CompactMask<16> s, const f32x16 a, const f32x16 b) {
  return _mm512_mask_mov_ps(b, s, a);
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline f32x16 if_add(const CompactMask<16> f, const f32x16 a, const f32x16 b) {
  return _mm512_mask_add_ps(a, f, a, b);
}

// Conditional subtract
inline f32x16 if_sub(const CompactMask<16> f, const f32x16 a, const f32x16 b) {
  return _mm512_mask_sub_ps(a, f, a, b);
}

// Conditional multiply
inline f32x16 if_mul(const CompactMask<16> f, const f32x16 a, const f32x16 b) {
  return _mm512_mask_mul_ps(a, f, a, b);
}

// Conditional divide
inline f32x16 if_div(const CompactMask<16> f, const f32x16 a, const f32x16 b) {
  return _mm512_mask_div_ps(a, f, a, b);
}

// sign functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0f, -INF and -NAN
// Note that sign_bit(f32x16(-0.0f)) gives true, while f32x16(-0.0f) < f32x16(0.0f) gives false (the
// underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
inline CompactMask<16> sign_bit(const f32x16 a) {
  const i32x16 t1 = _mm512_castps_si512(a); // reinterpret as 32-bit integer
  return {t1 < 0};
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
inline f32x16 sign_combine(const f32x16 a, const f32x16 b) {
  // return a ^ (b & f32x16(-0.0f));
  return _mm512_castsi512_ps(
    _mm512_ternarylogic_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b),
                              (i32x16(std::numeric_limits<i32>::min())), 0x78));
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, denormal or zero,
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
inline CompactMask<16> is_finite(const f32x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  __mmask16 f = _mm512_fpclass_ps_mask(a, 0x99);
  return _mm512_knot(f);
#else
  i32x16 t1 = _mm512_castps_si512(a); // reinterpret as 32-bit integer
  i32x16 t2 = t1 << 1; // shift out sign bit
  auto t3 = i32x16(t2 & i32(0xFF000000)) != i32x16(i32(0xFF000000)); // exponent field is not all 1s
  return {t3};
#endif
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
inline CompactMask<16> is_inf(const f32x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_fpclass_ps_mask(a, 0x18);
#else
  i32x16 t1 = _mm512_castps_si512(a); // reinterpret as 32-bit integer
  i32x16 t2 = t1 << 1; // shift out sign bit
  return {t2 == i32x16(i32(0xFF000000))}; // exponent is all 1s, fraction is 0
#endif
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
inline CompactMask<16> is_nan(const f32x16 a) {
  // assume that compiler does not optimize this away with -ffinite-math-only:
  return _mm512_fpclass_ps_mask(a, 0x81);
}
// #elif defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
//__attribute__((optimize("-fno-unsafe-math-optimizations")))
//  inline CompactMask<16> is_nan(f32x16 const a) {
//     return a != a; // not safe with -ffinite-math-only compiler option
// }
#elif (defined(__GNUC__) || defined(__clang__)) && !defined(__INTEL_COMPILER)
inline CompactMask<16> is_nan(const f32x16 a) {
  __m512 aa = a;
  __mmask16 unordered;
  __asm volatile("vcmpps $3, %1, %1, %0" : "=Yk"(unordered) : "v"(aa));
  return {unordered};
}
#else
inline CompactMask<16> is_nan(const f32x16 a) {
  // assume that compiler does not optimize this away with -ffinite-math-only:
  return CompactMask<16>().load_bits(_mm512_cmp_ps_mask(a, a, 3)); // compare unordered
  // return a != a; // This is not safe with -ffinite-math-only, -ffast-math, or /fp:fast compiler
  // option
}
#endif

// Function is_subnormal: gives true for elements that are denormal (subnormal)
// false for finite numbers, zero, NAN and INF
inline CompactMask<16> is_subnormal(const f32x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_fpclass_ps_mask(a, 0x20);
#else
  i32x16 t1 = _mm512_castps_si512(a); // reinterpret as 32-bit integer
  i32x16 t2 = t1 << 1; // shift out sign bit
  i32x16 t3 = i32(0xFF000000); // exponent mask
  i32x16 t4 = t2 & t3; // exponent
  i32x16 t5 = _mm512_andnot_si512(t3, t2); // fraction
  return {t4 == i32x16(0) && t5 != i32x16(0)}; // exponent = 0 and fraction != 0
#endif
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
inline CompactMask<16> is_zero_or_subnormal(const f32x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_fpclass_ps_mask(a, 0x26);
#else
  i32x16 t = _mm512_castps_si512(a); // reinterpret as 32-bit integer
  t &= 0x7F800000; // isolate exponent
  return {t == i32x16(0)}; // exponent = 0
#endif
}

// change signs on vectors f32x16
// Each index i0 - i7 is 1 for changing sign on the corresponding element, 0 for no change
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline f32x16 change_sign(const f32x16 a) {
  static constexpr auto m = __mmask16(
    (i0 & 1) | (i1 & 1) << 1 | (i2 & 1) << 2 | (i3 & 1) << 3 | (i4 & 1) << 4 | (i5 & 1) << 5 |
    (i6 & 1) << 6 | (i7 & 1) << 7 | (i8 & 1) << 8 | (i9 & 1) << 9 | (i10 & 1) << 10 |
    (i11 & 1) << 11 | (i12 & 1) << 12 | (i13 & 1) << 13 | (i14 & 1) << 14 | (i15 & 1) << 15);
  if constexpr (u16(m) == 0) {
    return a;
  }
  __m512 s = _mm512_castsi512_ps(_mm512_maskz_set1_epi32(m, std::numeric_limits<i32>::min()));
  return a ^ s;
}

// Horizontal add: Calculates the sum of all vector elements.
inline f32 horizontal_add(const f32x16 a) {
#if defined(__INTEL_COMPILER)
  return _mm512_reduce_add_ps(a);
#else
  return horizontal_add(a.get_low() + a.get_high());
#endif
}

// function max: a > b ? a : b
inline f32x16 max(const f32x16 a, const f32x16 b) {
  return _mm512_max_ps(a, b);
}

// function min: a < b ? a : b
inline f32x16 min(const f32x16 a, const f32x16 b) {
  return _mm512_min_ps(a, b);
}
// NAN-safe versions of maximum and minimum are in vector-convert.hpp

// function abs: absolute value
inline f32x16 abs(const f32x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512DQ
  return _mm512_range_ps(a, a, 8);
#else
  return a & f32x16(_mm512_castsi512_ps(i32x16(0x7FFFFFFF)));
#endif
}

// function sqrt: square root
inline f32x16 sqrt(const f32x16 a) {
  return _mm512_sqrt_ps(a);
}

// function square: a * a
inline f32x16 square(const f32x16 a) {
  return a * a;
}

// pow(f32x16, int):
template<typename TT>
static f32x16 pow(f32x16 x0, TT n);

// Raise floating point numbers to integer power n
template<>
inline f32x16 pow<i32>(const f32x16 x0, const i32 n) {
  return pow_template_i<f32x16>(x0, n);
}

// allow conversion from unsigned int
template<>
inline f32x16 pow<u32>(const f32x16 x0, const u32 n) {
  return pow_template_i<f32x16>(x0, i32(n));
}

// Raise floating point numbers to integer power n, where n is a compile-time constant
template<i32 n>
inline f32x16 pow(const f32x16 a, ConstInt<n> /*n*/) {
  return pow_n<f32x16, n>(a);
}

// function round: round to nearest integer (even). (result as f32 vector)
inline f32x16 round(const f32x16 a) {
  return _mm512_roundscale_ps(a, 0 + 8);
}

// function truncate: round towards zero. (result as f32 vector)
inline f32x16 truncate(const f32x16 a) {
  return _mm512_roundscale_ps(a, 3 + 8);
}

// function floor: round towards minus infinity. (result as f32 vector)
inline f32x16 floor(const f32x16 a) {
  return _mm512_roundscale_ps(a, 1 + 8);
}

// function ceil: round towards plus infinity. (result as f32 vector)
inline f32x16 ceil(const f32x16 a) {
  return _mm512_roundscale_ps(a, 2 + 8);
}

// function roundi: round to nearest integer (even). (result as integer vector)
inline i32x16 roundi(const f32x16 a) {
  return _mm512_cvt_roundps_epi32(a, 0 + 8 /*_MM_FROUND_NO_EXC*/);
}

// function truncatei: round towards zero. (result as integer vector)
inline i32x16 truncatei(const f32x16 a) {
  return _mm512_cvtt_roundps_epi32(a, 0 + 8 /*_MM_FROUND_NO_EXC*/);
}

// function to_f32: convert integer vector to f32 vector
inline f32x16 to_f32(const i32x16 a) {
  return _mm512_cvtepi32_ps(a);
}

// function to_f32: convert unsigned integer vector to f32 vector
inline f32x16 to_f32(const u32x16 a) {
  return _mm512_cvtepu32_ps(a);
}

// Approximate math functions

// approximate reciprocal (Faster than 1.f / a.
// relative accuracy better than 2^-11 without AVX512, 2^-14 with AVX512F, full precision with
// AVX512ER)
inline f32x16 approx_recipr(const f32x16 a) {
#ifdef __AVX512ER__
  // AVX512ER instruction set includes fast reciprocal with better precision
  return _mm512_rcp28_round_ps(a, _MM_FROUND_NO_EXC);
#else
  return _mm512_rcp14_ps(a);
#endif
}

// approximate reciprocal squareroot (Faster than 1.f / sqrt(a).
// Relative accuracy better than 2^-11 without AVX512, 2^-14 with AVX512F, full precision with
// AVX512ER)
inline f32x16 approx_rsqrt(const f32x16 a) {
#ifdef __AVX512ER__
  // AVX512ER instruction set includes fast reciprocal squareroot with better precision
  return _mm512_rsqrt28_round_ps(a, _MM_FROUND_NO_EXC);
#else
  return _mm512_rsqrt14_ps(a);
#endif
}

// Fused multiply and add functions

// Multiply and add
inline f32x16 mul_add(const f32x16 a, const f32x16 b, const f32x16 c) {
  return _mm512_fmadd_ps(a, b, c);
}

// Multiply and subtract
inline f32x16 mul_sub(const f32x16 a, const f32x16 b, const f32x16 c) {
  return _mm512_fmsub_ps(a, b, c);
}

// Multiply and inverse subtract
inline f32x16 nmul_add(const f32x16 a, const f32x16 b, const f32x16 c) {
  return _mm512_fnmadd_ps(a, b, c);
}

// Multiply and subtract with extra precision on the intermediate calculations,
// Do not use mul_sub_x in general code because it is inaccurate in certain cases when FMA is not
// supported
inline f32x16 mul_sub_x(const f32x16 a, const f32x16 b, const f32x16 c) {
  return _mm512_fmsub_ps(a, b, c);
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
inline i32x16 exponent(const f32x16 a) {
  // return roundi(i32x16(_mm512_getexp_ps(a)));
  const u32x16 t1 = _mm512_castps_si512(a); // reinterpret as 32-bit integers
  const u32x16 t2 = t1 << 1; // shift out sign bit
  const u32x16 t3 = t2 >> 24; // shift down logical to position 0
  i32x16 t4 = i32x16(t3) - 0x7F; // subtract bias from exponent
  return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0f) = 1.0f, fraction(5.0f) = 1.25f
inline f32x16 fraction(const f32x16 a) {
  return _mm512_getmant_ps(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
}

// Fast calculation of pow(2,n) with n integer
// n  =    0 gives 1.0f
// n >=  128 gives +INF
// n <= -127 gives 0.0f
// This function will never produce denormals, and never raise exceptions
inline f32x16 exp2(const i32x16 n) {
  const i32x16 t1 = max(n, -0x7F); // limit to allowed range
  const i32x16 t2 = min(t1, 0x80);
  const i32x16 t3 = t2 + 0x7F; // add bias
  const i32x16 t4 = t3 << 23; // put exponent into position 23
  return _mm512_castsi512_ps(t4); // reinterpret as f32
}
// static f32x16 exp2(f32x16 const x); // defined in
// vectormath_exp.h

inline f32x16 shuffle_up(f32x16 vec) {
  return _mm512_maskz_expand_ps(__mmask16{0xFFFE}, vec);
}
inline f32x16 shuffle_up(f32x16 vec, f32 first) {
  u32x16 idx_vec{16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  return _mm512_mask_permutex2var_ps(vec, __mmask16{0xFFFF}, idx_vec, f32x16::expand_undef(first));
}

inline f32x16 shuffle_down(f32x16 vec) {
  u32x16 idx_vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31};
  return _mm512_mask_permutex2var_ps(vec, __mmask16{0xFFFF}, idx_vec, _mm512_setzero_ps());
}
inline f32x16 shuffle_down(f32x16 vec, f32 last) {
  u32x16 idx_vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31};
  return _mm512_mask_permutex2var_ps(vec, __mmask16{0xFFFF}, idx_vec, _mm512_set1_ps(last));
}

template<bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8, bool b9,
         bool b10, bool b11, bool b12, bool b13, bool b14, bool b15>
inline f32x16 blend(f32x16 a, f32x16 b) {
  return _mm512_mask_blend_ps(
    __mmask16{unsigned{b0} | (unsigned{b1} << 1U) | (unsigned{b2} << 2U) | (unsigned{b3} << 3U) |
              (unsigned{b4} << 4U) | (unsigned{b5} << 5U) | (unsigned{b6}) << 6U |
              (unsigned{b7}) << 7U | (unsigned{b8} << 8U) | (unsigned{b9} << 9U) |
              (unsigned{b10}) << 10U | (unsigned{b11} << 11U) | (unsigned{b12} << 12U) |
              (unsigned{b13}) << 13U | (unsigned{b14} << 14U) | (unsigned{b15} << 15U)},
    b, a);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F32X16_HPP
