#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F32X04_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F32X04_HPP

#include <limits>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask.hpp"
#include "stado/vector/native/operations/i32x04.hpp"
#include "stado/vector/native/operations/u32x04.hpp"
#include "stado/vector/native/types/f32x04.hpp"
#include "stado/vector/native/types/i32x04.hpp"

namespace stado {
// vector operator + : add element by element
inline f32x4 operator+(const f32x4 a, const f32x4 b) {
  return _mm_add_ps(a, b);
}

// vector operator + : add vector and scalar
inline f32x4 operator+(const f32x4 a, f32 b) {
  return a + f32x4(b);
}
inline f32x4 operator+(f32 a, const f32x4 b) {
  return f32x4(a) + b;
}

// vector operator += : add
inline f32x4& operator+=(f32x4& a, const f32x4 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline f32x4 operator++(f32x4& a, int) {
  f32x4 a0 = a;
  a = a + 1.0F;
  return a0;
}

// prefix operator ++
inline f32x4& operator++(f32x4& a) {
  a = a + 1.0F;
  return a;
}

// vector operator - : subtract element by element
inline f32x4 operator-(const f32x4 a, const f32x4 b) {
  return _mm_sub_ps(a, b);
}

// vector operator - : subtract vector and scalar
inline f32x4 operator-(const f32x4 a, f32 b) {
  return a - f32x4(b);
}
inline f32x4 operator-(f32 a, const f32x4 b) {
  return f32x4(a) - b;
}

// vector operator - : unary minus
// Change sign bit, even for 0, INF and NAN
inline f32x4 operator-(const f32x4 a) {
  return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(std::numeric_limits<i32>::min())));
}

// vector operator -= : subtract
inline f32x4& operator-=(f32x4& a, const f32x4 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline f32x4 operator--(f32x4& a, int) {
  f32x4 a0 = a;
  a = a - 1.0F;
  return a0;
}

// prefix operator --
inline f32x4& operator--(f32x4& a) {
  a = a - 1.0F;
  return a;
}

// vector operator * : multiply element by element
inline f32x4 operator*(const f32x4 a, const f32x4 b) {
  return _mm_mul_ps(a, b);
}

// vector operator * : multiply vector and scalar
inline f32x4 operator*(const f32x4 a, f32 b) {
  return a * f32x4(b);
}
inline f32x4 operator*(f32 a, const f32x4 b) {
  return f32x4(a) * b;
}

// vector operator *= : multiply
inline f32x4& operator*=(f32x4& a, const f32x4 b) {
  a = a * b;
  return a;
}

// vector operator / : divide all elements by same integer
inline f32x4 operator/(const f32x4 a, const f32x4 b) {
  return _mm_div_ps(a, b);
}

// vector operator / : divide vector and scalar
inline f32x4 operator/(const f32x4 a, f32 b) {
  return a / f32x4(b);
}
inline f32x4 operator/(f32 a, const f32x4 b) {
  return f32x4(a) / b;
}

// vector operator /= : divide
inline f32x4& operator/=(f32x4& a, const f32x4 b) {
  a = a / b;
  return a;
}

// vector operator == : returns true for elements for which a == b
inline Mask<32, 4> operator==(const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_ps_mask(a, b, 0);
#else
  return _mm_cmpeq_ps(a, b);
#endif
}

// vector operator != : returns true for elements for which a != b
inline Mask<32, 4> operator!=(const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_ps_mask(a, b, 4);
#else
  return _mm_cmpneq_ps(a, b);
#endif
}

// vector operator < : returns true for elements for which a < b
inline Mask<32, 4> operator<(const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_ps_mask(a, b, 1);
#else
  return _mm_cmplt_ps(a, b);
#endif
}

// vector operator <= : returns true for elements for which a <= b
inline Mask<32, 4> operator<=(const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_ps_mask(a, b, 2);
#else
  return _mm_cmple_ps(a, b);
#endif
}

// vector operator > : returns true for elements for which a > b
inline Mask<32, 4> operator>(const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_ps_mask(a, b, 6);
#else
  return b < a;
#endif
}

// vector operator >= : returns true for elements for which a >= b
inline Mask<32, 4> operator>=(const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_cmp_ps_mask(a, b, 5);
#else
  return b <= a;
#endif
}

// Bitwise logical operators

// vector operator & : bitwise and
inline f32x4 operator&(const f32x4 a, const f32x4 b) {
  return _mm_and_ps(a, b);
}

// vector operator &= : bitwise and
inline f32x4& operator&=(f32x4& a, const f32x4 b) {
  a = a & b;
  return a;
}

// vector operator & : bitwise and of f32x4 and Mask<32, 4>
inline f32x4 operator&(const f32x4 a, const Mask<32, 4> b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_maskz_mov_ps(b, a);
#else
  return _mm_and_ps(a, b);
#endif
}
inline f32x4 operator&(const Mask<32, 4> a, const f32x4 b) {
  return b & a;
}

// vector operator | : bitwise or
inline f32x4 operator|(const f32x4 a, const f32x4 b) {
  return _mm_or_ps(a, b);
}

// vector operator |= : bitwise or
inline f32x4& operator|=(f32x4& a, const f32x4 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline f32x4 operator^(const f32x4 a, const f32x4 b) {
  return _mm_xor_ps(a, b);
}

// vector operator ^= : bitwise xor
inline f32x4& operator^=(f32x4& a, const f32x4 b) {
  a = a ^ b;
  return a;
}

// vector operator ! : logical not. Returns Boolean vector
inline Mask<32, 4> operator!(const f32x4 a) {
  return a == f32x4(0.0F);
}

// Select between two __m128 sources, element by element, with broad boolean vector.
// Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
// Each element in s must be either 0 (false) or 0xFFFFFFFF (true).
// No other values are allowed for broad boolean vectors.
// The implementation depends on the instruction set:
// If SSE4.1 is supported then only bit 31 in each dword of s is checked,
// otherwise all bits in s are used.
inline __m128 selectf(const __m128 s, const __m128 a, const __m128 b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_blendv_ps(b, a, s);
#else
  return _mm_or_ps(_mm_and_ps(s, a), _mm_andnot_ps(s, b));
#endif
}

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
inline f32x4 select(const Mask<32, 4> s, const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm_mask_mov_ps(b, s, a);
#else
  return selectf(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline f32x4 if_add(const Mask<32, 4> f, const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_mask_add_ps(a, f, a, b);
#else
  return a + (f32x4(f) & b);
#endif
}

// Conditional subtract: For all vector elements i: result[i] = f[i] ? (a[i] - b[i]) : a[i]
inline f32x4 if_sub(const Mask<32, 4> f, const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_mask_sub_ps(a, f, a, b);
#else
  return a - (f32x4(f) & b);
#endif
}

// Conditional multiply: For all vector elements i: result[i] = f[i] ? (a[i] * b[i]) : a[i]
inline f32x4 if_mul(const Mask<32, 4> f, const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_mask_mul_ps(a, f, a, b);
#else
  return a * select(f, b, 1.F);
#endif
}

// Conditional divide: For all vector elements i: result[i] = f[i] ? (a[i] / b[i]) : a[i]
inline f32x4 if_div(const Mask<32, 4> f, const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_mask_div_ps(a, f, a, b);
#else
  return a / select(f, b, 1.F);
#endif
}

// Sign functions

// Function sign_bit: gives true for elements that have the sign bit set
// even for -0.0F, -INF and -NAN
// Note that sign_bit(f32x4(-0.0F)) gives true, while f32x4(-0.0F)
// < f32x4(0.0f) gives false (the underscore in the name avoids a conflict with a
// macro in Intel's mathimf.h)
inline Mask<32, 4> sign_bit(const f32x4 a) {
  const i32x4 t1 = _mm_castps_si128(a); // reinterpret as 32-bit integer
  const i32x4 t2 = t1 >> 31; // extend sign bit
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return t2 != i32x4(0);
#else
  return _mm_castsi128_ps(t2); // reinterpret as 32-bit Boolean
#endif
}

// Function sign_combine: changes the sign of a when b has the sign bit set
// same as select(sign_bit(b), -a, a)
inline f32x4 sign_combine(const f32x4 a, const f32x4 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_castsi128_ps(_mm_ternarylogic_epi32(_mm_castps_si128(a), _mm_castps_si128(b),
                                                 (i32x4(std::numeric_limits<i32>::min())), 0x78));
#else
  return a ^ (b & f32x4(-0.0F));
#endif
}

// Categorization functions

// Function is_finite: gives true for elements that are normal, denormal or zero,
// false for INF and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
inline Mask<32, 4> is_finite(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return __mmask8(_mm_fpclass_ps_mask(a, 0x99) ^ 0x0FU);
#else
  i32x4 t1 = _mm_castps_si128(a); // reinterpret as 32-bit integer
  i32x4 t2 = t1 << 1; // shift out sign bit
  i32x4 t3 = i32x4(t2 & i32(0xFF000000)) != i32x4(i32(0xFF000000)); // exponent field is not all 1s
  return {t3};
#endif
}

// Function is_inf: gives true for elements that are +INF or -INF
// false for finite numbers and NAN
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
inline Mask<32, 4> is_inf(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_fpclass_ps_mask(a, 0x18);
#else
  i32x4 t1 = _mm_castps_si128(a); // reinterpret as 32-bit integer
  i32x4 t2 = t1 << 1; // shift out sign bit
  return t2 == i32x4(i32(0xFF000000)); // exponent is all 1s, fraction is 0
#endif
}

// Function is_nan: gives true for elements that are +NAN or -NAN
// false for finite numbers and +/-INF
// (the underscore in the name avoids a conflict with a macro in Intel's mathimf.h)
inline Mask<32, 4> is_nan(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  // assume that compiler does not optimize this away with -ffinite-math-only:
  return {_mm_fpclass_ps_mask(a, 0x81)};
#elif STADO_INSTRUCTION_SET >= STADO_AVX
  return _mm_cmp_ps(a, a, 3); // compare unordered
#else
  // This is not safe with -ffinite-math-only, -ffast-math, or /fp:fast compiler option
  return a != a;
#endif
}

// Function is_subnormal: gives true for elements that are denormal (subnormal)
// false for finite numbers, zero, NAN and INF
inline Mask<32, 4> is_subnormal(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return {_mm_fpclass_ps_mask(a, 0x20)};
#else
  i32x4 t1 = _mm_castps_si128(a); // reinterpret as 32-bit integer
  i32x4 t2 = t1 << 1; // shift out sign bit
  i32x4 t3 = i32(0xFF000000); // exponent mask
  i32x4 t4 = t2 & t3; // exponent
  i32x4 t5 = _mm_andnot_si128(t3, t2); // fraction
  return Mask<32, 4>((t4 == i32x4(0)) & (t5 != i32x4(0))); // exponent = 0 and fraction != 0
#endif
}

// Function is_zero_or_subnormal: gives true for elements that are zero or subnormal (denormal)
// false for finite numbers, NAN and INF
inline Mask<32, 4> is_zero_or_subnormal(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return {_mm_fpclass_ps_mask(a, 0x26)};
#else
  i32x4 t = _mm_castps_si128(a); // reinterpret as 32-bit integer
  t &= 0x7F800000; // isolate exponent
  return t == i32x4(0); // exponent = 0
#endif
}

// Function infinite4f: returns a vector where all elements are +INF
inline f32x4 infinite4f() {
  return _mm_castsi128_ps(_mm_set1_epi32(0x7F800000));
}

// Function nan4f: returns a vector where all elements are NAN (quiet)
inline f32x4 nan4f(u32 n = 0x10) {
  return nan_vec<f32x4>(n);
}

// General arithmetic functions, etc.

// Horizontal add: Calculates the sum of all vector elements.
inline f32 horizontal_add(const f32x4 a) {
  // The hadd instruction is inefficient, and may be split into two instructions for faster decoding
  const __m128 t1 = _mm_movehl_ps(a, a);
  const __m128 t2 = _mm_add_ps(a, t1);
  const __m128 t3 = _mm_shuffle_ps(t2, t2, 1);
  const __m128 t4 = _mm_add_ss(t2, t3);
  return _mm_cvtss_f32(t4);
}

// function max: a > b ? a : b
inline f32x4 max(const f32x4 a, const f32x4 b) {
  return _mm_max_ps(a, b);
}

// function min: a < b ? a : b
inline f32x4 min(const f32x4 a, const f32x4 b) {
  return _mm_min_ps(a, b);
}
// NAN-safe versions of maximum and minimum are in vector-convert.hpp

// function abs: absolute value
inline f32x4 abs(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm_range_ps(a, a, 8);
#else
  __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  return _mm_and_ps(a, mask);
#endif
}

// function sqrt: square root
inline f32x4 sqrt(const f32x4 a) {
  return _mm_sqrt_ps(a);
}

// function square: a * a
inline f32x4 square(const f32x4 a) {
  return a * a;
}

// pow(vector, int) function template
template<typename TVec>
inline TVec pow_template_i(const TVec x0, i32 n) {
  TVec x = x0; // a^(2^i)
  TVec y(1); // accumulator
  if (n >= 0) { // make sure n is not negative
    while (true) { // loop for each bit in n
      if (n & 1) {
        y *= x; // multiply if bit = 1
      }
      n >>= 1; // get next bit of n
      if (n == 0) {
        return y; // finished
      }
      x *= x; // x = a^2, a^4, a^8, etc.
    }
  } else {
    // n < 0
    if (u32(n) == 0x80000000U) {
      return nan_vec<TVec>(); // integer overflow
    }
    return TVec(1) / pow_template_i<TVec>(x0, -n); // reciprocal
  }
}

// The purpose of this template is to prevent implicit conversion of a f32
// exponent to int when calling pow(vector, f32) and vectormath_exp.h is not included
template<typename TT>
static f32x4 pow(f32x4 x0, TT n); // = delete

// Raise floating point numbers to integer power n
template<>
inline f32x4 pow<i32>(const f32x4 x0, const i32 n) {
  return pow_template_i<f32x4>(x0, n);
}

// allow conversion from unsigned int
template<>
inline f32x4 pow<u32>(const f32x4 x0, const u32 n) {
  return pow_template_i<f32x4>(x0, i32(n));
}

// Raise floating point numbers to integer power n, where n is a compile-time constant

// gcc can optimize pow_template_i to generate the same as the code below. MS and Clang can not.
// Therefore, this code is kept
// to do: test on Intel compiler
template<typename TVec, i32 tN>
inline TVec pow_n(const TVec a) {
  if (tN == 0x80000000) {
    return nan_vec<TVec>(); // integer overflow
  }
  if (tN < 0) {
    return TVec(1.0F) / pow_n<TVec, -tN>(a);
  }
  if (tN == 0) {
    return TVec(1.0F);
  }
  if (tN >= 256) {
    return pow(a, tN);
  }
  TVec x = a; // a^(2^i)
  TVec y; // accumulator
  const i32 lowest = tN - (tN & (tN - 1)); // lowest set bit in n
  if (tN & 1) {
    y = x;
  }
  if (tN < 2) {
    return y;
  }
  x = x * x; // x^2
  if (tN & 2) {
    if (lowest == 2) {
      y = x;
    } else {
      y *= x;
    }
  }
  if (tN < 4) {
    return y;
  }
  x = x * x; // x^4
  if (tN & 4) {
    if (lowest == 4) {
      y = x;
    } else {
      y *= x;
    }
  }
  if (tN < 8) {
    return y;
  }
  x = x * x; // x^8
  if (tN & 8) {
    if (lowest == 8) {
      y = x;
    } else {
      y *= x;
    }
  }
  if (tN < 16) {
    return y;
  }
  x = x * x; // x^16
  if (tN & 16) {
    if (lowest == 16) {
      y = x;
    } else {
      y *= x;
    }
  }
  if (tN < 32) {
    return y;
  }
  x = x * x; // x^32
  if (tN & 32) {
    if (lowest == 32) {
      y = x;
    } else {
      y *= x;
    }
  }
  if (tN < 64) {
    return y;
  }
  x = x * x; // x^64
  if (tN & 64) {
    if (lowest == 64) {
      y = x;
    } else {
      y *= x;
    }
  }
  if (tN < 128) {
    return y;
  }
  x = x * x; // x^128
  if (tN & 128) {
    if (lowest == 128) {
      y = x;
    } else {
      y *= x;
    }
  }
  return y;
}

// implement as function pow(vector, const_int)
template<i32 tN>
inline f32x4 pow(const f32x4 a, ConstInt<tN> /*n*/) {
  return pow_n<f32x4, tN>(a);
}

inline f32x4 round(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_round_ps(a, 8);
#else // SSE2
  i32x4 y1 = _mm_cvtps_epi32(a); // convert to integer
  f32x4 y2 = _mm_cvtepi32_ps(y1); // convert back to f32
#ifdef SIGNED_ZERO
  y2 |= (a & f32x4(-0.0F)); // sign of zero
#endif
  return select(y1 != i32x4(i32(0x80000000)), y2, a); // use original value if integer overflows
#endif
}

// function truncate: round towards zero. (result as f32 vector)
inline f32x4 truncate(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_round_ps(a, 3 + 8);
#else // SSE2
  i32x4 y1 = _mm_cvttps_epi32(a); // truncate to integer
  f32x4 y2 = _mm_cvtepi32_ps(y1); // convert back to f32
#ifdef SIGNED_ZERO
  y2 |= (a & f32x4(-0.0F)); // sign of zero
#endif
  return select(y1 != i32x4(i32(0x80000000)), y2, a); // use original value if integer overflows
#endif
}

// function floor: round towards minus infinity. (result as f32 vector)
inline f32x4 floor(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_round_ps(a, 1 + 8);
#else // SSE2
  f32x4 y = round(a); // round
  y -= f32x4(1.F) & (y > a); // subtract 1 if bigger
#ifdef SIGNED_ZERO
  y |= (a & f32x4(-0.0F)); // sign of zero
#endif
  return y;
#endif
}

// function ceil: round towards plus infinity. (result as f32 vector)
inline f32x4 ceil(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_round_ps(a, 2 + 8);
#else // SSE2
  f32x4 y = round(a); // round
  y += f32x4(1.F) & (y < a); // add 1 if bigger
#ifdef SIGNED_ZERO
  y |= (a & f32x4(-0.0F)); // sign of zero
#endif
  return y;
#endif
}

// function roundi: round to nearest integer (even). (result as integer vector)
inline i32x4 roundi(const f32x4 a) {
  // Note: assume MXCSR control register is set to rounding
  return _mm_cvtps_epi32(a);
}

// function truncatei: round towards zero. (result as integer vector)
inline i32x4 truncatei(const f32x4 a) {
  return _mm_cvttps_epi32(a);
}

// function to_f32: convert integer vector to f32 vector
inline f32x4 to_f32(const i32x4 a) {
  return _mm_cvtepi32_ps(a);
}

// function to_f32: convert unsigned integer vector to f32 vector
inline f32x4 to_f32(const u32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && (!defined(_MSC_VER) || defined(__INTEL_COMPILER))
  // _mm_cvtepu32_ps missing in MS VS2019
  return _mm_cvtepu32_ps(a);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  return _mm512_castps512_ps128(_mm512_cvtepu32_ps(_mm512_castsi128_si512(a)));
#else
  f32x4 b = to_f32(i32x4(a & 0xFFFFF)); // 20 bits
  f32x4 c = to_f32(i32x4(a >> 20)); // remaining bits
  f32x4 d = b + c * 1048576.F; // 2^20
  return d;
#endif
}

// Approximate math functions

// approximate reciprocal (Faster than 1.F / a. relative accuracy better than 2^-11)
inline f32x4 approx_recipr(const f32x4 a) {
#ifdef __AVX512ER__ // AVX512ER: full precision
  // todo: if future processors have both AVX512ER and AVX512VL:
  // _mm128_rcp28_round_ps(a, _MM_FROUND_NO_EXC);
  return _mm512_castps512_ps128(
    _mm512_rcp28_round_ps(_mm512_castps128_ps512(a), _MM_FROUND_NO_EXC));
#elif STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL: 14 bit precision
  return _mm_rcp14_ps(a);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F // AVX512F: 14 bit precision
  return _mm512_castps512_ps128(_mm512_rcp14_ps(_mm512_castps128_ps512(a)));
#else // AVX: 11 bit precision
  return _mm_rcp_ps(a);
#endif
}

// approximate reciprocal squareroot (Faster than 1.F / sqrt(a). Relative accuracy better than
// 2^-11)
inline f32x4 approx_rsqrt(const f32x4 a) {
  // use more accurate version if available. (none of these will raise exceptions on zero)
#ifdef __AVX512ER__ // AVX512ER: full precision
  // todo: if future processors have both AVX512ER and AVX521VL:
  // _mm128_rsqrt28_round_ps(a, _MM_FROUND_NO_EXC);
  return _mm512_castps512_ps128(
    _mm512_rsqrt28_round_ps(_mm512_castps128_ps512(a), _MM_FROUND_NO_EXC));
#elif STADO_INSTRUCTION_SET >= STADO_AVX512SKL && !defined(_MSC_VER) // missing in VS2019
  return _mm_rsqrt14_ps(a);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F // AVX512F: 14 bit precision
  return _mm512_castps512_ps128(_mm512_rsqrt14_ps(_mm512_castps128_ps512(a)));
#else // SSE: 11 bit precision
  return _mm_rsqrt_ps(a);
#endif
}

// Fused multiply and add functions

// Multiply and add
inline f32x4 mul_add(const f32x4 a, const f32x4 b, const f32x4 c) {
#ifdef __FMA__
  return _mm_fmadd_ps(a, b, c);
#elif defined(__FMA4__)
  return _mm_macc_ps(a, b, c);
#else
  return a * b + c;
#endif
}

// Multiply and subtract
inline f32x4 mul_sub(const f32x4 a, const f32x4 b, const f32x4 c) {
#ifdef __FMA__
  return _mm_fmsub_ps(a, b, c);
#elif defined(__FMA4__)
  return _mm_msub_ps(a, b, c);
#else
  return a * b - c;
#endif
}

// Multiply and inverse subtract
inline f32x4 nmul_add(const f32x4 a, const f32x4 b, const f32x4 c) {
#ifdef __FMA__
  return _mm_fnmadd_ps(a, b, c);
#elif defined(__FMA4__)
  return _mm_nmacc_ps(a, b, c);
#else
  return c - a * b;
#endif
}

// Multiply and subtract with extra precision on the intermediate calculations,
// even if FMA instructions not supported, using Veltkamp-Dekker split.
// This is used in mathematical functions. Do not use it in general code
// because it is inaccurate in certain cases
inline f32x4 mul_sub_x(const f32x4 a, const f32x4 b, const f32x4 c) {
#ifdef __FMA__
  return _mm_fmsub_ps(a, b, c);
#elif defined(__FMA4__)
  return _mm_msub_ps(a, b, c);
#else
  // calculate a * b - c with extra precision
  i32x4 upper_mask = -(1 << 12); // mask to remove lower 12 bits
  f32x4 a_high = a & f32x4(_mm_castsi128_ps(upper_mask)); // split into high and low parts
  f32x4 b_high = b & f32x4(_mm_castsi128_ps(upper_mask));
  f32x4 a_low = a - a_high;
  f32x4 b_low = b - b_high;
  f32x4 r1 = a_high * b_high; // this product is exact
  f32x4 r2 = r1 - c; // subtract c from high product
  f32x4 r3 = r2 + (a_high * b_low + b_high * a_low) + a_low * b_low; // add rest of product
  return r3; // + ((r2 - r1) + c);
#endif
}

// Math functions using fast bit manipulation

// Extract the exponent as an integer
// exponent(a) = floor(log2(abs(a)));
// exponent(1.0f) = 0, exponent(0.0f) = -127, exponent(INF) = +128, exponent(NAN) = +128
inline i32x4 exponent(const f32x4 a) {
  const u32x4 t1 = _mm_castps_si128(a); // reinterpret as 32-bit integer
  const u32x4 t2 = t1 << 1; // shift out sign bit
  const u32x4 t3 = t2 >> 24; // shift down logical to position 0
  using SVec = i32x4;
  const SVec t4 = SVec(t3) - 0x7F; // subtract bias from exponent
  return t4;
}

// Extract the fraction part of a floating point number
// a = 2^exponent(a) * fraction(a), except for a = 0
// fraction(1.0f) = 1.0f, fraction(5.0f) = 1.25f
// NOTE: The name fraction clashes with an ENUM in MAC XCode CarbonCore script.h !
inline f32x4 fraction(const f32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm_getmant_ps(a, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);
#else
  u32x4 t1 = _mm_castps_si128(a); // reinterpret as 32-bit integer
  u32x4 t2 = (t1 & 0x007FFFFF) | 0x3F800000; // set exponent to 0 + bias
  return _mm_castsi128_ps(t2);
#endif
}

// Fast calculation of pow(2,n) with n integer
// n  =    0 gives 1.0f
// n >=  128 gives +INF
// n <= -127 gives 0.0f
// This function will never produce denormals, and never raise exceptions
inline f32x4 exp2(const i32x4 n) {
  const i32x4 t1 = max(n, -0x7F); // limit to allowed range
  const i32x4 t2 = min(t1, 0x80);
  const i32x4 t3 = t2 + 0x7F; // add bias
  const i32x4 t4 = t3 << 23; // put exponent into position 23
  return _mm_castsi128_ps(t4); // reinterpret as f32
}
// static f32x4 exp2(const f32x4 x);    // defined in
// vectormath_exp.h

// Control word manipulaton
// ------------------------
// The MXCSR control word has the following bits:
//  0:    Invalid Operation Flag
//  1:    Denormal Flag (=subnormal)
//  2:    Divide-by-Zero Flag
//  3:    Overflow Flag
//  4:    Underflow Flag
//  5:    Precision Flag
//  6:    Denormals Are Zeros (=subnormals)
//  7:    Invalid Operation Mask
//  8:    Denormal Operation Mask (=subnormal)
//  9:    Divide-by-Zero Mask
// 10:    Overflow Mask
// 11:    Underflow Mask
// 12:    Precision Mask
// 13-14: Rounding control
//        00: round to nearest or even
//        01: round down towards -infinity
//        10: round up   towards +infinity
//        11: round towards zero (truncate)
// 15: Flush to Zero

// Function get_control_word:
// Read the MXCSR control word
inline u32 get_control_word() {
  return _mm_getcsr();
}

// Function set_control_word:
// Write the MXCSR control word
inline void set_control_word(u32 w) {
  _mm_setcsr(w);
}

// Function no_subnormals:
// Set "Denormals Are Zeros" and "Flush to Zero" mode to avoid the extremely
// time-consuming denormals in case of underflow
inline void no_subnormals() {
  u32 t1 = get_control_word();
  t1 |= (1U << 6U) | (1U << 15U); // set bit 6 and 15 in MXCSR
  set_control_word(t1);
}

// Function reset_control_word:
// Set the MXCSR control word to the default value 0x1F80.
// This will mask floating point exceptions, set rounding mode to nearest (or even),
// and allow denormals.
inline void reset_control_word() {
  set_control_word(0x1F80);
}

// change signs on vectors f32x4
// Each index i0 - i3 is 1 for changing sign on the corresponding element, 0 for no change
template<int i0, int i1, int i2, int i3>
inline f32x4 change_sign(const f32x4 a) {
  if ((i0 | i1 | i2 | i3) == 0) {
    return a;
  }
  __m128i mask = _mm_setr_epi32((i0 ? 0x80000000 : 0), (i1 ? 0x80000000 : 0), (i2 ? 0x80000000 : 0),
                                (i3 ? 0x80000000 : 0));
  return _mm_xor_ps(a, _mm_castsi128_ps(mask)); // flip sign bits
}

inline f32x4 shuffle_up(f32x4 vec) {
  return _mm_castsi128_ps(_mm_bslli_si128(_mm_castps_si128(vec), 4));
}
inline f32x4 shuffle_up(f32x4 vec, f32 first) {
  return _mm_shuffle_ps(_mm_movelh_ps(f32x4::expand_undef(first), vec), vec, 0x98);
}

inline f32x4 shuffle_down(f32x4 vec) {
  return _mm_castsi128_ps(_mm_bsrli_si128(_mm_castps_si128(vec), 4));
}
inline f32x4 shuffle_down(f32x4 vec, f32 last) {
  return _mm_shuffle_ps(vec, _mm_shuffle_ps(f32x4::expand_undef(last), vec, 0xF4), 0x29);
}

template<bool b0, bool b1, bool b2, bool b3>
inline f32x4 blend(f32x4 a, f32x4 b) {
  return _mm_blend_ps(
    b, a, unsigned{b0} | (unsigned{b1} << 1U) | (unsigned{b2} << 2U) | (unsigned{b3} << 3U));
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_F32X04_HPP
