#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I08X32_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I08X32_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i08x32.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
// vector operator + : add element by element
inline i8x32 operator+(const i8x32 a, const i8x32 b) {
  return _mm256_add_epi8(a, b);
}
// vector operator += : add
inline i8x32& operator+=(i8x32& a, const i8x32 b) {
  a = a + b;
  return a;
}

// postfix operator ++
inline i8x32 operator++(i8x32& a, int) {
  i8x32 a0 = a;
  a = a + 1;
  return a0;
}
// prefix operator ++
inline i8x32& operator++(i8x32& a) {
  a = a + 1;
  return a;
}

// vector operator - : subtract element by element
inline i8x32 operator-(const i8x32 a, const i8x32 b) {
  return _mm256_sub_epi8(a, b);
}
// vector operator - : unary minus
inline i8x32 operator-(const i8x32 a) {
  return _mm256_sub_epi8(_mm256_setzero_si256(), a);
}
// vector operator -= : add
inline i8x32& operator-=(i8x32& a, const i8x32 b) {
  a = a - b;
  return a;
}

// postfix operator --
inline i8x32 operator--(i8x32& a, int) {
  i8x32 a0 = a;
  a = a - 1;
  return a0;
}
// prefix operator --
inline i8x32& operator--(i8x32& a) {
  a = a - 1;
  return a;
}

// vector operator * : multiply element by element
inline i8x32 operator*(const i8x32 a, const i8x32 b) {
  // There is no 8-bit multiply in AVX2. Split into two 16-bit multiplications
  const __m256i aodd = _mm256_srli_epi16(a, 8); // odd numbered elements of a
  const __m256i bodd = _mm256_srli_epi16(b, 8); // odd numbered elements of b
  const __m256i muleven = _mm256_mullo_epi16(a, b); // product of even numbered elements
  __m256i mulodd = _mm256_mullo_epi16(aodd, bodd); // product of odd  numbered elements
  mulodd = _mm256_slli_epi16(mulodd, 8); // put odd numbered elements back in place
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
  return _mm256_mask_mov_epi8(mulodd, 0x55555555, muleven);
#else
  __m256i mask = _mm256_set1_epi32(0x00FF00FF); // mask for even positions
  __m256i product = selectb(mask, muleven, mulodd); // interleave even and odd
  return product;
#endif
}

// vector operator *= : multiply
inline i8x32& operator*=(i8x32& a, const i8x32 b) {
  a = a * b;
  return a;
}

// vector operator << : shift left all elements
inline i8x32 operator<<(const i8x32 a, i32 b) {
  const u32 mask = u32{0xFF} >> u32(b); // mask to remove bits that are shifted out
  // remove bits that will overflow
  const __m256i am = _mm256_and_si256(a, _mm256_set1_epi8(i8(mask)));
  const __m256i res = _mm256_sll_epi16(am, _mm_cvtsi32_si128(b)); // 16-bit shifts
  return res;
}

// vector operator <<= : shift left
inline i8x32& operator<<=(i8x32& a, i32 b) {
  a = a << b;
  return a;
}

// vector operator >> : shift right arithmetic all elements
inline i8x32 operator>>(const i8x32 a, i32 b) {
  __m256i aeven = _mm256_slli_epi16(a, 8); // even numbered elements of a. get sign bit in position
  aeven = _mm256_sra_epi16(aeven, _mm_cvtsi32_si128(b + 8)); // shift arithmetic, back to position
  const __m256i aodd =
    _mm256_sra_epi16(a, _mm_cvtsi32_si128(b)); // shift odd numbered elements arithmetic
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
  return _mm256_mask_mov_epi8(aodd, 0x55555555, aeven);
#else
  __m256i mask = _mm256_set1_epi32(0x00FF00FF); // mask for even positions
  __m256i res = selectb(mask, aeven, aodd); // interleave even and odd
  return res;
#endif
}

// vector operator >>= : shift right artihmetic
inline i8x32& operator>>=(i8x32& a, i32 b) {
  a = a >> b;
  return a;
}

// vector operator == : returns true for elements for which a == b
inline Mask<8, 32> operator==(const i8x32 a, const i8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi8_mask(a, b, 0);
#else
  return _mm256_cmpeq_epi8(a, b);
#endif
}

// vector operator != : returns true for elements for which a != b
inline Mask<8, 32> operator!=(const i8x32 a, const i8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi8_mask(a, b, 4);
#else
  return Mask<8, 32>(i8x32(~(a == b)));
#endif
}

// vector operator > : returns true for elements for which a > b (signed)
inline Mask<8, 32> operator>(const i8x32 a, const i8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi8_mask(a, b, 6);
#else
  return _mm256_cmpgt_epi8(a, b);
#endif
}

// vector operator < : returns true for elements for which a < b (signed)
inline Mask<8, 32> operator<(const i8x32 a, const i8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi8_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator >= : returns true for elements for which a >= b (signed)
inline Mask<8, 32> operator>=(const i8x32 a, const i8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi8_mask(a, b, 5);
#else
  return Mask<8, 32>(i8x32(~(b > a)));
#endif
}

// vector operator <= : returns true for elements for which a <= b (signed)
inline Mask<8, 32> operator<=(const i8x32 a, const i8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi8_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator & : bitwise and
inline i8x32 operator&(const i8x32 a, const i8x32 b) {
  return i8x32(si256(a) & si256(b));
}
inline i8x32 operator&&(const i8x32 a, const i8x32 b) {
  return a & b;
}
// vector operator &= : bitwise and
inline i8x32& operator&=(i8x32& a, const i8x32 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
inline i8x32 operator|(const i8x32 a, const i8x32 b) {
  return i8x32(si256(a) | si256(b));
}
inline i8x32 operator||(const i8x32 a, const i8x32 b) {
  return a | b;
}
// vector operator |= : bitwise or
inline i8x32& operator|=(i8x32& a, const i8x32 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
inline i8x32 operator^(const i8x32 a, const i8x32 b) {
  return i8x32(si256(a) ^ si256(b));
}
// vector operator ^= : bitwise xor
inline i8x32& operator^=(i8x32& a, const i8x32 b) {
  a = a ^ b;
  return a;
}

// vector operator ~ : bitwise not
inline i8x32 operator~(const i8x32 a) {
  return i8x32(~si256(a));
}

// vector operator ! : logical not, returns true for elements == 0
inline Mask<8, 32> operator!(const i8x32 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epi8_mask(a, _mm256_setzero_si256(), 0);
#else
  return _mm256_cmpeq_epi8(a, _mm256_setzero_si256());
#endif
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
inline i8x32 select(const Mask<8, 32> s, const i8x32 a, const i8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mov_epi8(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline i8x32 if_add(const Mask<8, 32> f, const i8x32 a, const i8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_add_epi8(a, f, a, b);
#else
  return a + (i8x32(f) & b);
#endif
}

// Conditional subtract
inline i8x32 if_sub(const Mask<8, 32> f, const i8x32 a, const i8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_sub_epi8(a, f, a, b);
#else
  return a - (i8x32(f) & b);
#endif
}

// Conditional multiply
inline i8x32 if_mul(const Mask<8, 32> f, const i8x32 a, const i8x32 b) {
  return select(f, a * b, a);
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline i8 horizontal_add(const i8x32 a) {
  const __m256i sum1 = _mm256_sad_epu8(a, _mm256_setzero_si256());
  const __m256i sum2 = _mm256_shuffle_epi32(sum1, 2);
  const __m256i sum3 = _mm256_add_epi16(sum1, sum2);
  const __m128i sum4 = _mm256_extracti128_si256(sum3, 1);
  const __m128i sum5 = _mm_add_epi16(_mm256_castsi256_si128(sum3), sum4);
  auto sum6 = i8(_mm_cvtsi128_si32(sum5)); // truncate to 8 bits
  return sum6; // sign extend to 32 bits
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is sign-extended before addition to avoid overflow
inline i32 horizontal_add_x(const i8x32 a) {
  __m256i aeven = _mm256_slli_epi16(a, 8); // even numbered elements of a. get sign bit in position
  aeven = _mm256_srai_epi16(aeven, 8); // sign extend even numbered elements
  const __m256i aodd = _mm256_srai_epi16(a, 8); // sign extend odd  numbered elements
  const __m256i sum1 = _mm256_add_epi16(aeven, aodd); // add even and odd elements
  const __m128i sum2 =
    _mm_add_epi16(_mm256_extracti128_si256(sum1, 1), _mm256_castsi256_si128(sum1));
  // The hadd instruction is inefficient, and may be split into two instructions for faster decoding
  const __m128i sum3 = _mm_add_epi16(sum2, _mm_unpackhi_epi64(sum2, sum2));
  const __m128i sum4 = _mm_add_epi16(sum3, _mm_shuffle_epi32(sum3, 1));
  const __m128i sum5 = _mm_add_epi16(sum4, _mm_shufflelo_epi16(sum4, 1));
  auto sum6 = i16(_mm_cvtsi128_si32(sum5)); // 16 bit sum
  return sum6; // sign extend to 32 bits
}

// function add_saturated: add element by element, signed with saturation
inline i8x32 add_saturated(const i8x32 a, const i8x32 b) {
  return _mm256_adds_epi8(a, b);
}

// function sub_saturated: subtract element by element, signed with saturation
inline i8x32 sub_saturated(const i8x32 a, const i8x32 b) {
  return _mm256_subs_epi8(a, b);
}

// function max: a > b ? a : b
inline i8x32 max(const i8x32 a, const i8x32 b) {
  return _mm256_max_epi8(a, b);
}

// function min: a < b ? a : b
inline i8x32 min(const i8x32 a, const i8x32 b) {
  return _mm256_min_epi8(a, b);
}

// function abs: a >= 0 ? a : -a
inline i8x32 abs(const i8x32 a) {
  return _mm256_abs_epi8(a);
}

// function abs_saturated: same as abs, saturate if overflow
inline i8x32 abs_saturated(const i8x32 a) {
  const __m256i absa = abs(a); // abs(a)
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_min_epu8(absa, i8x32(0x7F));
#else
  __m256i overfl = _mm256_cmpgt_epi8(_mm256_setzero_si256(), absa); // 0 > a
  return _mm256_add_epi8(absa, overfl); // subtract 1 if 0x80
#endif
}

// function rotate_left all elements
// Use negative count to rotate right
inline i8x32 rotate_left(const i8x32 a, i32 b) {
  const u8 mask = 0xFF << b; // mask off overflow bits
  const __m256i m = _mm256_set1_epi8(i8(mask));
  const __m128i bb = _mm_cvtsi32_si128(b & 7); // b modulo 8
  const __m128i mbb = _mm_cvtsi32_si128((-b) & 7); // 8-b modulo 8
  __m256i left = _mm256_sll_epi16(a, bb); // a << b
  __m256i right = _mm256_srl_epi16(a, mbb); // a >> 8-b
  left = _mm256_and_si256(m, left); // mask off overflow bits
  right = _mm256_andnot_si256(m, right);
  return _mm256_or_si256(left, right); // combine left and right shifted bits
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_I08X32_HPP
