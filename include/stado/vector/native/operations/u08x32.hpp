#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U08X32_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U08X32_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask.hpp"
#include "stado/vector/native/operations/i08x32.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i08x32.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
// vector operator + : add
inline u8x32 operator+(const u8x32 a, const u8x32 b) {
  return {i8x32(a) + i8x32(b)};
}

// vector operator - : subtract
inline u8x32 operator-(const u8x32 a, const u8x32 b) {
  return {i8x32(a) - i8x32(b)};
}

// vector operator * : multiply
inline u8x32 operator*(const u8x32 a, const u8x32 b) {
  return {i8x32(a) * i8x32(b)};
}

// vector operator << : shift left all elements
inline u8x32 operator<<(const u8x32 a, u32 b) {
  const u32 mask = u32{0xFF} >> b; // mask to remove bits that are shifted out
  // remove bits that will overflow
  const __m256i am = _mm256_and_si256(a, _mm256_set1_epi8(i8(mask)));
  const __m256i res = _mm256_sll_epi16(am, _mm_cvtsi32_si128(i32(b))); // 16-bit shifts
  return res;
}

// vector operator << : shift left all elements
inline u8x32 operator<<(const u8x32 a, i32 b) {
  return a << u32(b);
}

// vector operator >> : shift right logical all elements
inline u8x32 operator>>(const u8x32 a, u32 b) {
  const u32 mask = u32{0xFF} << b; // mask to remove bits that are shifted out
  // remove bits that will overflow
  const __m256i am = _mm256_and_si256(a, _mm256_set1_epi8(i8(mask)));
  const __m256i res = _mm256_srl_epi16(am, _mm_cvtsi32_si128(i32(b))); // 16-bit shifts
  return res;
}

// vector operator >> : shift right logical all elements
inline u8x32 operator>>(const u8x32 a, i32 b) {
  return a >> u32(b);
}

// vector operator >>= : shift right artihmetic
inline u8x32& operator>>=(u8x32& a, u32 b) {
  a = a >> b;
  return a;
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline Mask<8, 32> operator>=(const u8x32 a, const u8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu8_mask(a, b, 5);
#else
  return _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a); // a == max(a,b)
#endif
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline Mask<8, 32> operator<=(const u8x32 a, const u8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu8_mask(a, b, 2);
#else
  return b >= a;
#endif
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline Mask<8, 32> operator>(const u8x32 a, const u8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu8_mask(a, b, 6);
#else
  return Mask<8, 32>(i8x32(~(b >= a)));
#endif
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline Mask<8, 32> operator<(const u8x32 a, const u8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_cmp_epu8_mask(a, b, 1);
#else
  return b > a;
#endif
}

// vector operator & : bitwise and
inline u8x32 operator&(const u8x32 a, const u8x32 b) {
  return {si256(a) & si256(b)};
}
inline u8x32 operator&&(const u8x32 a, const u8x32 b) {
  return a & b;
}

// vector operator | : bitwise or
inline u8x32 operator|(const u8x32 a, const u8x32 b) {
  return {si256(a) | si256(b)};
}
inline u8x32 operator||(const u8x32 a, const u8x32 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
inline u8x32 operator^(const u8x32 a, const u8x32 b) {
  return {si256(a) ^ si256(b)};
}

// vector operator ~ : bitwise not
inline u8x32 operator~(const u8x32 a) {
  return {~si256(a)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 32; i++) result[i] = s[i] ? a[i] : b[i];
inline u8x32 select(const Mask<8, 32> s, const u8x32 a, const u8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_mov_epi8(b, s, a);
#else
  return selectb(s, a, b);
#endif
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u8x32 if_add(const Mask<8, 32> f, const u8x32 a, const u8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_add_epi8(a, f, a, b);
#else
  return a + (u8x32(f) & b);
#endif
}

// Conditional subtract
inline u8x32 if_sub(const Mask<8, 32> f, const u8x32 a, const u8x32 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors
  return _mm256_mask_sub_epi8(a, f, a, b);
#else
  return a - (u8x32(f) & b);
#endif
}

// Conditional multiply
inline u8x32 if_mul(const Mask<8, 32> f, const u8x32 a, const u8x32 b) {
  return select(f, a * b, a);
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
// (Note: horizontal_add_x(u8x32) is slightly faster)
inline u8 horizontal_add(const u8x32 a) {
  const __m256i sum1 = _mm256_sad_epu8(a, _mm256_setzero_si256());
  const __m256i sum2 = _mm256_shuffle_epi32(sum1, 2);
  const __m256i sum3 = _mm256_add_epi16(sum1, sum2);
  const __m128i sum4 = _mm256_extracti128_si256(sum3, 1);
  const __m128i sum5 = _mm_add_epi16(_mm256_castsi256_si128(sum3), sum4);
  auto sum6 = u8(_mm_cvtsi128_si32(sum5)); // truncate to 8 bits
  return sum6; // zero extend to 32 bits
}

// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is zero-extended before addition to avoid overflow
inline u32 horizontal_add_x(const u8x32 a) {
  const __m256i sum1 = _mm256_sad_epu8(a, _mm256_setzero_si256());
  const __m256i sum2 = _mm256_shuffle_epi32(sum1, 2);
  const __m256i sum3 = _mm256_add_epi16(sum1, sum2);
  const __m128i sum4 = _mm256_extracti128_si256(sum3, 1);
  const __m128i sum5 = _mm_add_epi16(_mm256_castsi256_si128(sum3), sum4);
  return u32(_mm_cvtsi128_si32(sum5));
}

// function add_saturated: add element by element, unsigned with saturation
inline u8x32 add_saturated(const u8x32 a, const u8x32 b) {
  return _mm256_adds_epu8(a, b);
}

// function sub_saturated: subtract element by element, unsigned with saturation
inline u8x32 sub_saturated(const u8x32 a, const u8x32 b) {
  return _mm256_subs_epu8(a, b);
}

// function max: a > b ? a : b
inline u8x32 max(const u8x32 a, const u8x32 b) {
  return _mm256_max_epu8(a, b);
}

// function min: a < b ? a : b
inline u8x32 min(const u8x32 a, const u8x32 b) {
  return _mm256_min_epu8(a, b);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U08X32_HPP
