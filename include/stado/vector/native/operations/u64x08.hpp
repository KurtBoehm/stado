#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U64X08_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U64X08_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/08.hpp"
#include "stado/vector/native/operations/i64x08.hpp"
#include "stado/vector/native/types/i64x08.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
// vector operator + : add
inline u64x8 operator+(const u64x8 a, const u64x8 b) {
  return {i64x8(a) + i64x8(b)};
}

// vector operator - : subtract
inline u64x8 operator-(const u64x8 a, const u64x8 b) {
  return {i64x8(a) - i64x8(b)};
}

// vector operator * : multiply element by element
inline u64x8 operator*(const u64x8 a, const u64x8 b) {
  return {i64x8(a) * i64x8(b)};
}

// vector operator >> : shift right logical all elements
inline u64x8 operator>>(const u64x8 a, u32 b) {
  return _mm512_srl_epi64(a, _mm_cvtsi32_si128(i32(b)));
}
inline u64x8 operator>>(const u64x8 a, i32 b) {
  return a >> u32(b);
}
// vector operator >>= : shift right artihmetic
inline u64x8& operator>>=(u64x8& a, u32 b) {
  a = a >> b;
  return a;
}
// vector operator >>= : shift right logical
inline u64x8& operator>>=(u64x8& a, i32 b) {
  a = a >> u32(b);
  return a;
}

// vector operator << : shift left all elements
inline u64x8 operator<<(const u64x8 a, u32 b) {
  return {i64x8(a) << i32(b)};
}
// vector operator << : shift left all elements
inline u64x8 operator<<(const u64x8 a, i32 b) {
  return {i64x8(a) << b};
}

// vector operator < : returns true for elements for which a < b (unsigned)
inline CompactMask<8> operator<(const u64x8 a, const u64x8 b) {
  return _mm512_cmp_epu64_mask(a, b, 1);
}

// vector operator > : returns true for elements for which a > b (unsigned)
inline CompactMask<8> operator>(const u64x8 a, const u64x8 b) {
  return _mm512_cmp_epu64_mask(a, b, 6);
}

// vector operator >= : returns true for elements for which a >= b (unsigned)
inline CompactMask<8> operator>=(const u64x8 a, const u64x8 b) {
  return _mm512_cmp_epu64_mask(a, b, 5);
}

// vector operator <= : returns true for elements for which a <= b (unsigned)
inline CompactMask<8> operator<=(const u64x8 a, const u64x8 b) {
  return _mm512_cmp_epu64_mask(a, b, 2);
}

// vector operator & : bitwise and
inline u64x8 operator&(const u64x8 a, const u64x8 b) {
  return {i64x8(a) & i64x8(b)};
}

// vector operator | : bitwise or
inline u64x8 operator|(const u64x8 a, const u64x8 b) {
  return {i64x8(a) | i64x8(b)};
}

// vector operator ^ : bitwise xor
inline u64x8 operator^(const u64x8 a, const u64x8 b) {
  return {i64x8(a) ^ i64x8(b)};
}

// Functions for this class

// Select between two operands. Corresponds to this pseudocode:
// for (int i = 0; i < 4; i++) result[i] = s[i] ? a[i] : b[i];
inline u64x8 select(const CompactMask<8> s, const u64x8 a, const u64x8 b) {
  return {select(s, i64x8(a), i64x8(b))};
}

// Conditional add: For all vector elements i: result[i] = f[i] ? (a[i] + b[i]) : a[i]
inline u64x8 if_add(const CompactMask<8> f, const u64x8 a, const u64x8 b) {
  return _mm512_mask_add_epi64(a, u8(f), a, b);
}

// Conditional subtract
inline u64x8 if_sub(const CompactMask<8> f, const u64x8 a, const u64x8 b) {
  return _mm512_mask_sub_epi64(a, u8(f), a, b);
}

// Conditional multiply
inline u64x8 if_mul(const CompactMask<8> f, const u64x8 a, const u64x8 b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm512_mask_mullo_epi64(a, f, a, b); // AVX512DQ
#else
  return select(f, a * b, a);
#endif
}

// Horizontal add: Calculates the sum of all vector elements. Overflow will wrap around
inline u64 horizontal_add(const u64x8 a) {
  return u64(horizontal_add(i64x8(a)));
}

// function max: a > b ? a : b
inline u64x8 max(const u64x8 a, const u64x8 b) {
  return _mm512_max_epu64(a, b);
}

// function min: a < b ? a : b
inline u64x8 min(const u64x8 a, const u64x8 b) {
  return _mm512_min_epu64(a, b);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_U64X08_HPP
