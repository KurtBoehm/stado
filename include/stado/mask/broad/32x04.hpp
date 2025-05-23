#ifndef INCLUDE_STADO_MASK_BROAD_32X04_HPP
#define INCLUDE_STADO_MASK_BROAD_32X04_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad/base.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i32x04.hpp"

namespace stado {
template<>
struct BroadMask<32, 4> : public i32x4 {
  using Element = bool;

  // Default constructor:
  BroadMask() = default;
  // Constructor to build from all elements:
  BroadMask(bool x0, bool x1, bool x2, bool x3) : i32x4{-i32(x0), -i32(x1), -i32(x2), -i32(x3)} {}
  // Constructor to convert from type __m128i used in intrinsics:
  BroadMask(const __m128i x) : i32x4(x) {}
  BroadMask(const __m128 x) : i32x4(_mm_castps_si128(x)) {}
  // Assignment operator to convert from type __m128i used in intrinsics:
  BroadMask& operator=(const __m128i x) {
    xmm = x;
    return *this;
  }
  // Constructor to broadcast scalar value:
  BroadMask(bool b) : i32x4(-i32(b)) {}
  // Assignment operator to broadcast scalar value:
  BroadMask& operator=(bool b) {
    *this = BroadMask(b);
    return *this;
  }
  operator __m128i() const {
    return xmm;
  }
  operator __m128() const {
    return _mm_castsi128_ps(xmm);
  }
  BroadMask& insert(std::size_t index, bool a) {
    i32x4::insert(index, -i32(a));
    return *this;
  }
  // Member function extract a single element from vector
  bool extract(std::size_t index) const {
    return i32x4::extract(index) != 0;
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  bool operator[](std::size_t index) const {
    return extract(index);
  }
  // Member function to change a bitfield to a boolean vector
  BroadMask& load_bits(u8 a) {
    // broadcast byte
    __m128i b1 = _mm_set1_epi8(i8(a));
    __m128i m1 = _mm_setr_epi32(1, 2, 4, 8);
    // isolate one bit in each byte
    __m128i c1 = _mm_and_si128(b1, m1);
    // compare signed because no numbers are negative
    xmm = _mm_cmpgt_epi32(c1, _mm_setzero_si128());
    return *this;
  }

  // Prevent constructing from int, etc.
  BroadMask(int b) = delete;
  BroadMask& operator=(int x) = delete;
};
using b32x4 = BroadMask<32, 4>;

// vector operator & : bitwise and
static inline b32x4 operator&(const b32x4 a, const b32x4 b) {
  return {si128(a) & si128(b)};
}
static inline b32x4 operator&&(const b32x4 a, const b32x4 b) {
  return a & b;
}
// vector operator &= : bitwise and
static inline b32x4& operator&=(b32x4& a, const b32x4 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline b32x4 operator|(const b32x4 a, const b32x4 b) {
  return {si128(a) | si128(b)};
}
static inline b32x4 operator||(const b32x4 a, const b32x4 b) {
  return a | b;
}
// vector operator |= : bitwise or
static inline b32x4& operator|=(b32x4& a, const b32x4 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
static inline b32x4 operator^(const b32x4 a, const b32x4 b) {
  return {si128(a) ^ si128(b)};
}
// vector operator ^= : bitwise xor
static inline b32x4& operator^=(b32x4& a, const b32x4 b) {
  a = a ^ b;
  return a;
}

// vector operator == : xnor
static inline b32x4 operator==(const b32x4 a, const b32x4 b) {
  return {a ^ (~b)};
}

// vector operator != : xor
static inline b32x4 operator!=(const b32x4 a, const b32x4 b) {
  return {a ^ b};
}

// vector operator ~ : bitwise not
static inline b32x4 operator~(const b32x4 a) {
  return {~si128(a)};
}

// vector operator ! : element not
static inline b32x4 operator!(const b32x4 a) {
  return ~a;
}

// vector function andnot
static inline b32x4 andnot(const b32x4 a, const b32x4 b) {
  return {andnot(si128(a), si128(b))};
}

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(const b32x4 a) {
  return _mm_movemask_epi8(a) == 0xFFFF;
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(const b32x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1 // use ptest
  return _mm_testz_si128(a, a) == 0;
#else
  return _mm_movemask_epi8(a) != 0;
#endif
}

// to_bits: convert boolean vector to integer bitfield
static inline u8 to_bits(const b32x4 x) {
  __m128i a = _mm_packs_epi32(x, x); // 32-bit dwords to 16-bit words
  __m128i b = _mm_packs_epi16(a, a); // 16-bit words to bytes
  return u8(_mm_movemask_epi8(b) & 0xF);
}
} // namespace stado

#endif // INCLUDE_STADO_MASK_BROAD_32X04_HPP
