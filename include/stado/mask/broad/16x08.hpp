#ifndef INCLUDE_STADO_MASK_BROAD_16X08_HPP
#define INCLUDE_STADO_MASK_BROAD_16X08_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad/base.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i16x8.hpp"

namespace stado {
template<>
struct BroadMask<16, 8> : public i16x8 {
  using Element = bool;

  // Constructor to build from all elements:
  BroadMask(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7) {
    xmm = i16x8(-i16(x0), -i16(x1), -i16(x2), -i16(x3), -i16(x4), -i16(x5), -i16(x6), -i16(x7));
  }
  // Default constructor:
  BroadMask() = default;
  // Constructor to convert from type __m128i used in intrinsics:
  BroadMask(const __m128i x) {
    xmm = x;
  }
  // Assignment operator to convert from type __m128i used in intrinsics:
  BroadMask& operator=(const __m128i x) {
    xmm = x;
    return *this;
  }
  // Constructor to broadcast scalar value:
  BroadMask(bool b) : i16x8(-i16(b)) {}
  // Assignment operator to broadcast scalar value:
  BroadMask& operator=(bool b) {
    *this = BroadMask(b);
    return *this;
  }
  BroadMask& insert(std::size_t index, bool a) {
    i16x8::insert(index, -i16(a));
    return *this;
  }
  // Member function extract a single element from vector
  // Note: This function is inefficient. Use store function if extracting more than one element
  bool extract(std::size_t index) const {
    return i16x8::extract(index) != 0;
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  bool operator[](std::size_t index) const {
    return extract(index);
  }
  // Member function to change a bitfield to a boolean vector
  BroadMask& load_bits(u8 a) {
    // broadcast byte. Invert because we have no compare-not-equal
    __m128i b1 = _mm_set1_epi8(i8(a));
    __m128i m1 = _mm_setr_epi32(0x00020001, 0x00080004, 0x00200010, 0x00800040);
    __m128i c1 = _mm_and_si128(b1, m1); // isolate one bit in each byte
    xmm = _mm_cmpgt_epi16(c1, _mm_setzero_si128()); // compare with 0
    return *this;
  }

  // Prevent constructing from int, etc.
  BroadMask(int b) = delete;
  BroadMask& operator=(int x) = delete;
};
using b16x8 = BroadMask<16, 8>;

// vector operator & : bitwise and
static inline b16x8 operator&(const b16x8 a, const b16x8 b) {
  return {si128(a) & si128(b)};
}
static inline b16x8 operator&&(const b16x8 a, const b16x8 b) {
  return a & b;
}
// vector operator &= : bitwise and
static inline b16x8& operator&=(b16x8& a, const b16x8 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline b16x8 operator|(const b16x8 a, const b16x8 b) {
  return {si128(a) | si128(b)};
}
static inline b16x8 operator||(const b16x8 a, const b16x8 b) {
  return a | b;
}
// vector operator |= : bitwise or
static inline b16x8& operator|=(b16x8& a, const b16x8 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
static inline b16x8 operator^(const b16x8 a, const b16x8 b) {
  return {si128(a) ^ si128(b)};
}
// vector operator ^= : bitwise xor
static inline b16x8& operator^=(b16x8& a, const b16x8 b) {
  a = a ^ b;
  return a;
}

// vector operator == : xnor
static inline b16x8 operator==(const b16x8 a, const b16x8 b) {
  return {a ^ (~b)};
}

// vector operator != : xor
static inline b16x8 operator!=(const b16x8 a, const b16x8 b) {
  return a ^ b;
}

// vector operator ~ : bitwise not
static inline b16x8 operator~(const b16x8 a) {
  return {~si128(a)};
}

// vector operator ! : element not
static inline b16x8 operator!(const b16x8 a) {
  return ~a;
}

// vector function andnot
static inline b16x8 andnot(const b16x8 a, const b16x8 b) {
  return {andnot(si128(a), si128(b))};
}

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(const b16x8 a) {
  return _mm_movemask_epi8(a) == 0xFFFF;
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(const b16x8 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1 // use ptest
  return _mm_testz_si128(a, a) == 0;
#else
  return _mm_movemask_epi8(a) != 0;
#endif
}

// to_bits: convert boolean vector to integer bitfield
static inline u8 to_bits(const b16x8 x) {
  __m128i a = _mm_packs_epi16(x, x); // 16-bit words to bytes
  return (u8)_mm_movemask_epi8(a);
}
} // namespace stado

#endif // INCLUDE_STADO_MASK_BROAD_16X08_HPP
