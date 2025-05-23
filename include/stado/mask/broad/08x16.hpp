#ifndef INCLUDE_STADO_MASK_BROAD_08X16_HPP
#define INCLUDE_STADO_MASK_BROAD_08X16_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad/base.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i08x16.hpp"

namespace stado {
template<>
struct BroadMask<8, 16> : public i8x16 {
  using Element = bool;
  static constexpr std::size_t element_bits = 8;

  // Default constructor
  BroadMask() = default;
  // Constructor to build from all elements:
  BroadMask(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7, bool x8,
            bool x9, bool x10, bool x11, bool x12, bool x13, bool x14, bool x15) {
    xmm = i8x16(-i8(x0), -i8(x1), -i8(x2), -i8(x3), -i8(x4), -i8(x5), -i8(x6), -i8(x7), -i8(x8),
                -i8(x9), -i8(x10), -i8(x11), -i8(x12), -i8(x13), -i8(x14), -i8(x15));
  }
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
  BroadMask(bool b) : i8x16(-i8(b)) {}
  // Assignment operator to broadcast scalar value:
  BroadMask& operator=(bool b) {
    *this = BroadMask(b);
    return *this;
  }
  // Member function to change a single element in vector
  BroadMask& insert(std::size_t index, bool a) {
    i8x16::insert(index, -i8(a));
    return *this;
  }
  // Member function to change a bitfield to a boolean vector
  BroadMask& load_bits(u16 a) {
    u16 an = u16(~a); // invert because we have no compare-not-equal
#if STADO_INSTRUCTION_SET >= STADO_SSSE3 // pshufb available
    __m128i a1 = _mm_cvtsi32_si128(an); // load into xmm register
    __m128i dist = _mm_setr_epi32(0, 0, 0x01010101, 0x01010101);
    __m128i a2 = _mm_shuffle_epi8(a1, dist); // one byte of a in each element
    __m128i mask = _mm_setr_epi32(0x08040201, i32(0x80402010), 0x08040201, i32(0x80402010));
    __m128i a3 = _mm_and_si128(a2, mask); // isolate one bit in each byte
#else
    __m128i b1 = _mm_set1_epi8((i8)an); // broadcast low byte
    __m128i b2 = _mm_set1_epi8((i8)(an >> 8)); // broadcast high byte
    __m128i m1 = _mm_setr_epi32(0x08040201, i32(0x80402010), 0, 0);
    __m128i m2 = _mm_setr_epi32(0, 0, 0x08040201, i32(0x80402010));
    __m128i c1 = _mm_and_si128(b1, m1); // isolate one bit in each byte of lower half
    __m128i c2 = _mm_and_si128(b2, m2); // isolate one bit in each byte of upper half
    __m128i a3 = _mm_or_si128(c1, c2);
#endif
    xmm = _mm_cmpeq_epi8(a3, _mm_setzero_si128()); // compare with 0
    return *this;
  }
  // Member function extract a single element from vector
  bool extract(std::size_t index) const {
    return i8x16::extract(index) != 0;
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  bool operator[](std::size_t index) const {
    return extract(index);
  }

  // Prevent constructing from int, etc.
  BroadMask(int b) = delete;
  BroadMask& operator=(int x) = delete;
};
using b8x16 = BroadMask<8, 16>;

// vector operator & : bitwise and
static inline b8x16 operator&(const b8x16 a, const b8x16 b) {
  return {si128(a) & si128(b)};
}
static inline b8x16 operator&&(const b8x16 a, const b8x16 b) {
  return a & b;
}
// vector operator &= : bitwise and
static inline b8x16& operator&=(b8x16& a, const b8x16 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline b8x16 operator|(const b8x16 a, const b8x16 b) {
  return {si128(a) | si128(b)};
}
static inline b8x16 operator||(const b8x16 a, const b8x16 b) {
  return a | b;
}
// vector operator |= : bitwise or
static inline b8x16& operator|=(b8x16& a, const b8x16 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
static inline b8x16 operator^(const b8x16 a, const b8x16 b) {
  return {si128(a) ^ si128(b)};
}
// vector operator ^= : bitwise xor
static inline b8x16& operator^=(b8x16& a, const b8x16 b) {
  a = a ^ b;
  return a;
}

// vector operator == : xnor
static inline b8x16 operator==(const b8x16 a, const b8x16 b) {
  return {a ^ (~b)};
}

// vector operator != : xor
static inline b8x16 operator!=(const b8x16 a, const b8x16 b) {
  return {a ^ b};
}

// vector operator ~ : bitwise not
static inline b8x16 operator~(const b8x16 a) {
  return {~si128(a)};
}

// vector operator ! : element not
static inline b8x16 operator!(const b8x16 a) {
  return ~a;
}

// vector function andnot
static inline b8x16 andnot(const b8x16 a, const b8x16 b) {
  return {andnot(si128(a), si128(b))};
}

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(const b8x16 a) {
  return _mm_movemask_epi8(a) == 0xFFFF;
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(const b8x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1 // use ptest
  return _mm_testz_si128(a, a) == 0;
#else
  return _mm_movemask_epi8(a) != 0;
#endif
}

// to_bits: convert boolean vector to integer bitfield
static inline u16 to_bits(const b8x16 x) {
  return u16(_mm_movemask_epi8(x));
}
} // namespace stado

#endif // INCLUDE_STADO_MASK_BROAD_08X16_HPP
