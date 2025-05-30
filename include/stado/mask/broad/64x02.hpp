#ifndef INCLUDE_STADO_MASK_BROAD_64X02_HPP
#define INCLUDE_STADO_MASK_BROAD_64X02_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad/base.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i64x02.hpp"

namespace stado {
template<>
struct BroadMask<64, 2> : public i64x2 {
  using Value = bool;
  static constexpr std::size_t element_bits = 64;

  // Default constructor:
  BroadMask() = default;
  // Constructor to build from all elements:
  BroadMask(bool x0, bool x1) : i64x2(-i64(x0), -i64(x1)) {}
  // Constructor to convert from type __m128i used in intrinsics:
  BroadMask(const __m128i x) : i64x2(x) {}
  BroadMask(const __m128d x) : i64x2(_mm_castpd_si128(x)) {}
  // Assignment operator to convert from type __m128i used in intrinsics:
  BroadMask& operator=(const __m128i x) {
    xmm = x;
    return *this;
  }
  // Constructor to broadcast scalar value:
  BroadMask(bool b) : i64x2(-i64(b)) {}
  // Assignment operator to broadcast scalar value:
  BroadMask& operator=(bool b) {
    *this = BroadMask(b);
    return *this;
  }
  operator __m128i() const {
    return xmm;
  }
  operator __m128d() const {
    return _mm_castsi128_pd(xmm);
  }
  BroadMask& insert(std::size_t index, bool a) {
    i64x2::insert(index, -i64(a));
    return *this;
  }
  // Member function extract a single element from vector
  bool extract(std::size_t index) const {
    return i64x2::extract(index) != 0;
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
    __m128i m1 = _mm_setr_epi32(1, 1, 2, 2);
    // isolate one bit in each byte
    __m128i c1 = _mm_and_si128(b1, m1);
    // compare with 0 (64 bit compare requires SSE4.1)
    xmm = _mm_cmpgt_epi32(c1, _mm_setzero_si128());
    return *this;
  }

  // Prevent constructing from int, etc.
  BroadMask(int b) = delete;
  BroadMask& operator=(int x) = delete;
};
using b64x2 = BroadMask<64, 2>;

// vector operator & : bitwise and
static inline b64x2 operator&(const b64x2 a, const b64x2 b) {
  return {si128(a) & si128(b)};
}
static inline b64x2 operator&&(const b64x2 a, const b64x2 b) {
  return a & b;
}
// vector operator &= : bitwise and
static inline b64x2& operator&=(b64x2& a, const b64x2 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline b64x2 operator|(const b64x2 a, const b64x2 b) {
  return {si128(a) | si128(b)};
}
static inline b64x2 operator||(const b64x2 a, const b64x2 b) {
  return a | b;
}
// vector operator |= : bitwise or
static inline b64x2& operator|=(b64x2& a, const b64x2 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
static inline b64x2 operator^(const b64x2 a, const b64x2 b) {
  return {si128(a) ^ si128(b)};
}
// vector operator ^= : bitwise xor
static inline b64x2& operator^=(b64x2& a, const b64x2 b) {
  a = a ^ b;
  return a;
}

// vector operator == : xnor
static inline b64x2 operator==(const b64x2 a, const b64x2 b) {
  return {a ^ (~b)};
}

// vector operator != : xor
static inline b64x2 operator!=(const b64x2 a, const b64x2 b) {
  return {a ^ b};
}

// vector operator ~ : bitwise not
static inline b64x2 operator~(const b64x2 a) {
  return {~si128(a)};
}

// vector operator ! : element not
static inline b64x2 operator!(const b64x2 a) {
  return ~a;
}

// vector function andnot
static inline b64x2 andnot(const b64x2 a, const b64x2 b) {
  return {andnot(si128(a), si128(b))};
}

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(const b64x2 a) {
  return _mm_movemask_epi8(a) == 0xFFFF;
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(const b64x2 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1 // use ptest
  return _mm_testz_si128(a, a) == 0;
#else
  return _mm_movemask_epi8(a) != 0;
#endif
}

// to_bits: convert boolean vector to integer bitfield
static inline u8 to_bits(const b64x2 x) {
  u32 a = u32(_mm_movemask_epi8(x));
  return (a & 1) | ((a >> 7) & 2);
}
} // namespace stado

#endif // INCLUDE_STADO_MASK_BROAD_64X02_HPP
