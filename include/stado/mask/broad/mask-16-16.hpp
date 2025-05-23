#ifndef INCLUDE_STADO_MASK_BROAD_MASK_16_16_HPP
#define INCLUDE_STADO_MASK_BROAD_MASK_16_16_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad/base.hpp"
#include "stado/mask/broad/mask-16-8.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i16x16.hpp"
#include "stado/vector/native/types/i16x8.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
template<>
struct BroadMask<16, 16> : public i16x16 {
  using Element = bool;
  using Half = b16x8;

  // Default constructor:
  BroadMask() = default;
  // Constructor to build from all elements:
  BroadMask(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7, bool x8,
            bool x9, bool x10, bool x11, bool x12, bool x13, bool x14, bool x15)
      : i16x16(-i16(x0), -i16(x1), -i16(x2), -i16(x3), -i16(x4), -i16(x5), -i16(x6), -i16(x7),
               -i16(x8), -i16(x9), -i16(x10), -i16(x11), -i16(x12), -i16(x13), -i16(x14),
               -i16(x15)) {}
  // Constructor to convert from type __m256i used in intrinsics:
  BroadMask(const __m256i x) : i16x16(x) {}
  // Assignment operator to convert from type __m256i used in intrinsics:
  BroadMask& operator=(const __m256i x) {
    ymm = x;
    return *this;
  }
  // Constructor to broadcast scalar value:
  BroadMask(bool b) : i16x16(-i16(b)) {}
  // Constructor to convert from type si256 used in emulation:
  BroadMask(const si256& x) : i16x16(x) {}
  // Assignment operator to broadcast scalar value:
  BroadMask& operator=(bool b) {
    *this = BroadMask(b);
    return *this;
  }
  // Constructor to build from two Half:
  BroadMask(const Half a0, const Half a1) : i16x16(i16x8(a0), i16x8(a1)) {}
  Half get_low() const {
    return {i16x16::get_low()};
  }
  Half get_high() const {
    return {i16x16::get_high()};
  }
  BroadMask& insert(std::size_t index, bool a) {
    i16x16::insert(index, -i16(a));
    return *this;
  }
  // Member function extract a single element from vector
  bool extract(std::size_t index) const {
    return i16x16::extract(index) != 0;
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  bool operator[](std::size_t index) const {
    return extract(index);
  }
  // Member function to change a bitfield to a boolean vector
  BroadMask& load_bits(u16 a) {
    __m256i b1 = _mm256_set1_epi16(i16(a)); // broadcast a
    __m256i m1 = _mm256_setr_epi32(0, 0, 0, 0, 0x00010001, 0x00010001, 0x00010001, 0x00010001);
    __m256i c1 = _mm256_shuffle_epi8(b1, m1); // get right byte in each position
    __m256i m2 = _mm256_setr_epi32(0x00020001, 0x00080004, 0x00200010, 0x00800040, 0x00020001,
                                   0x00080004, 0x00200010, 0x00800040);
    __m256i d1 = _mm256_and_si256(c1, m2); // isolate one bit in each byte
    ymm = _mm256_cmpgt_epi16(d1, _mm256_setzero_si256()); // compare with 0
    return *this;
  }

  // Prevent constructing from int, etc.
  BroadMask(int b) = delete;
  BroadMask& operator=(int x) = delete;
};
using b16x16 = BroadMask<16, 16>;

// vector operator & : bitwise and
static inline b16x16 operator&(const b16x16 a, const b16x16 b) {
  return {si256(a) & si256(b)};
}
static inline b16x16 operator&&(const b16x16 a, const b16x16 b) {
  return a & b;
}
// vector operator &= : bitwise and
static inline b16x16& operator&=(b16x16& a, const b16x16 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline b16x16 operator|(const b16x16 a, const b16x16 b) {
  return {si256(a) | si256(b)};
}
static inline b16x16 operator||(const b16x16 a, const b16x16 b) {
  return a | b;
}
// vector operator |= : bitwise or
static inline b16x16& operator|=(b16x16& a, const b16x16 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
static inline b16x16 operator^(const b16x16 a, const b16x16 b) {
  return {si256(a) ^ si256(b)};
}
// vector operator ^= : bitwise xor
static inline b16x16& operator^=(b16x16& a, const b16x16 b) {
  a = a ^ b;
  return a;
}

// vector operator == : xnor
static inline b16x16 operator==(const b16x16 a, const b16x16 b) {
  return {a ^ b16x16(~b)};
}

// vector operator != : xor
static inline b16x16 operator!=(const b16x16 a, const b16x16 b) {
  return {a ^ b};
}

// vector operator ~ : bitwise not
static inline b16x16 operator~(const b16x16 a) {
  return {~si256(a)};
}

// vector operator ! : element not
static inline b16x16 operator!(const b16x16 a) {
  return ~a;
}

// vector function andnot
static inline b16x16 andnot(const b16x16 a, const b16x16 b) {
  return {andnot(si256(a), si256(b))};
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_MASK_BROAD_MASK_16_16_HPP
