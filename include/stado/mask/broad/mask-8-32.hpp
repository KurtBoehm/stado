#ifndef INCLUDE_STADO_MASK_BROAD_MASK_8_32_HPP
#define INCLUDE_STADO_MASK_BROAD_MASK_8_32_HPP

#include <cstddef>
#include <cstdint>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad/base.hpp"
#include "stado/mask/broad/mask-8-16.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i8x16.hpp"
#include "stado/vector/native/types/i8x32.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
template<>
struct BroadMask<8, 32> : public i8x32 {
  using Element = bool;
  using Half = b8x16;

  // Default constructor:
  BroadMask() = default;
  // Constructor to build from all elements:
  BroadMask(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7, bool x8,
            bool x9, bool x10, bool x11, bool x12, bool x13, bool x14, bool x15, bool x16, bool x17,
            bool x18, bool x19, bool x20, bool x21, bool x22, bool x23, bool x24, bool x25,
            bool x26, bool x27, bool x28, bool x29, bool x30, bool x31)
      : i8x32(-i8(x0), -i8(x1), -i8(x2), -i8(x3), -i8(x4), -i8(x5), -i8(x6), -i8(x7), -i8(x8),
              -i8(x9), -i8(x10), -i8(x11), -i8(x12), -i8(x13), -i8(x14), -i8(x15), -i8(x16),
              -i8(x17), -i8(x18), -i8(x19), -i8(x20), -i8(x21), -i8(x22), -i8(x23), -i8(x24),
              -i8(x25), -i8(x26), -i8(x27), -i8(x28), -i8(x29), -i8(x30), -i8(x31)) {}
  // Constructor to convert from type __m256i used in intrinsics:
  BroadMask(const __m256i x) : i8x32(x) {}
  // Assignment operator to convert from type __m256i used in intrinsics:
  BroadMask& operator=(const __m256i x) {
    ymm = x;
    return *this;
  }
  // Constructor to broadcast scalar value:
  BroadMask(bool b) : i8x32(-i8(b)) {}
  // Constructor to convert from NativeVector<i8, 32>
  BroadMask(const i8x32 a) : i8x32(a) {}
  // Assignment operator to broadcast scalar value:
  BroadMask& operator=(bool b) {
    *this = BroadMask(b);
    return *this;
  }
  // Constructor to build from two Half:
  BroadMask(const Half a0, const Half a1) : i8x32(i8x16(a0), i8x16(a1)) {}
  // Member functions to split into two i8x16:
  Half get_low() const {
    return {i8x32::get_low()};
  }
  Half get_high() const {
    return {i8x32::get_high()};
  }
  BroadMask& insert(std::size_t index, bool a) {
    i8x32::insert(index, -(i8)a);
    return *this;
  }
  // Member function extract a single element from vector
  bool extract(std::size_t index) const {
    return i8x32::extract(index) != 0;
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  bool operator[](int index) const {
    return extract(index);
  }
  // Member function to change a bitfield to a boolean vector
  BroadMask& load_bits(uint32_t a) {
    __m256i b1 =
      _mm256_set1_epi32((int32_t)~a); // broadcast a. Invert because we have no compare-not-equal
    __m256i m1 = _mm256_setr_epi32(0, 0, 0x01010101, 0x01010101, 0x02020202, 0x02020202, 0x03030303,
                                   0x03030303);
    __m256i c1 = _mm256_shuffle_epi8(b1, m1); // get right byte in each position
    __m256i m2 = _mm256_setr_epi32(0x08040201, i32(0x80402010), 0x08040201, i32(0x80402010),
                                   0x08040201, i32(0x80402010), 0x08040201, i32(0x80402010));
    __m256i d1 = _mm256_and_si256(c1, m2); // isolate one bit in each byte
    ymm = _mm256_cmpeq_epi8(d1, _mm256_setzero_si256()); // compare with 0
    return *this;
  }

  // Prevent constructing from int, etc.
  BroadMask(int b) = delete;
  BroadMask& operator=(int x) = delete;
};
using b8x32 = BroadMask<8, 32>;

// vector operator & : bitwise and
static inline b8x32 operator&(const b8x32 a, const b8x32 b) {
  return i8x32(si256(a) & si256(b));
}
static inline b8x32 operator&&(const b8x32 a, const b8x32 b) {
  return a & b;
}
// vector operator &= : bitwise and
static inline b8x32& operator&=(b8x32& a, const b8x32 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline b8x32 operator|(const b8x32 a, const b8x32 b) {
  return i8x32(si256(a) | si256(b));
}
static inline b8x32 operator||(const b8x32 a, const b8x32 b) {
  return a | b;
}
// vector operator |= : bitwise or
static inline b8x32& operator|=(b8x32& a, const b8x32 b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
static inline b8x32 operator^(const b8x32 a, const b8x32 b) {
  return i8x32(si256(a) ^ si256(b));
}
// vector operator ^= : bitwise xor
static inline b8x32& operator^=(b8x32& a, const b8x32 b) {
  a = a ^ b;
  return a;
}

// vector operator == : xnor
static inline b8x32 operator==(const b8x32 a, const b8x32 b) {
  return i8x32(a ^ (~b));
}

// vector operator != : xor
static inline b8x32 operator!=(const b8x32 a, const b8x32 b) {
  return b8x32(a ^ b);
}

// vector operator ~ : bitwise not
static inline b8x32 operator~(const b8x32 a) {
  return i8x32(~si256(a));
}

// vector operator ! : element not
static inline b8x32 operator!(const b8x32 a) {
  return ~a;
}

// vector function andnot
static inline b8x32 andnot(const b8x32 a, const b8x32 b) {
  return i8x32(andnot(si256(a), si256(b)));
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_MASK_BROAD_MASK_8_32_HPP
