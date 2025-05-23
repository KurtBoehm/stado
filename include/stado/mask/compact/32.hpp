#ifndef INCLUDE_STADO_MASK_COMPACT_32_HPP
#define INCLUDE_STADO_MASK_COMPACT_32_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/base.hpp"
#include "stado/mask/compact/mask-16.hpp"

// 32-bit and 64-bit masks require AVX512BW
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
namespace stado {
// Compact vector of 32 booleans
template<>
struct CompactMask<32> {
  using Element = bool;
  using Register = __mmask32;
  using Half = CompactMask<16>;
  static constexpr std::size_t size = 32;

  // Default constructor:
  CompactMask() = default;
  // Constructor to convert from type __mmask32 used in intrinsics
  // Made explicit to prevent implicit conversion from int
  CompactMask(__mmask32 x) : mm(x) {}
  // Constructor to build from all elements:
  CompactMask(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8,
              bool b9, bool b10, bool b11, bool b12, bool b13, bool b14, bool b15, bool b16,
              bool b17, bool b18, bool b19, bool b20, bool b21, bool b22, bool b23, bool b24,
              bool b25, bool b26, bool b27, bool b28, bool b29, bool b30, bool b31)
      : mm(u32(b0) | u32(b1) << 1 | u32(b2) << 2 | u32(b3) << 3 | u32(b4) << 4 | u32(b5) << 5 |
           u32(b6) << 6 | u32(b7) << 7 | u32(b8) << 8 | u32(b9) << 9 | u32(b10) << 10 |
           u32(b11) << 11 | u32(b12) << 12 | u32(b13) << 13 | u32(b14) << 14 | u32(b15) << 15 |
           u32(b16) << 16 | u32(b17) << 17 | u32(b18) << 18 | u32(b19) << 19 | u32(b20) << 20 |
           u32(b21) << 21 | u32(b22) << 22 | u32(b23) << 23 | u32(b24) << 24 | u32(b25) << 25 |
           u32(b26) << 26 | u32(b27) << 27 | u32(b28) << 28 | u32(b29) << 29 | u32(b30) << 30 |
           u32(b31) << 31) {}
  // Constructor to broadcast single value:
  CompactMask(bool b) : mm(__mmask32(-i32(b))) {}
  // Constructor to make from two halves
  CompactMask(const Half x0, const Half x1) : mm(u16(__mmask16(x0)) | u32(__mmask16(x1)) << 16) {}
  // Assignment operator to convert from type __mmask32 used in intrinsics:
  CompactMask& operator=(__mmask32 x) {
    mm = x;
    return *this;
  }
  // Assignment operator to broadcast scalar value:
  CompactMask& operator=(bool b) {
    mm = CompactMask(b);
    return *this;
  }
  // Type cast operator to convert to __mmask32 used in intrinsics
  operator __mmask32() const {
    return mm;
  }
  // split into two halves
  Half get_low() const {
    return {__mmask16(mm)};
  }
  Half get_high() const {
    return {__mmask16(mm >> 16)};
  }
  // Member function to change a single element in vector
  CompactMask& insert(std::size_t index, bool value) {
    mm = __mmask32((u32(mm) & ~(1U << index)) | u32(value) << index);
    return *this;
  }
  // Member function extract a single element from vector
  bool extract(std::size_t index) const {
    return ((u32(mm) >> index) & 1) != 0U;
  }
  // Extract a single element. Operator [] can only read an element, not write.
  bool operator[](std::size_t index) const {
    return extract(index);
  }
  // Member function to change a bitfield to a boolean vector
  CompactMask& load_bits(u32 a) {
    mm = __mmask32(a);
    return *this;
  }

private:
  Register mm;
};

// vector operator & : bitwise and
static inline CompactMask<32> operator&(const CompactMask<32> a, const CompactMask<32> b) {
  return __mmask32(__mmask32(a) & __mmask32(b)); // _kand_mask32 not defined in all compilers
}
static inline CompactMask<32> operator&&(const CompactMask<32> a, const CompactMask<32> b) {
  return a & b;
}
// vector operator &= : bitwise and
static inline CompactMask<32>& operator&=(CompactMask<32>& a, const CompactMask<32> b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline CompactMask<32> operator|(const CompactMask<32> a, const CompactMask<32> b) {
  return __mmask32(__mmask32(a) | __mmask32(b)); // _kor_mask32
}
static inline CompactMask<32> operator||(const CompactMask<32> a, const CompactMask<32> b) {
  return a | b;
}
// vector operator |= : bitwise or
static inline CompactMask<32>& operator|=(CompactMask<32>& a, const CompactMask<32> b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
static inline CompactMask<32> operator^(const CompactMask<32> a, const CompactMask<32> b) {
  return __mmask32(__mmask32(a) ^ __mmask32(b)); // _kxor_mask32
}
// vector operator ^= : bitwise xor
static inline CompactMask<32>& operator^=(CompactMask<32>& a, const CompactMask<32> b) {
  a = a ^ b;
  return a;
}

// vector operator == : xnor
static inline CompactMask<32> operator==(const CompactMask<32> a, const CompactMask<32> b) {
  return __mmask32(__mmask32(a) ^ ~__mmask32(b)); // _kxnor_mask32
}

// vector operator != : xor
static inline CompactMask<32> operator!=(const CompactMask<32> a, const CompactMask<32> b) {
  return CompactMask<32>(a ^ b);
}

// vector operator ~ : bitwise not
static inline CompactMask<32> operator~(const CompactMask<32> a) {
  return __mmask32(~__mmask32(a)); // _knot_mask32
}

// vector operator ! : element not
static inline CompactMask<32> operator!(const CompactMask<32> a) {
  return ~a;
}

// vector function andnot
static inline CompactMask<32> andnot(const CompactMask<32> a, const CompactMask<32> b) {
  return __mmask32(~__mmask32(b) & __mmask32(a)); // _kandn_mask32
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_MASK_COMPACT_32_HPP
