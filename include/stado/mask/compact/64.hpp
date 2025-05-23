#ifndef INCLUDE_STADO_MASK_COMPACT_64_HPP
#define INCLUDE_STADO_MASK_COMPACT_64_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/32.hpp"
#include "stado/mask/compact/base.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL

namespace stado {
template<>
struct CompactMask<64> {
  using Element = bool;
  using Register = __mmask64;
  using Half = CompactMask<32>;
  static constexpr std::size_t size = 64;

  // Default constructor:
  CompactMask() = default;
  // Constructor to build from all elements:
  CompactMask(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8,
              bool b9, bool b10, bool b11, bool b12, bool b13, bool b14, bool b15, bool b16,
              bool b17, bool b18, bool b19, bool b20, bool b21, bool b22, bool b23, bool b24,
              bool b25, bool b26, bool b27, bool b28, bool b29, bool b30, bool b31, bool b32,
              bool b33, bool b34, bool b35, bool b36, bool b37, bool b38, bool b39, bool b40,
              bool b41, bool b42, bool b43, bool b44, bool b45, bool b46, bool b47, bool b48,
              bool b49, bool b50, bool b51, bool b52, bool b53, bool b54, bool b55, bool b56,
              bool b57, bool b58, bool b59, bool b60, bool b61, bool b62, bool b63)
      : mm(u64(b0) | u64(b1) << 1U | u64(b2) << 2U | u64(b3) << 3U | u64(b4) << 4U | u64(b5) << 5U |
           u64(b6) << 6U | u64(b7) << 7U | u64(b8) << 8U | u64(b9) << 9U | u64(b10) << 10U |
           u64(b11) << 11U | u64(b12) << 12U | u64(b13) << 13U | u64(b14) << 14U | u64(b15) << 15U |
           u64(b16) << 16U | u64(b17) << 17U | u64(b18) << 18U | u64(b19) << 19U | u64(b20) << 20U |
           u64(b21) << 21U | u64(b22) << 22U | u64(b23) << 23U | u64(b24) << 24U | u64(b25) << 25U |
           u64(b26) << 26U | u64(b27) << 27U | u64(b28) << 28U | u64(b29) << 29U | u64(b30) << 30U |
           u64(b31) << 31U | u64(b32) << 32U | u64(b33) << 33U | u64(b34) << 34U | u64(b35) << 35U |
           u64(b36) << 36U | u64(b37) << 37U | u64(b38) << 38U | u64(b39) << 39U | u64(b40) << 40U |
           u64(b41) << 41U | u64(b42) << 42U | u64(b43) << 43U | u64(b44) << 44U | u64(b45) << 45U |
           u64(b46) << 46U | u64(b47) << 47U | u64(b48) << 48U | u64(b49) << 49U | u64(b50) << 50U |
           u64(b51) << 51U | u64(b52) << 52U | u64(b53) << 53U | u64(b54) << 54U | u64(b55) << 55U |
           u64(b56) << 56U | u64(b57) << 57U | u64(b58) << 58U | u64(b59) << 59U | u64(b60) << 60U |
           u64(b61) << 61U | u64(b62) << 62U | u64(b63) << 63U) {}
  // Constructor to convert from type __mmask64 used in intrinsics:
  CompactMask(__mmask64 x) : mm(x) {}
  // Constructor to broadcast single value:
  CompactMask(bool b) : mm(__mmask64(-i64(b))) {}
  // Constructor to make from two halves
  CompactMask(const CompactMask<32> x0, const CompactMask<32> x1)
      : mm(u32(__mmask32(x0)) | u64(__mmask32(x1)) << 32) {}
  // Assignment operator to convert from type __mmask64 used in intrinsics:
  CompactMask& operator=(__mmask64 x) {
    mm = x;
    return *this;
  }
  // Assignment operator to broadcast scalar value:
  CompactMask& operator=(bool b) {
    mm = CompactMask(b);
    return *this;
  }
  // split into two halves
  CompactMask<32> get_low() const {
    return {__mmask32(mm)};
  }
  CompactMask<32> get_high() const {
    return {__mmask32(mm >> 32)};
  }
  // Member function to change a single element in vector
  CompactMask& insert(std::size_t index, bool a) {
    u64 mask = u64(1) << index;
    mm = (mm & ~mask) | u64(a) << index;
    return *this;
  }
  // Member function extract a single element from vector
  bool extract(std::size_t index) const {
    return ((mm >> index) & 1) != 0;
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  bool operator[](std::size_t index) const {
    return extract(index);
  }
  // Type cast operator to convert to __mmask64 used in intrinsics
  operator __mmask64() const {
    return mm;
  }
  // Member function to change a bitfield to a boolean vector
  CompactMask& load_bits(u64 a) {
    mm = __mmask64(a);
    return *this;
  }

private:
  Register mm;
};

// vector operator & : bitwise and
static inline CompactMask<64> operator&(const CompactMask<64> a, const CompactMask<64> b) {
  // return _kand_mask64(a, b);
  return __mmask64(a) & __mmask64(b);
}
static inline CompactMask<64> operator&&(const CompactMask<64> a, const CompactMask<64> b) {
  return a & b;
}
// vector operator &= : bitwise and
static inline CompactMask<64>& operator&=(CompactMask<64>& a, const CompactMask<64> b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline CompactMask<64> operator|(const CompactMask<64> a, const CompactMask<64> b) {
  // return _kor_mask64(a, b);
  return __mmask64(a) | __mmask64(b);
}
static inline CompactMask<64> operator||(const CompactMask<64> a, const CompactMask<64> b) {
  return a | b;
}
// vector operator |= : bitwise or
static inline CompactMask<64>& operator|=(CompactMask<64>& a, const CompactMask<64> b) {
  a = a | b;
  return a;
}

// vector operator ^ : bitwise xor
static inline CompactMask<64> operator^(const CompactMask<64> a, const CompactMask<64> b) {
  // return _kxor_mask64(a, b);
  return __mmask64(a) ^ __mmask64(b);
}
// vector operator ^= : bitwise xor
static inline CompactMask<64>& operator^=(CompactMask<64>& a, const CompactMask<64> b) {
  a = a ^ b;
  return a;
}

// vector operator == : xnor
static inline CompactMask<64> operator==(const CompactMask<64> a, const CompactMask<64> b) {
  return __mmask64(a) ^ ~__mmask64(b);
  // return _kxnor_mask64(a, b); // not all compilers have this intrinsic
}

// vector operator != : xor
static inline CompactMask<64> operator!=(const CompactMask<64> a, const CompactMask<64> b) {
  // return _kxor_mask64(a, b);
  return __mmask64(a) ^ __mmask64(b);
}

// vector operator ~ : bitwise not
static inline CompactMask<64> operator~(const CompactMask<64> a) {
  // return _knot_mask64(a);
  return ~__mmask64(a);
}

// vector operator ! : element not
static inline CompactMask<64> operator!(const CompactMask<64> a) {
  return ~a;
}

// vector function andnot
static inline CompactMask<64> andnot(const CompactMask<64> a, const CompactMask<64> b) {
  // return  _kxnor_mask64(b, a);
  return __mmask64(a) & ~__mmask64(b);
}

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and(const CompactMask<64> a) {
  return i64(__mmask64(a)) == -(i64)(1);
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or(const CompactMask<64> a) {
  return i64(__mmask64(a)) != 0;
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_MASK_COMPACT_64_HPP
