#ifndef INCLUDE_STADO_MASK_COMPACT_02_HPP
#define INCLUDE_STADO_MASK_COMPACT_02_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/base.hpp"
#include "stado/mask/compact/mask-8.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
namespace stado {
template<>
struct CompactMask<2> : public CompactMask<8> {
  static constexpr std::size_t size = 2;

  // Default constructor:
  CompactMask() = default;
  // Constructor to convert from type __mmask8 used in intrinsics
  CompactMask(__mmask8 x) : CompactMask<8>(x) {}
  // Assignment operator to convert from type __mmask16 used in intrinsics:
  CompactMask& operator=(__mmask8 x) {
    mm = x;
    return *this;
  }
  // Constructor to build from all elements:
  CompactMask(bool b0, bool b1) : CompactMask<8>(u8(u8{b0} | u8{b1} << 1)) {}
  // Constructor to broadcast single value:
  CompactMask(bool b) : CompactMask<8>(u8(-i8(b) & 0x03)) {}
  // Assignment operator to broadcast scalar value:
  CompactMask& operator=(bool b) {
    mm = CompactMask(b);
    return *this;
  }
  // Member function to change a bitfield to a boolean vector
  CompactMask& load_bits(u8 a) {
    mm = a & 0x03;
    return *this;
  }
};

// vector operator & : and
static inline CompactMask<2> operator&(CompactMask<2> a, CompactMask<2> b) {
  return __mmask8(__mmask8(a) &
                  __mmask8(b)); // _kand_mask8(__mmask8(a), __mmask8(b)) // not defined
}
static inline CompactMask<2> operator&&(CompactMask<2> a, CompactMask<2> b) {
  return a & b;
}

// vector operator | : or
static inline CompactMask<2> operator|(CompactMask<2> a, CompactMask<2> b) {
  return __mmask8(__mmask8(a) | __mmask8(b)); // _kor_mask8(__mmask8(a), __mmask8(b));
}
static inline CompactMask<2> operator||(CompactMask<2> a, CompactMask<2> b) {
  return a | b;
}

// vector operator ^ : xor
static inline CompactMask<2> operator^(CompactMask<2> a, CompactMask<2> b) {
  return __mmask8(__mmask8(a) ^ __mmask8(b)); // _kxor_mask8(__mmask8(a), __mmask8(b));
}

// vector operator ~ : not
static inline CompactMask<2> operator~(CompactMask<2> a) {
  return __mmask8(__mmask8(a) ^ 0x03U);
}

// vector operator == : xnor
static inline CompactMask<2> operator==(CompactMask<2> a, CompactMask<2> b) {
  return ~(a ^ b);
}

// vector operator != : xor
static inline CompactMask<2> operator!=(CompactMask<2> a, CompactMask<2> b) {
  return a ^ b;
}

// vector operator ! : element not
static inline CompactMask<2> operator!(CompactMask<2> a) {
  return ~a;
}

// vector operator &= : and
static inline CompactMask<2>& operator&=(CompactMask<2>& a, CompactMask<2> b) {
  a = a & b;
  return a;
}

// vector operator |= : or
static inline CompactMask<2>& operator|=(CompactMask<2>& a, CompactMask<2> b) {
  a = a | b;
  return a;
}

// vector operator ^= : xor
static inline CompactMask<2>& operator^=(CompactMask<2>& a, CompactMask<2> b) {
  a = a ^ b;
  return a;
}

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(const CompactMask<2> a) {
  return (__mmask8(a) & 0x03U) == 0x03;
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(const CompactMask<2> a) {
  return (__mmask8(a) & 0x03U) != 0;
}

// function andnot: a & ~ b
static inline CompactMask<2> andnot(const CompactMask<2> a, const CompactMask<2> b) {
  return __mmask8(andnot(CompactMask<8>(a), CompactMask<8>(b)));
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_MASK_COMPACT_02_HPP
