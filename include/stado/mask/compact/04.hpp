#ifndef INCLUDE_STADO_MASK_COMPACT_04_HPP
#define INCLUDE_STADO_MASK_COMPACT_04_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/base.hpp"
#include "stado/mask/compact/mask-2.hpp"
#include "stado/mask/compact/mask-8.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
namespace stado {
template<>
struct CompactMask<4> : public CompactMask<8> {
  using Base = CompactMask<8>;
  using Half = CompactMask<2>;
  static constexpr std::size_t size = 4;

  // Default constructor:
  CompactMask() = default;
  // Constructor to make from two halves. Defined in operations.hpp.
  CompactMask(Half x0, Half x1);

  // Constructor to convert from type __mmask8 used in intrinsics
  CompactMask(__mmask8 x) : Base(x) {}
  // Assignment operator to convert from type __mmask8 used in intrinsics:
  CompactMask& operator=(__mmask8 x) {
    mm = x;
    return *this;
  }
  // Constructor to build from all elements:
  CompactMask(bool b0, bool b1, bool b2, bool b3)
      : Base(__mmask8(u8{b0} | u8{b1} << 1 | u8{b2} << 2 | u8{b3} << 3)) {}
  // Constructor to broadcast single value:
  CompactMask(bool b) : Base(u8(-i8(b) & 0x0F)) {}
  // Assignment operator to broadcast scalar value:
  CompactMask& operator=(bool b) {
    mm = CompactMask(b);
    return *this;
  }
  // Split into two halves. Defined in operations.hpp.
  Half get_low() const;
  Half get_high() const;

  // Member function to change a bitfield to a boolean vector
  CompactMask& load_bits(u8 a) {
    mm = a & 0x0F;
    return *this;
  }
};

// vector operator & : and
static inline CompactMask<4> operator&(CompactMask<4> a, CompactMask<4> b) {
  return __mmask8(__mmask8(a) &
                  __mmask8(b)); // _kand_mask8(__mmask8(a), __mmask8(b)) // not defined
}
static inline CompactMask<4> operator&&(CompactMask<4> a, CompactMask<4> b) {
  return a & b;
}

// vector operator | : or
static inline CompactMask<4> operator|(CompactMask<4> a, CompactMask<4> b) {
  return __mmask8(__mmask8(a) | __mmask8(b)); // _kor_mask8(__mmask8(a), __mmask8(b));
}
static inline CompactMask<4> operator||(CompactMask<4> a, CompactMask<4> b) {
  return a | b;
}

// vector operator ^ : xor
static inline CompactMask<4> operator^(CompactMask<4> a, CompactMask<4> b) {
  return __mmask8(__mmask8(a) ^ __mmask8(b)); // _kxor_mask8(__mmask8(a), __mmask8(b));
}

// vector operator ~ : not
static inline CompactMask<4> operator~(CompactMask<4> a) {
  return __mmask8(__mmask8(a) ^ 0x0FU);
}

// vector operator == : xnor
static inline CompactMask<4> operator==(CompactMask<4> a, CompactMask<4> b) {
  return ~(a ^ b);
}

// vector operator != : xor
static inline CompactMask<4> operator!=(CompactMask<4> a, CompactMask<4> b) {
  return a ^ b;
}

// vector operator ! : element not
static inline CompactMask<4> operator!(CompactMask<4> a) {
  return ~a;
}

// vector operator &= : and
static inline CompactMask<4>& operator&=(CompactMask<4>& a, CompactMask<4> b) {
  a = a & b;
  return a;
}

// vector operator |= : or
static inline CompactMask<4>& operator|=(CompactMask<4>& a, CompactMask<4> b) {
  a = a | b;
  return a;
}

// vector operator ^= : xor
static inline CompactMask<4>& operator^=(CompactMask<4>& a, CompactMask<4> b) {
  a = a ^ b;
  return a;
}

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(const CompactMask<4> a) {
  return (__mmask8(a) & 0x0FU) == 0x0F;
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(const CompactMask<4> a) {
  return (__mmask8(a) & 0x0FU) != 0;
}

// function andnot: a & ~ b
static inline CompactMask<4> andnot(const CompactMask<4> a, const CompactMask<4> b) {
  return __mmask8(andnot(CompactMask<8>(a), CompactMask<8>(b)));
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_MASK_COMPACT_04_HPP
