#ifndef INCLUDE_STADO_MASK_COMPACT_MASK_16_HPP
#define INCLUDE_STADO_MASK_COMPACT_MASK_16_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad/mask-32-8.hpp"
#include "stado/mask/compact/base.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
template<>
struct CompactMask<16> {
  using Element = bool;
  using Register = __mmask16;
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Half = CompactMask<8>;
#else // special case of mixed compact and broad vectors
  using Half = BroadMask<32, 8>;
#endif
  static constexpr std::size_t size = 16;

  // Default constructor:
  CompactMask() = default;
  // Constructor to convert from type __mmask16 used in intrinsics
  CompactMask(__mmask16 x) : mm(x) {}
  // Constructor to build from all elements:
  CompactMask(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8,
              bool b9, bool b10, bool b11, bool b12, bool b13, bool b14, bool b15)
      : mm(u16{b0} | u16{b1} << 1U | u16{b2} << 2U | u16{b3} << 3U | u16{b4} << 4U | u16{b5} << 5U |
           u16{b6} << 6U | u16{b7} << 7U | u16{b8} << 8U | u16{b9} << 9U | u16{b10} << 10U |
           u16{b11} << 11U | u16{b12} << 12U | u16{b13} << 13U | u16{b14} << 14U |
           u16{b15} << 15U) {}
  // Constructor to broadcast single value:
  CompactMask(bool b) : mm(__mmask16(-i16(b))) {}
  // Constructor to make from two halves. Defined in operations.hpp.
  inline CompactMask(CompactMask<8> x0, CompactMask<8> x1);
#if STADO_INSTRUCTION_SET == STADO_AVX512F // special case of mixed compact and broad vectors
  inline CompactMask(BroadMask<32, 8> const x0, BroadMask<32, 8> const x1);
#endif

  // Assignment operator to convert from type __mmask16 used in intrinsics:
  CompactMask& operator=(__mmask16 x) {
    mm = x;
    return *this;
  }
  // Assignment operator to broadcast scalar value:
  CompactMask& operator=(bool b) {
    mm = CompactMask(b);
    return *this;
  }
  // Type cast operator to convert to __mmask16 used in intrinsics
  operator __mmask16() const {
    return mm;
  }
  // Split into two halves. Defined in operations.hpp.
  Half get_low() const;
  Half get_high() const;
  // Member function to change a single element in vector
  CompactMask& insert(std::size_t index, bool value) {
    mm = __mmask16((u16(mm) & ~(1U << index)) | (unsigned)value << index);
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
  CompactMask& load_bits(u16 a) {
    mm = __mmask16(a);
    return *this;
  }

private:
  Register mm;
};

// vector operator & : and
static inline CompactMask<16> operator&(CompactMask<16> a, CompactMask<16> b) {
  return _mm512_kand(__mmask16(a), __mmask16(b));
}
static inline CompactMask<16> operator&&(CompactMask<16> a, CompactMask<16> b) {
  return a & b;
}

// vector operator | : or
static inline CompactMask<16> operator|(CompactMask<16> a, CompactMask<16> b) {
  return _mm512_kor(__mmask16(a), __mmask16(b));
}
static inline CompactMask<16> operator||(CompactMask<16> a, CompactMask<16> b) {
  return a | b;
}

// vector operator ^ : xor
static inline CompactMask<16> operator^(CompactMask<16> a, CompactMask<16> b) {
  return _mm512_kxor(__mmask16(a), __mmask16(b));
}

// vector operator == : xnor
static inline CompactMask<16> operator==(CompactMask<16> a, CompactMask<16> b) {
  return _mm512_kxnor(__mmask16(a), __mmask16(b));
}

// vector operator != : xor
static inline CompactMask<16> operator!=(CompactMask<16> a, CompactMask<16> b) {
  return a ^ b;
}

// vector operator ~ : not
static inline CompactMask<16> operator~(CompactMask<16> a) {
  return _mm512_knot(__mmask16(a));
}

// vector operator ! : element not
static inline CompactMask<16> operator!(CompactMask<16> a) {
  return ~a;
}

// vector operator &= : and
static inline CompactMask<16>& operator&=(CompactMask<16>& a, CompactMask<16> b) {
  a = a & b;
  return a;
}

// vector operator |= : or
static inline CompactMask<16>& operator|=(CompactMask<16>& a, CompactMask<16> b) {
  a = a | b;
  return a;
}

// vector operator ^= : xor
static inline CompactMask<16>& operator^=(CompactMask<16>& a, CompactMask<16> b) {
  a = a ^ b;
  return a;
}

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(const CompactMask<16> a) {
  return __mmask16(a) == 0xFFFF;
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(const CompactMask<16> a) {
  return __mmask16(a) != 0;
}

// function andnot: a & ~ b
static inline CompactMask<16> andnot(const CompactMask<16> a, const CompactMask<16> b) {
  return _mm512_kandn(b, a);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_MASK_COMPACT_MASK_16_HPP
