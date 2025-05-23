#ifndef INCLUDE_STADO_MASK_COMPACT_08_HPP
#define INCLUDE_STADO_MASK_COMPACT_08_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad/64x04.hpp"
#include "stado/mask/compact/base.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
template<>
struct CompactMask<8> {
  using Value = bool;
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Register = __mmask8;
#else
  // There is a problem in the case where we have AVX512F but not AVX512DQ:
  // We have 8-bit masks, but 8-bit mask operations (KMOVB, KANDB, etc.) require AVX512DQ.
  // We have to use 16-bit mask operations on 8-bit masks (KMOVW, KANDW, etc.).
  // I don't know if this is necessary, but I am using __mmask16 rather than __mmask8
  // in this case to avoid that the compiler generates 8-bit mask instructions.
  // We may get warnings in MS compiler when using __mmask16 on intrinsic functions
  // that require __mmask8, but I would rather have warnings than code that crashes.
  using Register = __mmask16;
#endif
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Half = CompactMask<4>;
#else // special case of mixed compact and broad vectors
  using Half = BroadMask<64, 4>;
#endif
  static constexpr std::size_t size = 8;

  // Default constructor:
  CompactMask() = default;
  // Constructor to convert from type  __mmask8 used in intrinsics
  CompactMask(__mmask8 x) : mm(x) {}
  // Constructor to convert from type  __mmask16 used in intrinsics
  CompactMask(__mmask16 x) : mm(Register(x)) {}
  // Constructor to make from two halves
  // Implemented after declaration of CompactMask<4>
  inline CompactMask(Half x0, Half x1);

  // Assignment operator to convert from type __mmask16 used in intrinsics:
  CompactMask& operator=(Register x) {
    mm = x;
    return *this;
  }
  // Constructor to build from all elements:
  CompactMask(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7)
      : mm(u8(u8{b0} | u8{b1} << 1 | u8{b2} << 2 | u8{b3} << 3 | u8{b4} << 4 | u8{b5} << 5 |
              u8{b6} << 6 | u8{b7} << 7)) {}
  // Constructor to broadcast single value:
  CompactMask(bool b) : mm(Register(-i16(b))) {}
  // Assignment operator to broadcast scalar value:
  CompactMask& operator=(bool b) {
    mm = Register(CompactMask(b));
    return *this;
  }
  // Type cast operator to convert to __mmask16 used in intrinsics
  operator Register() const {
    return mm;
  }
  // Split into two halves. Defined in operations.hpp.
  Half get_low() const;
  Half get_high() const;
  // Member function to change a single element in vector
  CompactMask& insert(std::size_t index, bool value) {
    mm = Register((u8(mm) & ~(1U << index)) | (unsigned)value << index);
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
  CompactMask& load_bits(u8 a) {
    mm = Register(a);
    return *this;
  }

protected:
  Register mm;
};

// vector operator & : and
static inline CompactMask<8> operator&(CompactMask<8> a, CompactMask<8> b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // 8-bit mask operations require AVX512DQ
  // _kand_mask8(__mmask8(a), __mmask8(b)) // not defined
  // must convert result to 8 bit, because bitwise operators promote everything to 32 bit results
  return __mmask8(__mmask8(a) & __mmask8(b));
#else
  return _mm512_kand(__mmask16(a), __mmask16(b));
#endif
}
static inline CompactMask<8> operator&&(CompactMask<8> a, CompactMask<8> b) {
  return a & b;
}

// vector operator | : or
static inline CompactMask<8> operator|(CompactMask<8> a, CompactMask<8> b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // 8-bit mask operations require AVX512DQ
  return __mmask8(__mmask8(a) | __mmask8(b)); // _kor_mask8(__mmask8(a), __mmask8(b));
#else
  return _mm512_kor(__mmask16(a), __mmask16(b));
#endif
}
static inline CompactMask<8> operator||(CompactMask<8> a, CompactMask<8> b) {
  return a | b;
}

// vector operator ^ : xor
static inline CompactMask<8> operator^(CompactMask<8> a, CompactMask<8> b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // 8-bit mask operations require AVX512DQ
  return __mmask8(__mmask8(a) ^ __mmask8(b)); // _kxor_mask8(__mmask8(a), __mmask8(b));
#else
  return _mm512_kxor(__mmask16(a), __mmask16(b));
#endif
}

// vector operator == : xnor
static inline CompactMask<8> operator==(CompactMask<8> a, CompactMask<8> b) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // 8-bit mask operations require AVX512DQ
  return __mmask8(~(__mmask8(a) ^ __mmask8(b))); // _kxnor_mask8(__mmask8(a), __mmask8(b));
#else
  return __mmask16(u8(__mmask8(a) ^ __mmask8(b)));
#endif
}

// vector operator != : xor
static inline CompactMask<8> operator!=(CompactMask<8> a, CompactMask<8> b) {
  return a ^ b;
}

// vector operator ~ : not
static inline CompactMask<8> operator~(CompactMask<8> a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // 8-bit mask operations require AVX512DQ
  return __mmask8(~__mmask8(a)); //_knot_mask8(__mmask8(a));
#else
  return _mm512_knot(__mmask16(a));
#endif
}

// vector operator ! : element not
static inline CompactMask<8> operator!(CompactMask<8> a) {
  return ~a;
}

// vector operator &= : and
static inline CompactMask<8>& operator&=(CompactMask<8>& a, CompactMask<8> b) {
  a = a & b;
  return a;
}

// vector operator |= : or
static inline CompactMask<8>& operator|=(CompactMask<8>& a, CompactMask<8> b) {
  a = a | b;
  return a;
}

// vector operator ^= : xor
static inline CompactMask<8>& operator^=(CompactMask<8>& a, CompactMask<8> b) {
  a = a ^ b;
  return a;
}

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(const CompactMask<8> a) {
  return u8(CompactMask<8>::Register(a)) == 0xFFU;
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(const CompactMask<8> a) {
  return u8(CompactMask<8>::Register(a)) != 0;
}

// function andnot: a & ~ b
static inline CompactMask<8> andnot(const CompactMask<8> a, const CompactMask<8> b) {
  return CompactMask<8>::Register(_mm512_kandn(b, a));
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_MASK_COMPACT_08_HPP
