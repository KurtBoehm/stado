#ifndef INCLUDE_STADO_MASK_COMPACT_OPERATIONS_HPP
#define INCLUDE_STADO_MASK_COMPACT_OPERATIONS_HPP

#include <cstdint>

#include "stado/instruction-set.hpp"
#include "stado/mask/compact/mask-16.hpp"
#include "stado/mask/compact/mask-8.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
#include "stado/mask/compact/mask-2.hpp"
#include "stado/mask/compact/mask-32.hpp"
#include "stado/mask/compact/mask-4.hpp"
#include "stado/mask/compact/mask-64.hpp"
#endif

#if STADO_INSTRUCTION_SET == STADO_AVX512F
#include "stado/mask/broad/mask-32-8.hpp"
#include "stado/mask/broad/mask-64-4.hpp"
#endif

namespace stado {
// special cases of mixed compact and broad vectors
#if STADO_INSTRUCTION_SET == STADO_AVX512F
inline CompactMask<16>::CompactMask(BroadMask<32, 8> x0, BroadMask<32, 8> x1)
    : mm(to_bits(x0) | uint16_t(to_bits(x1) << 8)) {}
inline CompactMask<8>::CompactMask(BroadMask<64, 4> x0, BroadMask<64, 4> x1)
    : mm(to_bits(x0) | (to_bits(x1) << 4)) {}

inline BroadMask<32, 8> CompactMask<16>::get_low() const {
  return BroadMask<32, 8>().load_bits(uint8_t(mm));
}
inline BroadMask<32, 8> CompactMask<16>::get_high() const {
  return BroadMask<32, 8>().load_bits(uint8_t((uint16_t)mm >> 8U));
}
inline BroadMask<64, 4> CompactMask<8>::get_low() const {
  return BroadMask<64, 4>().load_bits(mm & 0xFU);
}
inline BroadMask<64, 4> CompactMask<8>::get_high() const {
  return BroadMask<64, 4>().load_bits(mm >> 4U);
}
#endif

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
// Members of CompactMask<16> that refer to CompactMask<8>:
inline CompactMask<16>::CompactMask(CompactMask<8> x0, CompactMask<8> x1)
    : mm(uint8_t(x0) | uint16_t(x1) << 8U) {}
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
inline CompactMask<8> CompactMask<16>::get_low() const {
  return CompactMask<8>().load_bits(uint8_t(mm));
}
inline CompactMask<8> CompactMask<16>::get_high() const {
  return CompactMask<8>().load_bits(uint8_t((uint16_t)mm >> 8U));
}
#endif
#endif

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
// Members of CompactMask<8> that refer to CompactMask<4>:
inline CompactMask<8>::CompactMask(CompactMask<4> x0, CompactMask<4> x1)
    : mm((uint8_t(x0) & 0x0FU) | (uint8_t(x1) << 4U)) {}
inline CompactMask<4> CompactMask<8>::get_low() const {
  return CompactMask<4>().load_bits(mm & 0xFU);
}
inline CompactMask<4> CompactMask<8>::get_high() const {
  return CompactMask<4>().load_bits(mm >> 4U);
}
//  Members of CompactMask<4> that refer to CompactMask<2>:
inline CompactMask<4>::CompactMask(CompactMask<2> x0, CompactMask<2> x1)
    : CompactMask<4>(uint8_t((uint8_t(x0) & 0x03U) | (uint8_t(x1) << 2U))) {}
inline CompactMask<2> CompactMask<4>::get_low() const {
  return CompactMask<2>().load_bits(mm & 3U);
}
inline CompactMask<2> CompactMask<4>::get_high() const {
  return CompactMask<2>().load_bits(mm >> 2U);
}

// horizontal_and. Returns true if all elements are true
static inline bool horizontal_and(CompactMask<32> a) {
  return __mmask32(a) == 0xFFFFFFFF;
}

// horizontal_or. Returns true if at least one element is true
static inline bool horizontal_or(CompactMask<32> a) {
  return __mmask32(a) != 0;
}

// to_bits: convert boolean vector to integer bitfield
static inline uint32_t to_bits(CompactMask<32> x) {
  return __mmask32(x);
}
// to_bits: convert boolean vector to integer bitfield
static inline uint64_t to_bits(CompactMask<64> x) {
  return uint64_t(__mmask64(x));
}
#endif
} // namespace stado

#endif // INCLUDE_STADO_MASK_COMPACT_OPERATIONS_HPP
