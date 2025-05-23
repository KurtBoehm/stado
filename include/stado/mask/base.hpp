#ifndef INCLUDE_STADO_MASK_BASE_HPP
#define INCLUDE_STADO_MASK_BASE_HPP

#include <bit>
#include <cstddef>

#include "stado/instruction-set.hpp"
#include "stado/mask/broad.hpp"
#include "stado/mask/compact.hpp"

namespace stado {
template<std::size_t tElementBits, std::size_t tSize>
struct MaskTrait;

template<std::size_t tElementBits, std::size_t tSize>
requires(std::has_single_bit(tElementBits) && std::has_single_bit(tSize))
using Mask = typename MaskTrait<tElementBits, tSize>::Type;

// 128 bit
template<>
struct MaskTrait<8, 16> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Type = CompactMask<16>;
#else
  using Type = BroadMask<8, 16>;
#endif
};
template<>
struct MaskTrait<16, 8> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Type = CompactMask<8>;
#else
  using Type = BroadMask<16, 8>;
#endif
};
template<>
struct MaskTrait<32, 4> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Type = CompactMask<4>;
#else
  using Type = BroadMask<32, 4>;
#endif
};
template<>
struct MaskTrait<64, 2> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Type = CompactMask<2>;
#else
  using Type = BroadMask<64, 2>;
#endif
};

// 256 bit
#if STADO_INSTRUCTION_SET >= STADO_AVX2
template<>
struct MaskTrait<8, 32> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Type = CompactMask<32>;
#else
  using Type = BroadMask<8, 32>;
#endif
};
template<>
struct MaskTrait<16, 16> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Type = CompactMask<16>;
#else
  using Type = BroadMask<16, 16>;
#endif
};
#endif

#if STADO_INSTRUCTION_SET >= STADO_AVX
template<>
struct MaskTrait<32, 8> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Type = CompactMask<8>;
#else
  using Type = BroadMask<32, 8>;
#endif
};
template<>
struct MaskTrait<64, 4> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  using Type = CompactMask<4>;
#else
  using Type = BroadMask<64, 4>;
#endif
};
#endif

// 512 bit
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
template<>
struct MaskTrait<8, 64> {
  using Type = CompactMask<64>;
};
template<>
struct MaskTrait<16, 32> {
  using Type = CompactMask<32>;
};
#endif
#if STADO_INSTRUCTION_SET >= STADO_AVX512F
template<>
struct MaskTrait<32, 16> {
  using Type = CompactMask<16>;
};
template<>
struct MaskTrait<64, 8> {
  using Type = CompactMask<8>;
};
#endif
} // namespace stado

#endif // INCLUDE_STADO_MASK_BASE_HPP
