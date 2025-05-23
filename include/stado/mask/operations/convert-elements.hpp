#ifndef INCLUDE_STADO_MASK_OPERATIONS_CONVERT_ELEMENTS_HPP
#define INCLUDE_STADO_MASK_OPERATIONS_CONVERT_ELEMENTS_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad.hpp"
#include "stado/mask/compact.hpp"
#include "stado/mask/subnative.hpp"

namespace stado {
#define STADO_CONV_ESC(...) __VA_ARGS__

#define STADO_CONV_SAFE(FROM, TO) \
  template<> \
  struct ElementConvertTrait<STADO_CONV_ESC FROM, STADO_CONV_ESC TO> { \
    static constexpr bool is_safe = true; \
    static inline STADO_CONV_ESC TO convert(STADO_CONV_ESC FROM m); \
  }; \
  STADO_CONV_ESC TO ElementConvertTrait<STADO_CONV_ESC FROM, STADO_CONV_ESC TO>::convert( \
    const STADO_CONV_ESC FROM m)

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
// 4×8 → 4×compact
STADO_CONV_SAFE((SubNativeMask<8, 4>), (CompactMask<4>)) {
  return __mmask8(m.native());
}
// 4×16 → 4×compact
STADO_CONV_SAFE((SubNativeMask<16, 4>), (CompactMask<4>)) {
  return __mmask8(m.native());
}
#endif

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
// 2×32 → 2×compact
STADO_CONV_SAFE((SubNativeMask<32, 2>), (CompactMask<2>)) {
  return __mmask8(m.native());
}
#else
// 2×32 → 2×64
STADO_CONV_SAFE((SubNativeMask<32, 2>), (BroadMask<64, 2>)) {
  return _mm_unpacklo_epi32(m.native(), m.native());
}
#endif

#if STADO_INSTRUCTION_SET >= STADO_AVX2
// 4×32 → 4×64
STADO_CONV_SAFE((BroadMask<32, 4>), (BroadMask<64, 4>)) {
  return _mm256_cvtepi32_epi64(m);
}
// 4×64 → 4×32
STADO_CONV_SAFE((BroadMask<64, 4>), (BroadMask<32, 4>)) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  return _mm256_cvtepi64_epi32(m);
#else
  const auto lower = _mm256_extractf128_si256(m, 0);
  const auto upper = _mm256_extractf128_si256(m, 1);
  const auto shuffled = _mm_shuffle_ps(_mm_castsi128_ps(lower), _mm_castsi128_ps(upper), 0x88);
  return _mm_castps_si128(shuffled);
#endif
}
#endif

#undef STADO_CONV_SAFE
#undef STADO_CONV_ESC
} // namespace stado

#endif // INCLUDE_STADO_MASK_OPERATIONS_CONVERT_ELEMENTS_HPP
