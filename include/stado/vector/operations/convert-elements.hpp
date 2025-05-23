#ifndef INCLUDE_STADO_VECTOR_OPERATIONS_CONVERT_ELEMENTS_HPP
#define INCLUDE_STADO_VECTOR_OPERATIONS_CONVERT_ELEMENTS_HPP

#include <climits>
#include <concepts>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native.hpp"
#include "stado/vector/single.hpp"
#include "stado/vector/subnative.hpp"
#include "stado/vector/supernative.hpp"

namespace stado {
#define STADO_CONV_ESC(...) __VA_ARGS__

#define STADO_CONV(FROM, TO, SAFE) \
  template<> \
  struct ElementConvertTrait<STADO_CONV_ESC FROM, STADO_CONV_ESC TO> { \
    using From = STADO_CONV_ESC FROM; \
    using To = STADO_CONV_ESC TO; \
    static constexpr bool is_safe = SAFE; \
    static inline STADO_CONV_ESC TO convert(STADO_CONV_ESC FROM v); \
  }; \
  STADO_CONV_ESC TO ElementConvertTrait<STADO_CONV_ESC FROM, STADO_CONV_ESC TO>::convert( \
    const STADO_CONV_ESC FROM v)

#define STADO_CONV_SAFE(FROM, TO) STADO_CONV(FROM, TO, true)
#define STADO_CONV_UNSAFE(FROM, TO) STADO_CONV(FROM, TO, false)

// 64 → 128 bit
STADO_CONV_SAFE((SubNativeVector<u8, 8>), (u16x8)) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_cvtepu8_epi16(v.native());
#else // SSE 2
  return _mm_unpacklo_epi8(v.native(), _mm_setzero_si128());
#endif
}
STADO_CONV_SAFE((SubNativeVector<u16, 4>), (u32x4)) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_cvtepu16_epi32(v.native());
#else // SSE 2
  return _mm_unpacklo_epi16(v.native(), _mm_setzero_si128());
#endif
}
STADO_CONV_SAFE((SubNativeVector<u32, 2>), (u64x2)) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_cvtepu32_epi64(v.native());
#else // SSE 2
  return _mm_unpacklo_epi32(v.native(), _mm_setzero_si128());
#endif
}
STADO_CONV_SAFE((SubNativeVector<f32, 2>), (f64x2)) {
  return _mm_cvtps_pd(v.native());
}

// 32 → 128 bit
STADO_CONV_SAFE((SubNativeVector<u8, 4>), (u32x4)) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_cvtepu8_epi32(v.native());
#else // SSE 2
  const auto v16 = _mm_unpacklo_epi8(v.native(), _mm_setzero_si128());
  return _mm_unpacklo_epi16(v16, _mm_setzero_si128());
#endif
}
STADO_CONV_SAFE((SubNativeVector<u16, 2>), (u64x2)) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_cvtepu16_epi64(v.native());
#else // SSE 2
  const auto v32 = _mm_unpacklo_epi16(v.native(), _mm_setzero_si128());
  return _mm_unpacklo_epi32(v32, _mm_setzero_si128());
#endif
}

// 16 → 128 bit
STADO_CONV_SAFE((SubNativeVector<u8, 2>), (u64x2)) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  return _mm_cvtepu8_epi64(v.native());
#else // SSE 2
  const auto v16 = _mm_unpacklo_epi8(v.native(), _mm_setzero_si128());
  const auto v32 = _mm_unpacklo_epi16(v16, _mm_setzero_si128());
  return _mm_unpacklo_epi32(v32, _mm_setzero_si128());
#endif
}

// 128 → 64 bit
STADO_CONV_UNSAFE((f64x2), (SubNativeVector<f32, 2>)) {
  return To::from_register(_mm_cvtpd_ps(v));
}

#if STADO_INSTRUCTION_SET >= STADO_AVX2
// 128 → 256 Bit
STADO_CONV_SAFE((u8x16), (u16x16)) {
  return _mm256_cvtepu8_epi16(v);
}
STADO_CONV_SAFE((i8x16), (i16x16)) {
  return _mm256_cvtepi8_epi16(v);
}
STADO_CONV_SAFE((u16x8), (u32x8)) {
  return _mm256_cvtepu16_epi32(v);
}
STADO_CONV_SAFE((i16x8), (i32x8)) {
  return _mm256_cvtepi16_epi32(v);
}
STADO_CONV_SAFE((u32x4), (u64x4)) {
  return _mm256_cvtepu32_epi64(v);
}
STADO_CONV_SAFE((i32x4), (i64x4)) {
  return _mm256_cvtepi32_epi64(v);
}
// 64 → 256 bit
STADO_CONV_SAFE((SubNativeVector<u8, 8>), (u32x8)) {
  return _mm256_cvtepu8_epi32(v.native());
}
STADO_CONV_SAFE((SubNativeVector<u16, 4>), (u64x4)) {
  return _mm256_cvtepu16_epi64(v.native());
}
// 32 → 256 bit
STADO_CONV_SAFE((SubNativeVector<u8, 4>), (u64x4)) {
  return _mm256_cvtepu8_epi64(v.native());
}
#else
// 128 → 256 bit
STADO_CONV_SAFE((f32x4), (SuperNativeVector<f64, 4>)) {
  return {_mm_cvtps_pd(v), _mm_cvtps_pd(_mm_movehl_ps(v, v))};
}
#endif

#if STADO_INSTRUCTION_SET >= STADO_AVX
// 128 → 256 bit
STADO_CONV_SAFE((f32x4), (f64x4)) {
  return _mm256_cvtps_pd(v);
}
STADO_CONV_UNSAFE((f64x4), (f32x4)) {
  return _mm256_cvtpd_ps(v);
}
#endif

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
// 256 → 512 bit
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
STADO_CONV_SAFE((u8x32), (u16x32)) {
  return _mm512_cvtepu8_epi16(v);
}
STADO_CONV_SAFE((i8x32), (i16x32)) {
  return _mm512_cvtepi8_epi16(v);
}
#endif
STADO_CONV_SAFE((u16x16), (u32x16)) {
  return _mm512_cvtepu16_epi32(v);
}
STADO_CONV_SAFE((i16x16), (i32x16)) {
  return _mm512_cvtepi16_epi32(v);
}

STADO_CONV_SAFE((u32x8), (u64x8)) {
  return _mm512_cvtepu32_epi64(v);
}
STADO_CONV_SAFE((i32x8), (i64x8)) {
  return _mm512_cvtepi32_epi64(v);
}
// 128 → 512 bit
STADO_CONV_SAFE((u8x16), (u32x16)) {
  return _mm512_cvtepu8_epi32(v);
}
STADO_CONV_SAFE((u16x8), (u64x8)) {
  return _mm512_cvtepu16_epi64(v);
}
// 64 → 512 Bit
STADO_CONV_SAFE((SubNativeVector<u8, 8>), (u64x8)) {
  return _mm512_cvtepu8_epi64(v.native());
}
STADO_CONV_SAFE((f32x8), (f64x8)) {
  return _mm512_cvtps_pd(v);
}
STADO_CONV_UNSAFE((f64x8), (f32x8)) {
  return _mm512_cvtpd_ps(v);
}
#endif

#undef STADO_CONV_SAFE
#undef STADO_CONV
#undef STADO_CONV_ESC

template<typename TFrom, typename TTo>
requires(!std::same_as<TFrom, TTo>)
struct ElementConvertTrait<SingleVector<TFrom>, SingleVector<TTo>> {
  static constexpr bool is_safe = requires(TFrom v) { TTo{v}; };
  static SingleVector<TTo> convert(SingleVector<TFrom> v) {
    return SingleVector<TTo>(v.value());
  }
};

template<typename T1, std::size_t tElementNum1, typename T2, std::size_t tElementNum2>
requires(!std::same_as<T1, T2>)
struct ElementConvertTrait<SubNativeVector<T1, tElementNum1>, SubNativeVector<T2, tElementNum2>> {
  static constexpr std::size_t extended_num = 128 / (CHAR_BIT * sizeof(T2));
  using From = SubNativeVector<T1, tElementNum1>;
  using FromEx = SubNativeVector<T1, extended_num>;
  using To = SubNativeVector<T2, tElementNum2>;
  using ToEx = NativeVector<T2, extended_num>;
  static constexpr bool is_safe = ElementConvertTrait<FromEx, ToEx>::is_safe;

  static To convert(From v) {
    const auto vex = FromEx::from_native(v.native());
    const auto native = ElementConvertTrait<FromEx, ToEx>::convert(vex);
    return To::from_native(native);
  }
};
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_OPERATIONS_CONVERT_ELEMENTS_HPP
