#ifndef INCLUDE_STADO_MASK_OPERATIONS_LOOKUP_MASKED_HPP
#define INCLUDE_STADO_MASK_OPERATIONS_LOOKUP_MASKED_HPP

#include <array>
#include <climits>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/mask/broad.hpp"
#include "stado/mask/compact.hpp"
#include "stado/mask/single.hpp"
#include "stado/mask/subnative.hpp"
#include "stado/vector/base.hpp"
#include "stado/vector/native.hpp"
#include "stado/vector/single.hpp"
#include "stado/vector/subnative.hpp"

namespace stado {
template<typename TValue, typename TIdx, std::size_t tSize>
struct LookupMasked;

template<typename TMask, typename TIdxVec, typename TValue>
requires(TMask::size == TIdxVec::size &&
         (AnyCompactMask<TMask> || TMask::element_bits == CHAR_BIT * sizeof(TValue)) &&
         requires { sizeof(LookupMasked<TValue, typename TIdxVec::Element, TIdxVec::size>); })
inline auto lookup_masked(const TMask mask, const TIdxVec indices, const TValue* data) {
  using Out = LookupMasked<TValue, typename TIdxVec::Element, TIdxVec::size>;
  return Out::op(mask, indices, data);
}

template<typename TValue, typename TIdx, std::size_t tElementNum>
requires(AnySubNativeVector<Vector<TIdx, tElementNum>> &&
         AnySubNativeMask<Mask<sizeof(TValue) * CHAR_BIT, tElementNum>>)
struct LookupMasked<TValue, TIdx, tElementNum> {
  using IndexVec = Vector<TIdx, tElementNum>;
  using ValueMask = Mask<sizeof(TValue) * CHAR_BIT, tElementNum>;
  using Out = Vector<TValue, tElementNum>;

  static Out op(ValueMask mask, IndexVec indices, const TValue* data) {
    return Out::from_native(lookup_masked(mask.masked_native(), indices.native(), data));
  }
};

template<typename TValue, typename TIdx, std::size_t tElementNum>
requires(AnySingleVector<Vector<TIdx, tElementNum>> &&
         AnySingleMask<Mask<sizeof(TValue) * CHAR_BIT, tElementNum>>)
struct LookupMasked<TValue, TIdx, tElementNum> {
  using IndexVec = Vector<TIdx, tElementNum>;
  using ValueMask = Mask<sizeof(TValue) * CHAR_BIT, tElementNum>;
  using Out = Vector<TValue, tElementNum>;

  static Out op(ValueMask mask, IndexVec indices, const TValue* data) {
    if (mask.value()) {
      return SingleVector<TValue>{data[indices.value()]};
    }
    return SingleVector<TValue>{};
  }
};

#define STADO_LOOKMA_ESC(...) __VA_ARGS__

#define STADO_LOOKMA(VALUE, INDEX, ELEMENT_NUM) \
  template<> \
  struct LookupMasked<VALUE, INDEX, ELEMENT_NUM> { \
    using IndexVec = Vector<INDEX, ELEMENT_NUM>; \
    using ValueMask = Mask<sizeof(VALUE) * CHAR_BIT, ELEMENT_NUM>; \
    using Out = Vector<VALUE, ELEMENT_NUM>; \
\
    static inline Out op(ValueMask mask, IndexVec indices, const VALUE* data); \
  }; \
  inline Vector<VALUE, ELEMENT_NUM> LookupMasked<VALUE, INDEX, ELEMENT_NUM>::op( \
    const Mask<sizeof(VALUE) * CHAR_BIT, ELEMENT_NUM> mask, \
    const Vector<INDEX, ELEMENT_NUM> indices, const VALUE* data)

//////////////////
// 64 → 128 bit //
//////////////////

STADO_LOOKMA(double, u32, 2) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm_mmask_i32gather_pd(_mm_setzero_pd(), mask, indices.native(), data, 8);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm_mask_i32gather_pd(_mm_setzero_pd(), data, indices.native(), mask, 8);
#else
  std::array<u32, 2> ii;
  indices.store(ii.data());
  auto bits = to_bits(mask);
  return {((bits & 1U) != 0) ? data[ii[0]] : 0, ((bits & 2U) != 0) ? data[ii[1]] : 0};
#endif
}

//////////////////
// 128 → 64 bit //
//////////////////

STADO_LOOKMA(float, u64, 2) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return Out::from_register(
    _mm_mmask_i64gather_ps(_mm_setzero_ps(), mask.native(), indices, data, 4));
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  return Out::from_register(
    _mm_mask_i64gather_ps(_mm_setzero_ps(), data, indices, mask.native(), 4));
#else
  std::array<u64, 2> ii;
  indices.store(ii.data());
  auto bits = to_bits(mask.native());
  return Out::from_values(((bits & 1U) != 0) ? data[ii[0]] : 0,
                          ((bits & 2U) != 0) ? data[ii[1]] : 0);
#endif
}

/////////////
// 128 bit //
/////////////

STADO_LOOKMA(float, u32, 4) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm_mmask_i32gather_ps(_mm_setzero_ps(), mask, indices, data, 4);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm_mask_i32gather_ps(_mm_setzero_ps(), data, indices, mask, 4);
#else
  std::array<u32, 4> ii;
  indices.store(ii.data());
  auto bits = to_bits(mask);
  return {((bits & 1) != 0) ? data[ii[0]] : 0, ((bits & 2) != 0) ? data[ii[1]] : 0,
          ((bits & 4) != 0) ? data[ii[2]] : 0, ((bits & 8) != 0) ? data[ii[3]] : 0};
#endif
}

STADO_LOOKMA(u32, u32, 4) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm_mmask_i32gather_epi32(_mm_setzero_si128(), mask, indices, data, 4);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm_mask_i32gather_epi32(_mm_setzero_si128(), (const int*)data, indices, mask, 4);
#else
  std::array<u32, 4> ii;
  indices.store(ii.data());
  auto bits = to_bits(mask);
  return {((bits & 1) != 0) ? data[ii[0]] : 0, ((bits & 2) != 0) ? data[ii[1]] : 0,
          ((bits & 4) != 0) ? data[ii[2]] : 0, ((bits & 8) != 0) ? data[ii[3]] : 0};
#endif
}

STADO_LOOKMA(double, u64, 2) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm_mmask_i64gather_pd(_mm_setzero_pd(), mask, indices, data, 8);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm_mask_i64gather_pd(_mm_setzero_pd(), data, indices, mask, 8);
#else
  std::array<u64, 2> ii;
  indices.store(ii.data());
  auto bits = to_bits(mask);
  return {(bits & 1) ? data[ii[0]] : 0, (bits & 2) ? data[ii[1]] : 0};
#endif
}

STADO_LOOKMA(u64, u64, 2) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm_mmask_i64gather_epi64(_mm_setzero_si128(), mask, indices, data, 8);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm_mask_i64gather_epi64(_mm_setzero_si128(), (const long long*)data, indices, mask, 8);
#else
  std::array<u64, 2> ii;
  indices.store(ii.data());
  auto bits = to_bits(mask);
  return {((bits & 1) != 0) ? data[ii[0]] : 0, ((bits & 2) != 0) ? data[ii[1]] : 0};
#endif
}

#if STADO_INSTRUCTION_SET >= STADO_AVX2

///////////////////
// 256 → 128 bit //
///////////////////

STADO_LOOKMA(float, u64, 4) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm256_mmask_i64gather_ps(_mm_setzero_ps(), mask, indices, data, 4);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm256_mask_i64gather_ps(_mm_setzero_ps(), data, indices, mask, 4);
#endif
}

/////////////
// 256 bit //
/////////////

STADO_LOOKMA(float, u32, 8) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm256_mmask_i32gather_ps(_mm256_setzero_ps(), mask, indices, data, 4);
#else
  return _mm256_mask_i32gather_ps(_mm256_setzero_ps(), data, indices, mask, 4);
#endif
}

STADO_LOOKMA(u32, u32, 8) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm256_mmask_i32gather_epi32(_mm256_setzero_si256(), mask, indices, data, 4);
#else
  return _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), (const int*)data, indices, mask, 4);
#endif
}

STADO_LOOKMA(double, u32, 4) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm256_mmask_i32gather_pd(_mm256_setzero_pd(), mask, indices, data, 8);
#else
  return _mm256_mask_i32gather_pd(_mm256_setzero_pd(), data, indices, mask, 8);
#endif
}

STADO_LOOKMA(double, u64, 4) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm256_mmask_i64gather_pd(_mm256_setzero_pd(), mask, indices, data, 8);
#else
  return _mm256_mask_i64gather_pd(_mm256_setzero_pd(), data, indices, mask, 8);
#endif
}

STADO_LOOKMA(u64, u64, 4) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  return _mm256_mmask_i64gather_epi64(_mm256_setzero_si256(), mask, indices, data, 8);
#else
  return _mm256_mask_i64gather_epi64(_mm256_setzero_si256(), (const long long*)data, indices, mask,
                                     8);
#endif
}
#endif // AVX2

/////////////
// 512 bit //
/////////////

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
STADO_LOOKMA(float, u32, 16) {
  return _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, indices, data, 4);
}

STADO_LOOKMA(double, u32, 8) {
  return _mm512_mask_i32gather_pd(_mm512_setzero_pd(), mask, indices, data, 8);
}

STADO_LOOKMA(double, u64, 8) {
  return _mm512_mask_i64gather_pd(_mm512_setzero_pd(), mask, indices, data, 8);
}
#endif // AVX512F

// 8 and 16 bit indices

template<typename TFloat, typename TIdx, std::size_t tSize>
requires(sizeof(TIdx) <= 2)
struct LookupMasked<TFloat, TIdx, tSize> {
  using IndexVec = Vector<TIdx, tSize>;
  using ValueMask = Mask<sizeof(TFloat) * 8, tSize>;
  using Out = Vector<TFloat, tSize>;

  static Out op(ValueMask mask, IndexVec indices, const TFloat* data) {
    return LookupMasked<TFloat, u32, tSize>::op(mask, convert_safe<Vector<u32, tSize>>(indices),
                                                data);
  }
};

#undef STADO_LOOKMA
#undef STADO_LOOKMA_ESC
} // namespace stado

#endif // INCLUDE_STADO_MASK_OPERATIONS_LOOKUP_MASKED_HPP
