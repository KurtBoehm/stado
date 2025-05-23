#ifndef INCLUDE_STADO_VECTOR_OPERATIONS_LOOKUP_PART_HPP
#define INCLUDE_STADO_VECTOR_OPERATIONS_LOOKUP_PART_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/mask/operations/part-mask.hpp"
#include "stado/vector/native/types.hpp"
#include "stado/vector/operations/lookup-masked.hpp"

namespace stado {
inline f32x4 lookup_part(const u32x4 index, const float* table, std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  const auto mask = part_mask<Mask<32, 4>>(n);
  return lookup_masked(mask, index, table);
#else
  u32 ii[4];
  index.store(ii);
  return {(n > 0) ? table[ii[0]] : 0, (n > 1) ? table[ii[1]] : 0, (n > 2) ? table[ii[2]] : 0,
          (n > 3) ? table[ii[3]] : 0};
#endif
}

inline u32x4 lookup_part(const u32x4 index, const u32* table, std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  const auto mask = part_mask<Mask<32, 4>>(n);
  return lookup_masked(mask, index, table);
#else
  u32 ii[4];
  index.store(ii);
  return {(n > 0) ? table[ii[0]] : 0, (n > 1) ? table[ii[1]] : 0, (n > 2) ? table[ii[2]] : 0,
          (n > 3) ? table[ii[3]] : 0};
#endif
}

inline f64x2 lookup_part(const u64x2 index, const double* table, std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  const auto mask = part_mask<Mask<64, 2>>(n);
  return lookup_masked(mask, index, table);
#else
  u64 ii[2];
  index.store(ii);
  return {(n > 0) ? table[ii[0]] : 0, (n > 1) ? table[ii[1]] : 0};
#endif
}

#if STADO_INSTRUCTION_SET >= STADO_AVX2
inline f32x8 lookup_part(const u32x8 index, const float* table, std::size_t n) {
  const auto mask = part_mask<Mask<32, 8>>(n);
  return lookup_masked(mask, index, table);
}

inline f64x4 lookup_part(const u32x4 index, const double* table, std::size_t n) {
  const auto mask = part_mask<Mask<64, 4>>(n);
  return lookup_masked(mask, index, table);
}

inline f64x4 lookup_part(const u64x4 index, const double* table, std::size_t n) {
  const auto mask = part_mask<Mask<64, 4>>(n);
  return lookup_masked(mask, index, table);
}
#endif // AVX2

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
inline f32x16 lookup_part(const u32x16 index, const float* table, std::size_t n) {
  return _mm512_mask_i32gather_ps(_mm512_setzero_ps(), __mmask16((1U << n) - 1), index, table, 4);
}

inline f64x8 lookup_part(const u32x8 index, const double* table, std::size_t n) {
  return _mm512_mask_i32gather_pd(_mm512_setzero_pd(), __mmask8((1U << n) - 1), index, table, 8);
}

inline f64x8 lookup_part(const u64x8 index, const double* table, std::size_t n) {
  return _mm512_mask_i64gather_pd(_mm512_setzero_pd(), __mmask8((1U << n) - 1), index, table, 8);
}
#endif // AVX512F
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_OPERATIONS_LOOKUP_PART_HPP
