#ifndef INCLUDE_STADO_VECTOR_OPERATIONS_SHUFFLE_HPP
#define INCLUDE_STADO_VECTOR_OPERATIONS_SHUFFLE_HPP

#include "stado/instruction-set.hpp"
#include "stado/vector/native.hpp"

namespace stado {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
inline f32x8 shuffle(u32x8 idxs, f32x8 vals) {
  return _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(_mm256_castps_si256(vals), idxs));
}

inline f64x4 shuffle(u64x4 idxs, f64x4 vals) {
  u64x4 ti2 = idxs + idxs;
  u64x4 pl1 = ti2 + u64x4{1};
  u32x8 full_idxs(ti2 | (pl1 << 32));
  return _mm256_castps_pd(shuffle(full_idxs, _mm256_castpd_ps(vals)));
}

inline f64x4 shuffle(u32x4 idxs, f64x4 vals) {
  return shuffle(_mm256_cvtepu32_epi64(idxs), vals);
}
#endif
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_OPERATIONS_SHUFFLE_HPP
