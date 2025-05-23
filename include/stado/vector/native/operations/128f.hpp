#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_128F_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_128F_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/operations/f32x04.hpp"
#include "stado/vector/native/operations/f64x02.hpp"
#include "stado/vector/native/types.hpp"

namespace stado {
// permute vector f64x2
template<int i0, int i1>
inline f64x2 permute2(const f64x2 a) {
  static constexpr std::array<int, 2> indexs{i0, i1}; // indexes as array
  __m128d y = a; // result
  // get flags for possibilities that fit the permutation pattern
  static constexpr PermFlags flags = perm_flags<i64x2>(indexs);
  static_assert(!flags.outofrange, "Index out of range in permute function");
  if constexpr (flags.allzero) {
    return _mm_setzero_pd(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed
    // try to fit various instructions
    if constexpr (flags.shleft && flags.zeroing) {
      // pslldq does both permutation and zeroing. if zeroing not needed use punpckl instead
      return _mm_castsi128_pd(_mm_bslli_si128(_mm_castpd_si128(a), 8));
    }
    if constexpr (flags.shright && flags.zeroing) {
      // psrldq does both permutation and zeroing. if zeroing not needed use punpckh instead
      return _mm_castsi128_pd(_mm_bsrli_si128(_mm_castpd_si128(a), 8));
    }
    if constexpr (flags.punpckh) {
      // fits punpckhi
      y = _mm_unpackhi_pd(a, a);
    } else if constexpr (flags.punpckl) {
      // fits punpcklo
      y = _mm_unpacklo_pd(a, a);
    } else {
      // needs general permute
      y = _mm_shuffle_pd(a, a, (i0 & 1) | (i1 & 1) * 2);
    }
  }
  if constexpr (flags.zeroing) {
    // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    // use compact mask
    y = _mm_maskz_mov_pd(zero_mask<2>(indexs), y);
#else
    // use unpack to avoid using data cache
    if constexpr (i0 == -1) {
      y = _mm_unpackhi_pd(_mm_setzero_pd(), y);
    } else if constexpr (i1 == -1) {
      y = _mm_unpacklo_pd(y, _mm_setzero_pd());
    }
#endif
  }
  return y;
}

// permute vector f32x4
template<int i0, int i1, int i2, int i3>
inline f32x4 permute4(const f32x4 a) {
  static constexpr std::array<int, 4> indexs{i0, i1, i2, i3}; // indexes as array
  __m128 y = a; // result

  // get flags for possibilities that fit the permutation pattern
  static constexpr PermFlags flags = perm_flags<f32x4>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm_setzero_ps(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed
    if constexpr (flags.largeblock) {
      // use larger permutation
      static constexpr std::array<int, 2> l = largeblock_perm<4>(indexs); // permutation pattern
      y = reinterpret_f(permute2<l[0], l[1]>(f64x2(reinterpret_d(a))));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3 && STADO_INSTRUCTION_SET < STADO_AVX512SKL
    // SSSE3, but no compact mask
    else if constexpr (flags.zeroing) {
      // Do both permutation and zeroing with PSHUFB instruction
      static constexpr std::array<i8, 16> bm = pshufb_mask<i32x4>(indexs);
      return _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(a), i32x4().load(bm.data())));
    }
#endif
    else if constexpr (flags.punpckh) { // fits punpckhi
      y = _mm_unpackhi_ps(a, a);
    } else if constexpr (flags.punpckl) { // fits punpcklo
      y = _mm_unpacklo_ps(a, a);
    } else if constexpr (flags.shleft) { // fits pslldq
      y = _mm_castsi128_ps(_mm_bslli_si128(_mm_castps_si128(a), 16 - flags.rot_count));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.shright) { // fits psrldq
      y = _mm_castsi128_ps(_mm_bsrli_si128(_mm_castps_si128(a), flags.rot_count));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    }
#if STADO_INSTRUCTION_SET >= STADO_SSE3
    else if constexpr (i0 == 0 && i1 == 0 && i2 == 2 && i3 == 2) {
      return _mm_moveldup_ps(a);
    } else if constexpr (i0 == 1 && i1 == 1 && i2 == 3 && i3 == 3) {
      return _mm_movehdup_ps(a);
    }
#endif
    else { // needs general permute
      constexpr auto mask = (i0 & 3U) | (i1 & 3U) << 2U | (i2 & 3U) << 4U | (i3 & 3U) << 6U;
#if STADO_INSTRUCTION_SET >= STADO_AVX
      y = _mm_permute_ps(a, mask);
#else
      y = _mm_shuffle_ps(a, a, mask);
#endif
    }
  }
  if constexpr (flags.zeroing) {
    // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    // The mask-zero operation can be merged into the preceding instruction, whatever that is.
    // A good optimizing compiler will do this automatically.
    // I don't want to clutter all the branches above with this
    y = _mm_maskz_mov_ps(zero_mask<4>(indexs), y);
#else // use broad mask
    static constexpr std::array<i32, 4> bm = zero_mask_broad<i32, 4>(indexs);
    y = _mm_and_ps(_mm_castsi128_ps(i32x4().load(bm.data())), y);
#endif
  }
  return y;
}

/*****************************************************************************
 *
 *          NativeVector blend functions
 *
 *****************************************************************************/
// permute and blend f64x2
template<int i0, int i1>
inline f64x2 blend2(const f64x2 a, const f64x2 b) {
  static constexpr std::array<int, 2> indexs{i0, i1}; // indexes as array
  __m128d y = a; // result
  static constexpr u64 flags =
    blend_flags<f64x2>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm_setzero_pd(); // just return zero
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute2<i0, i1>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    return permute2<(i0 < 0) ? i0 : (i0 & 1), (i1 < 0) ? i1 : (i1 & 1)>(b);
  }

  if constexpr ((flags & (blend_perma | blend_permb)) == 0) { // no permutation, only blending
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm_mask_mov_pd(a, u8(make_bit_mask<2, 0x301>(indexs)), b);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
    y = _mm_blend_pd(a, b, ((i0 & 2) ? 0x01 : 0) | ((i1 & 2) ? 0x02 : 0));
#else // SSE2
    static constexpr std::array<u64, 2> bm =
      make_broad_mask<f64, 2>(make_bit_mask<2, 0x301>(indexs));
    y = selectd(_mm_castsi128_pd(i64x2().load(bm.data())), b, a);
#endif
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm_unpacklo_pd(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm_unpacklo_pd(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm_unpackhi_pd(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm_unpackhi_pd(b, a);
  } else if constexpr ((flags & blend_shufab) != 0) { // use floating point instruction shufpd
    y = _mm_shuffle_pd(a, b, (flags >> blend_shufpattern) & 3);
  } else if constexpr ((flags & blend_shufba) != 0) { // use floating point instruction shufpd
    y = _mm_shuffle_pd(b, a, (flags >> blend_shufpattern) & 3);
  } else { // No special cases. permute a and b separately, then blend.
           // This will not occur if ALLOW_FP_PERMUTE is true
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
    static constexpr bool dozero = false;
#else // SSE2
    static constexpr bool dozero = true;
#endif
    // get permutation indexes
    static constexpr std::array<int, 4> arr = blend_perm_indexes<2, (int)dozero>(indexs);
    __m128d ya = permute2<arr[0], arr[1]>(a);
    __m128d yb = permute2<arr[2], arr[3]>(b);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm_mask_mov_pd(ya, u8(make_bit_mask<2, 0x301>(indexs)), yb);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
    y = _mm_blend_pd(ya, yb, ((i0 & 2) ? 0x01 : 0) | ((i1 & 2) ? 0x02 : 0));
#else // SSE2
    return _mm_or_pd(ya, yb);
#endif
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm_maskz_mov_pd(zero_mask<2>(indexs), y);
#else // use broad mask
    static constexpr std::array<i64, 2> bm = zero_mask_broad<i64, 2>(indexs);
    y = _mm_and_pd(_mm_castsi128_pd(i64x2().load(bm.data())), y);
#endif
  }
  return y;
}

// permute and blend f32x4
template<int i0, int i1, int i2, int i3>
inline f32x4 blend4(const f32x4 a, const f32x4 b) {
  static constexpr std::array<int, 4> indexs{i0, i1, i2, i3}; // indexes as array
  __m128 y = a; // result
  static constexpr u64 flags =
    blend_flags<f32x4>(indexs); // get flags for possibilities that fit the index pattern

  static constexpr bool blendonly =
    (flags & (blend_perma | blend_permb)) == 0; // no permutation, only blending

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm_setzero_ps(); // just return zero
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute4<i0, i1, i2, i3>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    return permute4<(i0 < 0) ? i0 : i0 & 3, (i1 < 0) ? i1 : (i1 & 3), (i2 < 0) ? i2 : (i2 & 3),
                    (i3 < 0) ? i3 : (i3 & 3)>(b);
  }
  if constexpr ((flags & blend_largeblock) != 0) { // fits blending with larger block size
    static constexpr std::array<int, 2> l = largeblock_indexes<4>(indexs);
    y = _mm_castpd_ps(blend2<l[0], l[1]>(f64x2(_mm_castps_pd(a)), f64x2(_mm_castps_pd(b))));
    if constexpr ((flags & blend_addz) == 0) {
      return y; // any zeroing has been done by larger blend
    }
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm_unpacklo_ps(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm_unpacklo_ps(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm_unpackhi_ps(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm_unpackhi_ps(b, a);
  } else if constexpr ((flags & blend_shufab) != 0 &&
                       !blendonly) { // use floating point instruction shufps
    y = _mm_shuffle_ps(a, b, u8(flags >> blend_shufpattern));
  } else if constexpr ((flags & blend_shufba) != 0 &&
                       !blendonly) { // use floating point instruction shufps
    y = _mm_shuffle_ps(b, a, u8(flags >> blend_shufpattern));
  }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  else if constexpr ((flags & blend_rotateab) != 0) {
    y = _mm_castsi128_ps(
      _mm_alignr_epi8(_mm_castps_si128(a), _mm_castps_si128(b), flags >> blend_rotpattern));
  } else if constexpr ((flags & blend_rotateba) != 0) {
    y = _mm_castsi128_ps(
      _mm_alignr_epi8(_mm_castps_si128(b), _mm_castps_si128(a), flags >> blend_rotpattern));
  }
#endif
  else { // No special cases. permute a and b separately, then blend.
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
    static constexpr bool dozero = false;
#else // SSE2
    static constexpr bool dozero = true;
#endif
    // a and b permuted
    f32x4 ya = a;
    f32x4 yb = b;
    static constexpr std::array<int, 8> l =
      blend_perm_indexes<4, (int)dozero>(indexs); // get permutation indexes
    if constexpr ((flags & blend_perma) != 0 || dozero) {
      ya = permute4<l[0], l[1], l[2], l[3]>(a);
    }
    if constexpr ((flags & blend_permb) != 0 || dozero) {
      yb = permute4<l[4], l[5], l[6], l[7]>(b);
    }
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm_mask_mov_ps(ya, u8(make_bit_mask<4, 0x302>(indexs)), yb);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
    static constexpr u8 mm =
      ((i0 & 4) ? 0x01 : 0) | ((i1 & 4) ? 0x02 : 0) | ((i2 & 4) ? 0x04 : 0) | ((i3 & 4) ? 0x08 : 0);
    if constexpr (mm == 0x01) {
      y = _mm_move_ss(ya, yb);
    } else if constexpr (mm == 0x0E) {
      y = _mm_move_ss(yb, ya);
    } else {
      y = _mm_blend_ps(ya, yb, mm);
    }
#else // SSE2. dozero = true
    return _mm_or_ps(ya, yb);
#endif
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm_maskz_mov_ps(zero_mask<4>(indexs), y);
#else // use broad mask
    static constexpr std::array<i32, 4> bm = zero_mask_broad<i32, 4>(indexs);
    y = _mm_and_ps(_mm_castsi128_ps(i32x4().load(bm.data())), y);
#endif
  }
  return y;
}

/*****************************************************************************
 *
 *          NativeVector lookup functions
 *
 ******************************************************************************
 *
 * These functions use vector elements as indexes into a table.
 * The table is given as one or more vectors or as an array.
 *
 *****************************************************************************/

inline f32x4 lookup(const AnyInt32x4 auto index, const f32* table) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm_i32gather_ps(table, index, 4);
#else
  u32 ii[4];
  index.store(ii);
  return f32x4(table[ii[0]], table[ii[1]], table[ii[2]], table[ii[3]]);
#endif
}

inline f32x4 lookup4(const AnyInt32x4 auto index, const f32x4 table) {
#if STADO_INSTRUCTION_SET >= STADO_AVX
  return _mm_permutevar_ps(table, index);
#else
  f32 tt[4];
  table.store(tt);
  return lookup(index, tt);
#endif
}

inline f32x4 lookup8(const AnyInt32x4 auto index, const f32x4 table0, const f32x4 table1) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  __m256 tt = _mm256_insertf128_ps(_mm256_castps128_ps256(table0), table1, 1); // combine tables
  __m128 r = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(tt, _mm256_castsi128_si256(index)));
  return r;
#elif STADO_INSTRUCTION_SET >= STADO_AVX
  __m128 r0 = _mm_permutevar_ps(table0, index);
  __m128 r1 = _mm_permutevar_ps(table1, index);
  __m128i i4 = _mm_slli_epi32(index, 29);
  return _mm_blendv_ps(r0, r1, _mm_castsi128_ps(i4));
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
  f32x4 r0 = lookup4(index, table0);
  f32x4 r1 = lookup4(index, table1);
  __m128i i4 = _mm_slli_epi32(index, 29);
  return _mm_blendv_ps(r0, r1, _mm_castsi128_ps(i4));
#else // SSE2
  f32x4 r0 = lookup4(index, table0);
  f32x4 r1 = lookup4(index, table1);
  __m128i i4 = _mm_srai_epi32(_mm_slli_epi32(index, 29), 31);
  return selectf(_mm_castsi128_ps(i4), r1, r0);
#endif
}

template<std::size_t n>
inline f32x4 lookup_bounded(const AnyInt32x4 auto index, const f32* table) {
  if constexpr (n == 0) {
    return 0.0F;
  }
  if constexpr (n <= 4) {
    return lookup4(index, f32x4().load(table));
  }
  if constexpr (n <= 8) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
    __m256 tt = _mm256_loadu_ps(table);
    __m128 r = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(tt, _mm256_castsi128_si256(index)));
    return r;
#else // not AVX2
    return lookup8(index, f32x4().load(table), f32x4().load(table + 4));
#endif
  }
  // n > 8.
  lookup(index, table);
}

inline f64x2 lookup(const AnyInt64x2 auto index, const f64* table) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm_i64gather_pd(table, index, 8);
#else
  u64 ii[2];
  index.store(ii);
  return {table[ii[0]], table[ii[1]]};
#endif
}

inline f64x2 lookup2(const AnyInt64x2 auto index, const f64x2 table) {
#if STADO_INSTRUCTION_SET >= STADO_AVX
  return _mm_permutevar_pd(table, index + index);
#else
  f64 tt[2];
  table.store(tt);
  return lookup(index, tt);
#endif
}

inline f64x2 lookup4(const AnyInt64x2 auto index, const f64x2 table0, const f64x2 table1) {
#if STADO_INSTRUCTION_SET >= STADO_AVX
  i64x2 index2 = index + index; // index << 1
  __m128d r0 = _mm_permutevar_pd(table0, index2);
  __m128d r1 = _mm_permutevar_pd(table1, index2);
  __m128i i4 = _mm_slli_epi64(index, 62);
  return _mm_blendv_pd(r0, r1, _mm_castsi128_pd(i4));
#else
  u64 ii[2];
  f64 tt[4];
  table0.store(tt);
  table1.store(tt + 2);
  index.store(ii);
  return {tt[ii[0]], tt[ii[1]]};
#endif
}

template<std::size_t n>
inline f64x2 lookup_bounded(const AnyInt64x2 auto index, const f64* table) {
  if constexpr (n == 0) {
    return 0.0;
  }
  if constexpr (n <= 2) {
    return lookup2(index, f64x2().load(table));
  }
#if STADO_INSTRUCTION_SET < STADO_AVX2
  if constexpr (n <= 4) {
    return lookup4(index, f64x2().load(table), f64x2().load(table + 2));
  }
#endif
  // n >= 8
  lookup(index, table);
}

/*****************************************************************************
 *
 *          Gather functions with fixed indexes
 *
 *****************************************************************************/
// Load elements from array a with indices i0, i1, i2, i3
template<int i0, int i1, int i2, int i3>
inline f32x4 gather4f(const void* a) {
  return reinterpret_f(gather4i<i0, i1, i2, i3>(a));
}

// Load elements from array a with indices i0, i1
template<int i0, int i1>
inline f64x2 gather2d(const void* a) {
  return reinterpret_d(gather2q<i0, i1>(a));
}

/*****************************************************************************
 *
 *          NativeVector scatter functions
 *
 ******************************************************************************
 *
 * These functions write the elements of a vector to arbitrary positions in an
 * array in memory. Each vector element is written to an array position
 * determined by an index. An element is not written if the corresponding
 * index is out of range.
 * The indexes can be specified as constant template parameters or as an
 * integer vector.
 *
 *****************************************************************************/

template<int i0, int i1, int i2, int i3>
inline void scatter(const f32x4 data, f32* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL //  AVX512VL
  __m128i indx = _mm_setr_epi32(i0, i1, i2, i3);
  __mmask8 mask = (i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3);
  _mm_mask_i32scatter_ps(destination, mask, indx, data, 4);

#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  __m512i indx = _mm512_castsi128_si512(_mm_setr_epi32(i0, i1, i2, i3));
  __mmask16 mask = uint16_t((i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3));
  _mm512_mask_i32scatter_ps(destination, mask, indx, _mm512_castps128_ps512(data), 4);

#else
  const int index[4] = {i0, i1, i2, i3};
  for (int i = 0; i < 4; i++) {
    if (index[i] >= 0) {
      destination[index[i]] = data[i];
    }
  }
#endif
}

template<int i0, int i1>
inline void scatter(const f64x2 data, f64* destination) {
  if (i0 >= 0) {
    destination[i0] = data[0];
  }
  if (i1 >= 0) {
    destination[i1] = data[1];
  }
}

/*****************************************************************************
 *
 *          Scatter functions with variable indexes
 *
 *****************************************************************************/

inline void scatter(const i32x4 index, u32 limit, const f32x4 data, f32* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL //  AVX512VL
  __mmask8 mask = _mm_cmplt_epu32_mask(index, (u32x4(limit)));
  _mm_mask_i32scatter_ps(destination, mask, index, data, 4);
#else
  for (int i = 0; i < 4; i++) {
    if (u32(index[i]) < limit) {
      destination[index[i]] = data[i];
    }
  }
#endif
}

inline void scatter(const i64x2 index, u32 limit, const f64x2 data, f64* destination) {
  if (u64(index[0]) < u64(limit)) {
    destination[index[0]] = data[0];
  }
  if (u64(index[1]) < u64(limit)) {
    destination[index[1]] = data[1];
  }
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_128F_HPP
