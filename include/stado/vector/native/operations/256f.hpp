#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_256F_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_256F_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/operations/128f.hpp"
#include "stado/vector/native/operations/256i.hpp"
#include "stado/vector/native/types.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX
namespace stado {
// permute vector f64x4
template<int i0, int i1, int i2, int i3>
inline f64x4 permute4(const f64x4 a) {
  constexpr std::array<int, 4> indexs{i0, i1, i2, i3}; // indexes as array
  __m256d y = a; // result
  constexpr PermFlags flags = perm_flags<f64x4>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm256_setzero_pd(); // just return zero
  }

  if constexpr (flags.largeblock) { // permute 128-bit blocks
    constexpr std::array<int, 2> arr = largeblock_perm<4>(indexs); // permutation pattern
    constexpr int j0 = arr[0];
    constexpr int j1 = arr[1];
#ifndef ZEXT_MISSING
    if constexpr (j0 == 0 && j1 == -1 && !flags.addz) { // zero extend
      return _mm256_zextpd128_pd256(_mm256_castpd256_pd128(y));
    }
    if constexpr (j0 == 1 && j1 < 0 && !flags.addz) { // extract upper part, zero extend
      return _mm256_zextpd128_pd256(_mm256_extractf128_pd(y, 1));
    }
#endif
    if constexpr (flags.perm && !flags.zeroing) {
      return _mm256_permute2f128_pd(y, y, (j0 & 1) | (j1 & 1) << 4);
    }
  }
  if constexpr (flags.perm) { // permutation needed
    if constexpr (flags.same_pattern) { // same pattern in both lanes
      if constexpr (flags.punpckh) { // fits punpckhi
        y = _mm256_unpackhi_pd(y, y);
      } else if constexpr (flags.punpckl) { // fits punpcklo
        y = _mm256_unpacklo_pd(y, y);
      } else { // general permute
        constexpr u8 mm0 = (i0 & 1) | (i1 & 1) << 1 | (i2 & 1) << 2 | (i3 & 1) << 3;
        y = _mm256_permute_pd(a, mm0); // select within same lane
      }
    }
#if STADO_INSTRUCTION_SET >= STADO_AVX2
    else if constexpr (flags.broadcast && flags.rot_count == 0) {
      y = _mm256_broadcastsd_pd(_mm256_castpd256_pd128(y)); // broadcast first element
    }
#endif
    else { // different patterns in two lanes
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
      if constexpr (flags.rotate_big) { // fits big rotate
        constexpr u8 rot = flags.rot_count; // rotation count
        constexpr u8 zm = zero_mask<4>(indexs);
        return _mm256_castsi256_pd(
          _mm256_maskz_alignr_epi64(zm, _mm256_castpd_si256(y), _mm256_castpd_si256(y), rot));
      }
#endif
      if constexpr (!flags.cross_lane) { // no lane crossing
        constexpr u8 mm0 = (i0 & 1) | (i1 & 1) << 1 | (i2 & 1) << 2 | (i3 & 1) << 3;
        y = _mm256_permute_pd(a, mm0); // select within same lane
      } else {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
        // full permute
        constexpr u8 mms = (i0 & 3) | (i1 & 3) << 2 | (i2 & 3) << 4 | (i3 & 3) << 6;
        y = _mm256_permute4x64_pd(a, mms);
#else
        // permute lanes separately
        __m256d sw = _mm256_permute2f128_pd(a, a, 1); // swap the two 128-bit lanes
        constexpr u8 mml = (i0 & 1) | (i1 & 1) << 1 | (i2 & 1) << 2 | (i3 & 1) << 3;
        __m256d y1 = _mm256_permute_pd(a, mml); // select from same lane
        __m256d y2 = _mm256_permute_pd(sw, mml); // select from opposite lane
        constexpr u64 blendm = make_bit_mask<4, 0x101>(indexs); // blend mask
        y = _mm256_blend_pd(y1, y2, u8(blendm));
#endif
      }
    }
  }
  if constexpr (flags.zeroing) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_pd(zero_mask<4>(indexs), y);
#else // use broad mask
    constexpr std::array<i64, 4> bm = zero_mask_broad<i64, 4>(indexs);
    // this does not work with STADO_INSTRUCTION_SET = STADO_AVX
    // y = _mm256_and_pd(_mm256_castsi256_pd( i64x4().load(bm.data()) ), y);
    __m256i bm1 = _mm256_loadu_si256((const __m256i*)(bm.data()));
    y = _mm256_and_pd(_mm256_castsi256_pd(bm1), y);
#endif
  }
  return y;
}

// permute vector f32x8
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline f32x8 permute8(const f32x8 a) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  __m256 y = a; // result
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<f32x8>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm256_setzero_ps(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed

    if constexpr (flags.largeblock) { // use larger permutation
      constexpr std::array<int, 4> arr = largeblock_perm<8>(indexs); // permutation pattern
      y = _mm256_castpd_ps(permute4<arr[0], arr[1], arr[2], arr[3]>(f64x4(_mm256_castps_pd(a))));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.same_pattern) { // same pattern in both lanes
      if constexpr (flags.punpckh) { // fits punpckhi
        y = _mm256_unpackhi_ps(y, y);
      } else if constexpr (flags.punpckl) { // fits punpcklo
        y = _mm256_unpacklo_ps(y, y);
      } else { // general permute, same pattern in both lanes
        y = _mm256_shuffle_ps(a, a, flags.ipattern);
      }
    }
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    else if constexpr (flags.broadcast) {
      constexpr u8 e = flags.rot_count; // broadcast one element
      if constexpr (e > 0) {
        y = _mm256_castsi256_ps(
          _mm256_alignr_epi32(_mm256_castps_si256(y), _mm256_castps_si256(y), e));
      }
      y = _mm256_broadcastss_ps(_mm256_castps256_ps128(y));
    }
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
    else if constexpr (flags.broadcast && flags.rot_count == 0) {
      y = _mm256_broadcastss_ps(_mm256_castps256_ps128(y)); // broadcast first element
    }
#endif
#if STADO_INSTRUCTION_SET >= STADO_AVX2
    else if constexpr (flags.zext) { // zero extension
      y = _mm256_castsi256_ps(
        _mm256_cvtepu32_epi64(_mm256_castsi256_si128(_mm256_castps_si256(y)))); // zero extension
      if constexpr (!flags.addz2) {
        return y;
      }
    }
#endif
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    else if constexpr (flags.compress) {
      y = _mm256_maskz_compress_ps(__mmask8(compress_mask(indexs)), y); // compress
      if constexpr (!flags.addz2) {
        return y;
      }
    } else if constexpr (flags.expand) {
      y = _mm256_maskz_expand_ps(__mmask8(expand_mask(indexs)), y); // expand
      if constexpr (!flags.addz2) {
        return y;
      }
    }
#endif
    else { // different patterns in two lanes
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
      if constexpr (flags.rotate_big) { // fits big rotate
        constexpr u8 rot = flags.rot_count; // rotation count
        y = _mm256_castsi256_ps(
          _mm256_alignr_epi32(_mm256_castps_si256(y), _mm256_castps_si256(y), rot));
      } else
#endif
        if constexpr (!flags.cross_lane) { // no lane crossing. Use vpermilps
        __m256i m =
          _mm256_setr_epi32(i0 & 3, i1 & 3, i2 & 3, i3 & 3, i4 & 3, i5 & 3, i6 & 3, i7 & 3);
        y = _mm256_permutevar_ps(a, m);
      } else {
        // full permute needed
        __m256i permmask =
          _mm256_setr_epi32(i0 & 7, i1 & 7, i2 & 7, i3 & 7, i4 & 7, i5 & 7, i6 & 7, i7 & 7);
#if STADO_INSTRUCTION_SET >= STADO_AVX2
        y = _mm256_permutevar8x32_ps(a, permmask);
#else
        // permute lanes separately
        __m256 sw = _mm256_permute2f128_ps(a, a, 1); // swap the two 128-bit lanes
        __m256 y1 = _mm256_permutevar_ps(a, permmask); // select from same lane
        __m256 y2 = _mm256_permutevar_ps(sw, permmask); // select from opposite lane
        constexpr u64 blendm = make_bit_mask<8, 0x102>(indexs); // blend mask
        y = _mm256_blend_ps(y1, y2, u8(blendm));
#endif
      }
    }
  }
  if constexpr (flags.zeroing) {
    // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_ps(zero_mask<8>(indexs), y);
#else // use broad mask
    constexpr std::array<i32, 8> bm = zero_mask_broad<i32, 8>(indexs);
    __m256i bm1 = _mm256_loadu_si256((const __m256i*)(bm.data()));
    y = _mm256_and_ps(_mm256_castsi256_ps(bm1), y);
#endif
  }
  return y;
}

// blend vectors f64x4
template<int i0, int i1, int i2, int i3>
inline f64x4 blend4(const f64x4 a, const f64x4 b) {
  constexpr std::array<int, 4> indexs{i0, i1, i2, i3}; // indexes as array
  __m256d y = a; // result
  constexpr u64 flags =
    blend_flags<f64x4>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm256_setzero_pd(); // just return zero
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute4<i0, i1, i2, i3>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    return permute4<(i0 < 0 ? i0 : i0 & 3), (i1 < 0 ? i1 : i1 & 3), (i2 < 0 ? i2 : i2 & 3),
                    (i3 < 0 ? i3 : i3 & 3)>(b);
  }
  if constexpr ((flags & (blend_perma | blend_permb)) == 0) { // no permutation, only blending
    constexpr auto mb = (u8)make_bit_mask<4, 0x302>(indexs); // blend mask
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm256_mask_mov_pd(a, mb, b);
#else // AVX
    y = _mm256_blend_pd(a, b, mb); // duplicate each bit
#endif
  } else if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 128-bit blocks
    constexpr std::array<int, 2> arr = largeblock_perm<4>(indexs); // get 128-bit blend pattern
    constexpr u8 pp = (arr[0] & 0xF) | u8(arr[1] & 0xF) << 4;
    y = _mm256_permute2f128_pd(a, b, pp);
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm256_unpacklo_pd(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm256_unpacklo_pd(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm256_unpackhi_pd(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm256_unpackhi_pd(b, a);
  } else if constexpr ((flags & blend_shufab) != 0) {
    y = _mm256_shuffle_pd(a, b, (flags >> blend_shufpattern) & 0xF);
  } else if constexpr ((flags & blend_shufba) != 0) {
    y = _mm256_shuffle_pd(b, a, (flags >> blend_shufpattern) & 0xF);
  } else {
    // No special cases
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL. use vpermi2pd
    const __m256i maskp = _mm256_setr_epi32(i0 & 7, 0, i1 & 7, 0, i2 & 7, 0, i3 & 7, 0);
    return _mm256_maskz_permutex2var_pd(zero_mask<4>(indexs), a, maskp, b);
#else // permute a and b separately, then blend.
    constexpr std::array<int, 8> arr = blend_perm_indexes<4, 0>(indexs); // get permutation indexes
    __m256d ya = permute4<arr[0], arr[1], arr[2], arr[3]>(a);
    __m256d yb = permute4<arr[4], arr[5], arr[6], arr[7]>(b);
    constexpr u8 mb = (u8)make_bit_mask<4, 0x302>(indexs); // blend mask
    y = _mm256_blend_pd(ya, yb, mb);
#endif
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_pd(zero_mask<4>(indexs), y);
#else // use broad mask
    constexpr std::array<i64, 4> bm = zero_mask_broad<i64, 4>(indexs);
    __m256i bm1 = _mm256_loadu_si256((const __m256i*)(bm.data()));
    y = _mm256_and_pd(_mm256_castsi256_pd(bm1), y);
#endif
  }
  return y;
}

// blend vectors f32x8
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline f32x8 blend8(const f32x8 a, const f32x8 b) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  __m256 y = a; // result
  // get flags for possibilities that fit the index pattern
  constexpr u64 flags = blend_flags<f32x8>(indexs);

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm256_setzero_ps(); // just return zero
  }

  // blend and permute 32-bit blocks
  if constexpr ((flags & blend_largeblock) != 0) {
    // get 32-bit blend pattern
    constexpr std::array<int, 4> l = largeblock_perm<8>(indexs);
    y = _mm256_castpd_ps(
      blend4<l[0], l[1], l[2], l[3]>(f64x4(_mm256_castps_pd(a)), f64x4(_mm256_castps_pd(b))));
    if (!(flags & blend_addz)) {
      return y; // no remaining zeroing
    }
  } else if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute8<i0, i1, i2, i3, i4, i5, i6, i7>(a);
  } else if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    constexpr std::array<int, 16> l = blend_perm_indexes<8, 2>(indexs); // get permutation indexes
    return permute8<l[8], l[9], l[10], l[11], l[12], l[13], l[14], l[15]>(b);
  } else if constexpr ((flags & (blend_perma | blend_permb)) ==
                       0) { // no permutation, only blending
    constexpr auto mb = (u8)make_bit_mask<8, 0x303>(indexs); // blend mask
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm256_mask_mov_ps(a, mb, b);
#else // AVX2
    y = _mm256_blend_ps(a, b, mb);
#endif
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm256_unpacklo_ps(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm256_unpacklo_ps(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm256_unpackhi_ps(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm256_unpackhi_ps(b, a);
  } else if constexpr ((flags & blend_shufab) != 0) { // use floating point instruction shufpd
    y = _mm256_shuffle_ps(a, b, u8(flags >> blend_shufpattern));
  } else if constexpr ((flags & blend_shufba) != 0) { // use floating point instruction shufpd
    y = _mm256_shuffle_ps(b, a, u8(flags >> blend_shufpattern));
  } else {
    // No special cases
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL. use vpermi2d
    const __m256i maskp =
      _mm256_setr_epi32(i0 & 15, i1 & 15, i2 & 15, i3 & 15, i4 & 15, i5 & 15, i6 & 15, i7 & 15);
    return _mm256_maskz_permutex2var_ps(zero_mask<8>(indexs), a, maskp, b);
#else // permute a and b separately, then blend.
    constexpr std::array<int, 16> arr = blend_perm_indexes<8, 0>(indexs); // get permutation indexes
    __m256 ya = permute8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(a);
    __m256 yb = permute8<arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(b);
    constexpr u8 mb = (u8)make_bit_mask<8, 0x303>(indexs); // blend mask
    y = _mm256_blend_ps(ya, yb, mb);
#endif
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_ps(zero_mask<8>(indexs), y);
#else // use broad mask
    constexpr std::array<i32, 8> bm = zero_mask_broad<i32, 8>(indexs);
    __m256i bm1 = _mm256_loadu_si256((const __m256i*)(bm.data()));
    y = _mm256_and_ps(_mm256_castsi256_ps(bm1), y);
#endif
  }
  return y;
}

#if STADO_INSTRUCTION_SET >= STADO_AVX2
inline f32x8 lookup8(const i32x8 index, const f32x8 table) {
  return _mm256_permutevar8x32_ps(table, index);
}

inline f32x8 lookup(const AnyInt32x8 auto index, const f32* table) {
  return _mm256_i32gather_ps(table, index, 4);
}

template<std::size_t n>
inline f32x8 lookup_bounded(const i32x8 index, const f32* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 4) {
    f32x4 table1 = f32x4().load(table);
    return {lookup4(index.get_low(), table1), lookup4(index.get_high(), table1)};
  }
  // n > 4
  return lookup(index, table);
}

inline f64x4 lookup4(const AnyInt64x4 auto index, const f64x4 table) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  return _mm256_permutexvar_pd(index, table);
#else // AVX2
  // We can't use VPERMPD because it has constant indexes,
  // vpermilpd can permute only within 128-bit lanes
  // Convert the index to fit VPERMPS
  i32x8 index1 = permute8<0, 0, 2, 2, 4, 4, 6, 6>(i32x8(index + index));
  i32x8 index2 = index1 + i32x8(_mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1));
  return _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(table), index2));
#endif
}

inline f64x4 lookup(const AnyInt32x4 auto index, const double* table) {
  return _mm256_i32gather_pd(table, index, 8);
}

inline f64x4 lookup(const AnyInt64x4 auto index, const double* table) {
  return _mm256_i64gather_pd(table, index, 8);
}

template<std::size_t n>
inline f64x4 lookup_bounded(const AnyInt64x4 auto index, const double* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 2) {
    f64x2 table1 = f64x2().load(table);
    return f64x4(lookup2(index.get_low(), table1), lookup2(index.get_high(), table1));
  }
  // n > 4
  return lookup(index, table);
}

// Load elements from array a with indices i0, i1, i2, i3, ..
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline f32x8 gather8f(const void* a) {
  return reinterpret_f(gather8i<i0, i1, i2, i3, i4, i5, i6, i7>(a));
}

// Load elements from array a with indices i0, i1, i2, i3
template<int i0, int i1, int i2, int i3>
inline f64x4 gather4d(const void* a) {
  return reinterpret_d(gather4q<i0, i1, i2, i3>(a));
}
#endif // AVX2

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline void scatter(const f32x8 data, f32* array) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL //  AVX512VL
  __m256i indx = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
  __mmask8 mask = u16((i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3) |
                      ((i4 >= 0) << 4) | ((i5 >= 0) << 5) | ((i6 >= 0) << 6) | ((i7 >= 0) << 7));
  _mm256_mask_i32scatter_ps(array, mask, indx, data, 4);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  __m512i indx = _mm512_castsi256_si512(_mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7));
  __mmask16 mask = (i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3) |
                   ((i4 >= 0) << 4) | ((i5 >= 0) << 5) | ((i6 >= 0) << 6) | ((i7 >= 0) << 7);
  _mm512_mask_i32scatter_ps(array, mask, indx, _mm512_castps256_ps512(data), 4);
#else
  const std::array<int, 8> index{i0, i1, i2, i3, i4, i5, i6, i7};
  for (std::size_t i = 0; i < 8; ++i) {
    if (index[i] >= 0) {
      array[index[i]] = data[i];
    }
  }
#endif
}

template<int i0, int i1, int i2, int i3>
inline void scatter(const f64x4 data, double* array) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL //  AVX512VL
  __m128i indx = _mm_setr_epi32(i0, i1, i2, i3);
  __mmask8 mask = (i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3);
  _mm256_mask_i32scatter_pd(array, mask, indx, data, 8);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  __m256i indx = _mm256_castsi128_si256(_mm_setr_epi32(i0, i1, i2, i3));
  __mmask16 mask = (i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3);
  _mm512_mask_i32scatter_pd(array, (__mmask8)mask, indx, _mm512_castpd256_pd512(data), 8);
#else
  const std::array<int, 4> index{i0, i1, i2, i3};
  for (std::size_t i = 0; i < 4; ++i) {
    if (index[i] >= 0) {
      array[index[i]] = data[i];
    }
  }
#endif
}

#if STADO_INSTRUCTION_SET >= STADO_AVX2
inline void scatter(const i32x8 index, u32 limit, const f32x8 data, f32* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL //  AVX512VL
  __mmask8 mask = _mm256_cmplt_epu32_mask(index, (u32x8(limit)));
  _mm256_mask_i32scatter_ps(destination, mask, index, data, 4);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  __mmask16 mask = _mm512_mask_cmplt_epu32_mask(0xFFu, _mm512_castsi256_si512(index),
                                                _mm512_castsi256_si512(u32x8(limit)));
  _mm512_mask_i32scatter_ps(destination, mask, _mm512_castsi256_si512(index),
                            _mm512_castps256_ps512(data), 4);
#else
  for (std::size_t i = 0; i < 8; ++i) {
    if (u32(index[i]) < limit) {
      destination[index[i]] = data[i];
    }
  }
#endif
}

inline void scatter(const i64x4 index, u32 limit, const f64x4 data, double* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL //  AVX512VL
  __mmask8 mask = _mm256_cmplt_epu64_mask(index, (u64x4(u64(limit))));
  _mm256_mask_i64scatter_pd(destination, mask, index, data, 8);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  __mmask16 mask = _mm512_mask_cmplt_epu64_mask(0xF, _mm512_castsi256_si512(index),
                                                _mm512_castsi256_si512(u64x4(u64(limit))));
  _mm512_mask_i64scatter_pd(destination, (__mmask8)mask, _mm512_castsi256_si512(index),
                            _mm512_castpd256_pd512(data), 8);
#else
  for (std::size_t i = 0; i < 4; ++i) {
    if (u64(index[i]) < u64(limit)) {
      destination[index[i]] = data[i];
    }
  }
#endif
}

#endif // AVX2

inline void scatter(const i32x4 index, u32 limit, const f64x4 data, double* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL //  AVX512VL
  __mmask8 mask = _mm_cmplt_epu32_mask(index, (u32x4(limit)));
  _mm256_mask_i32scatter_pd(destination, mask, index, data, 8);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  __mmask16 mask = _mm512_mask_cmplt_epu32_mask(0xF, _mm512_castsi128_si512(index),
                                                _mm512_castsi128_si512(u32x4(limit)));
  _mm512_mask_i32scatter_pd(destination, (__mmask8)mask, _mm256_castsi128_si256(index),
                            _mm512_castpd256_pd512(data), 8);
#else
  for (std::size_t i = 0; i < 4; ++i) {
    if (u32(index[i]) < limit) {
      destination[index[i]] = data[i];
    }
  }
#endif
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_256F_HPP
