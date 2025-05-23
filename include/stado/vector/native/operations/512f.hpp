#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_512F_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_512F_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact/08.hpp"
#include "stado/mask/compact/16.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/operations/128i.hpp"
#include "stado/vector/native/operations/u32x16.hpp"
#include "stado/vector/native/operations/u64x08.hpp"
#include "stado/vector/native/types/f32x16.hpp"
#include "stado/vector/native/types/f64x08.hpp"
#include "stado/vector/native/types/i32x08.hpp"
#include "stado/vector/native/types/i32x16.hpp"
#include "stado/vector/native/types/i64x08.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
// Permute vector of 8 64-bit integers.
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline f64x8 permute8(const f64x8 a) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  __m512d y = a; // result
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<f64x8>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm512_setzero_pd(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed
    if constexpr (flags.largeblock) { // use larger permutation
      constexpr std::array<int, 4> arr = largeblock_perm<8>(indexs); // permutation pattern
      constexpr u8 ppat =
        (arr[0] & 3) | (arr[1] << 2 & 0xC) | (arr[2] << 4 & 0x30) | (arr[3] << 6 & 0xC0);
      y = _mm512_shuffle_f64x2(a, a, ppat);
    } else if constexpr (flags.same_pattern) { // same pattern in all lanes
      if constexpr (flags.punpckh) { // fits punpckhi
        y = _mm512_unpackhi_pd(y, y);
      } else if constexpr (flags.punpckl) { // fits punpcklo
        y = _mm512_unpacklo_pd(y, y);
      } else { // general permute within lanes
        constexpr u8 mm0 = (i0 & 1) | (i1 & 1) << 1 | (i2 & 1) << 2 | (i3 & 1) << 3 |
                           (i4 & 1) << 4 | (i5 & 1) << 5 | (i6 & 1) << 6 | (i7 & 1) << 7;
        y = _mm512_permute_pd(a, mm0); // select within same lane
      }
    } else { // different patterns in all lanes
      if constexpr (flags.rotate_big) { // fits big rotate
        constexpr u8 rot = flags.rot_count; // rotation count
        y = _mm512_castsi512_pd(
          _mm512_alignr_epi64(_mm512_castpd_si512(y), _mm512_castpd_si512(y), rot));
      } else if constexpr (flags.broadcast) { // broadcast one element
        constexpr int e = flags.rot_count;
        if constexpr (e != 0) {
          y = _mm512_castsi512_pd(
            _mm512_alignr_epi64(_mm512_castpd_si512(y), _mm512_castpd_si512(y), e));
        }
        y = _mm512_broadcastsd_pd(_mm512_castpd512_pd128(y));
      } else if constexpr (flags.compress) {
        y = _mm512_maskz_compress_pd(__mmask8(compress_mask(indexs)), y); // compress
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (flags.expand) {
        y = _mm512_maskz_expand_pd(__mmask8(expand_mask(indexs)), y); // expand
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (!flags.cross_lane) { // no lane crossing
        if constexpr (!flags.zeroing) { // no zeroing. use vpermilps
          const __m512i pmask = _mm512_setr_epi32(i0 << 1, 0, i1 << 1, 0, i2 << 1, 0, i3 << 1, 0,
                                                  i4 << 1, 0, i5 << 1, 0, i6 << 1, 0, i7 << 1, 0);
          return _mm512_permutevar_pd(a, pmask);
        } else { // with zeroing. pshufb may be marginally better because it needs no extra zero
                 // mask
          constexpr std::array<i8, 64> bm = pshufb_mask<i64x8>(indexs);
          return _mm512_castsi512_pd(
            _mm512_shuffle_epi8(_mm512_castpd_si512(y), i64x8().load(bm.data())));
        }
      } else {
        // full permute needed
        const __m512i pmask = _mm512_setr_epi32(i0 & 7, 0, i1 & 7, 0, i2 & 7, 0, i3 & 7, 0, i4 & 7,
                                                0, i5 & 7, 0, i6 & 7, 0, i7 & 7, 0);
        y = _mm512_permutexvar_pd(pmask, y);
      }
    }
  }
  if constexpr (flags.zeroing) { // additional zeroing needed
    y = _mm512_maskz_mov_pd(zero_mask<8>(indexs), y);
  }
  return y;
}

// Permute vector of 16 32-bit integers.
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline f32x16 permute16(const f32x16 a) {
  constexpr std::array<int, 16> indexs{// indexes as array
                                       i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                       i8, i9, i10, i11, i12, i13, i14, i15};
  __m512 y = a; // result
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<f32x16>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm512_setzero_ps(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed

    if constexpr (flags.largeblock) { // use larger permutation
      constexpr std::array<int, 8> arr = largeblock_perm<16>(indexs); // permutation pattern
      y = _mm512_castpd_ps(permute8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(
        f64x8(_mm512_castps_pd(a))));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.same_pattern) { // same pattern in all lanes
      if constexpr (flags.punpckh) { // fits punpckhi
        y = _mm512_unpackhi_ps(y, y);
      } else if constexpr (flags.punpckl) { // fits punpcklo
        y = _mm512_unpacklo_ps(y, y);
      } else { // general permute within lanes
        y = _mm512_permute_ps(a, flags.ipattern);
      }
    } else { // different patterns in all lanes
      if constexpr (flags.rotate_big) { // fits big rotate
        constexpr u8 rot = flags.rot_count; // rotation count
        y = _mm512_castsi512_ps(
          _mm512_alignr_epi32(_mm512_castps_si512(y), _mm512_castps_si512(y), rot));
      } else if constexpr (flags.broadcast) { // broadcast one element
        constexpr int e = flags.rot_count; // element index
        if constexpr (e != 0) {
          y = _mm512_castsi512_ps(
            _mm512_alignr_epi32(_mm512_castps_si512(y), _mm512_castps_si512(y), e));
        }
        y = _mm512_broadcastss_ps(_mm512_castps512_ps128(y));
      } else if constexpr (flags.zext) { // zero extension
        y = _mm512_castsi512_ps(
          _mm512_cvtepu32_epi64(_mm512_castsi512_si256(_mm512_castps_si512(y))));
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (flags.compress) {
        y = _mm512_maskz_compress_ps(__mmask16(compress_mask(indexs)), y); // compress
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (flags.expand) {
        y = _mm512_maskz_expand_ps(__mmask16(expand_mask(indexs)), y); // expand
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (!flags.cross_lane) { // no lane crossing
        if constexpr (!flags.zeroing) { // no zeroing. use vpermilps
          const __m512i pmask =
            _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
          return _mm512_permutevar_ps(a, pmask);
        } else {
          // with zeroing. pshufb may be marginally better because it needs no extra zero mask
          constexpr std::array<stado::i8, 64> bm = pshufb_mask<i32x16>(indexs);
          return _mm512_castsi512_ps(
            _mm512_shuffle_epi8(_mm512_castps_si512(a), i32x16().load(bm.data())));
        }
      } else {
        // full permute needed
        const __m512i pmaskf = _mm512_setr_epi32(
          i0 & 15, i1 & 15, i2 & 15, i3 & 15, i4 & 15, i5 & 15, i6 & 15, i7 & 15, i8 & 15, i9 & 15,
          i10 & 15, i11 & 15, i12 & 15, i13 & 15, i14 & 15, i15 & 15);
        y = _mm512_permutexvar_ps(pmaskf, a);
      }
    }
  }
  if constexpr (flags.zeroing) { // additional zeroing needed
    y = _mm512_maskz_mov_ps(zero_mask<16>(indexs), y);
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline f64x8 blend8(const f64x8 a, const f64x8 b) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  __m512d y = a;
  // get flags for possibilities that fit the index pattern
  constexpr u64 flags = blend_flags<f64x8>(indexs);

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm512_setzero_pd(); // just return zero
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute8<i0, i1, i2, i3, i4, i5, i6, i7>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    constexpr std::array<int, 16> arr = blend_perm_indexes<8, 2>(indexs); // get permutation indexes
    return permute8<arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(b);
  }
  if constexpr ((flags & (blend_perma | blend_permb)) == 0) { // no permutation, only blending
    constexpr auto mb = u8(make_bit_mask<8, 0x303>(indexs)); // blend mask
    y = _mm512_mask_mov_pd(a, mb, b);
  } else if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 128-bit blocks
    constexpr std::array<int, 4> arr = largeblock_perm<8>(indexs); // get 128-bit blend pattern
    constexpr u8 shuf = (arr[0] & 3) | (arr[1] & 3) << 2 | (arr[2] & 3) << 4 | (arr[3] & 3) << 6;
    if constexpr (make_bit_mask<8, 0x103>(indexs) == 0) { // fits vshufi64x2 (a,b)
      y = _mm512_shuffle_f64x2(a, b, shuf);
    } else if constexpr (make_bit_mask<8, 0x203>(indexs) == 0) { // fits vshufi64x2 (b,a)
      y = _mm512_shuffle_f64x2(b, a, shuf);
    } else {
      constexpr std::array<i64, 8> bm = perm_mask_broad<i64, 8>(indexs);
      y = _mm512_permutex2var_pd(a, i64x8().load(bm.data()), b);
    }
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm512_unpacklo_pd(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm512_unpacklo_pd(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm512_unpackhi_pd(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm512_unpackhi_pd(b, a);
  } else if constexpr ((flags & blend_shufab) != 0) { // use floating point instruction shufpd
    y = _mm512_shuffle_pd(a, b, u8(flags >> blend_shufpattern));
  } else if constexpr ((flags & blend_shufba) != 0) { // use floating point instruction shufpd
    y = _mm512_shuffle_pd(b, a, u8(flags >> blend_shufpattern));
  } else { // No special cases
    constexpr std::array<i64, 8> bm = perm_mask_broad<i64, 8>(indexs);
    y = _mm512_permutex2var_pd(a, i64x8().load(bm.data()), b);
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
    y = _mm512_maskz_mov_pd(zero_mask<8>(indexs), y);
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline f32x16 blend16(const f32x16 a, const f32x16 b) {
  // indexes as array
  constexpr std::array<int, 16> indexs{i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                       i8, i9, i10, i11, i12, i13, i14, i15};
  __m512 y = a;
  // get flags for possibilities that fit the index pattern
  constexpr u64 flags = blend_flags<f32x16>(indexs);

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm512_setzero_ps(); // just return zero
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    // get permutation indexes
    constexpr std::array<int, 32> arr = blend_perm_indexes<16, 2>(indexs);
    return permute16<arr[16], arr[17], arr[18], arr[19], arr[20], arr[21], arr[22], arr[23],
                     arr[24], arr[25], arr[26], arr[27], arr[28], arr[29], arr[30], arr[31]>(b);
  }
  if constexpr ((flags & (blend_perma | blend_permb)) == 0) { // no permutation, only blending
    constexpr auto mb = u16(make_bit_mask<16, 0x304>(indexs)); // blend mask
    y = _mm512_mask_mov_ps(a, mb, b);
  } else if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 64-bit blocks
    constexpr std::array<int, 8> arr = largeblock_perm<16>(indexs); // get 64-bit blend pattern
    y = _mm512_castpd_ps(blend8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(
      f64x8(_mm512_castps_pd(a)), f64x8(_mm512_castps_pd(b))));
    if (!(flags & blend_addz)) {
      return y; // no remaining zeroing
    }
  } else if constexpr ((flags & blend_same_pattern) != 0) {
    // same pattern in all 128-bit lanes. check if pattern fits special cases
    if constexpr ((flags & blend_punpcklab) != 0) {
      y = _mm512_unpacklo_ps(a, b);
    } else if constexpr ((flags & blend_punpcklba) != 0) {
      y = _mm512_unpacklo_ps(b, a);
    } else if constexpr ((flags & blend_punpckhab) != 0) {
      y = _mm512_unpackhi_ps(a, b);
    } else if constexpr ((flags & blend_punpckhba) != 0) {
      y = _mm512_unpackhi_ps(b, a);
    } else if constexpr ((flags & blend_shufab) != 0) { // use floating point instruction shufpd
      y = _mm512_shuffle_ps(a, b, u8(flags >> blend_shufpattern));
    } else if constexpr ((flags & blend_shufba) != 0) { // use floating point instruction shufpd
      y = _mm512_shuffle_ps(b, a, u8(flags >> blend_shufpattern));
    } else {
      // Use vshufps twice. This generates two instructions in the dependency chain,
      // but we are avoiding the slower lane-crossing instruction, and saving 64
      // bytes of data cache.
      auto shuf = [](const std::array<int, 16>& a) constexpr { // get pattern for vpshufd
        int pat[4] = {-1, -1, -1, -1};
        for (int i = 0; i < 16; i++) {
          int ix = a[i];
          if (ix >= 0 && pat[i & 3] < 0) {
            pat[i & 3] = ix;
          }
        }
        return (pat[0] & 3) | (pat[1] & 3) << 2 | (pat[2] & 3) << 4 | (pat[3] & 3) << 6;
      };
      constexpr auto pattern = u8(shuf(indexs)); // permute pattern
      constexpr auto froma = u16(make_bit_mask<16, 0x004>(indexs)); // elements from a
      constexpr auto fromb = u16(make_bit_mask<16, 0x304>(indexs)); // elements from b
      y = _mm512_maskz_shuffle_ps(froma, a, a, pattern);
      y = _mm512_mask_shuffle_ps(y, fromb, b, b, pattern);
      return y; // we have already zeroed any unused elements
    }
  } else { // No special cases
    constexpr std::array<i32, 16> bm = perm_mask_broad<i32, 16>(indexs);
    y = _mm512_permutex2var_ps(a, i32x16().load(bm.data()), b);
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
    y = _mm512_maskz_mov_ps(zero_mask<16>(indexs), y);
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

inline f32x16 lookup16(const AnyInt32x16 auto index, const f32x16 table) {
  return _mm512_permutexvar_ps(index, table);
}

inline f32x16 lookup(const AnyInt32x16 auto index, const f32* table) {
  return _mm512_i32gather_ps(index, table, 4);
}

template<std::size_t n>
inline f32x16 lookup_bounded(const AnyInt32x16 auto index, const f32* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 16) {
    f32x16 table1 = f32x16().load(table);
    return lookup16(index, table1);
  }
  if constexpr (n <= 32) {
    f32x16 table1 = f32x16().load(table);
    f32x16 table2 = f32x16().load(table + 16);
    return _mm512_permutex2var_ps(table1, index, table2);
  }
  // n > 32
  return lookup(index, table);
}

inline f64x8 lookup8(const AnyInt64x8 auto index, const f64x8 table) {
  return _mm512_permutexvar_pd(index, table);
}

inline f64x8 lookup(const AnyInt32x8 auto index, const f64* table) {
  return _mm512_i32gather_pd(index, reinterpret_cast<const f64*>(table), 8);
}

inline f64x8 lookup(const AnyInt64x8 auto index, const f64* table) {
  return _mm512_i64gather_pd(index, reinterpret_cast<const f64*>(table), 8);
}

template<std::size_t n>
inline f64x8 lookup_bounded(const AnyInt64x8 auto index, const f64* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 8) {
    f64x8 table1 = f64x8().load(table);
    return lookup8(index, table1);
  }
  if constexpr (n <= 16) {
    f64x8 table1 = f64x8().load(table);
    f64x8 table2 = f64x8().load(table + 8);
    return _mm512_permutex2var_pd(table1, index, table2);
  }
  // n > 16
  return lookup(index, table);
}

/*****************************************************************************
 *
 *          Gather functions with fixed indexes
 *
 *****************************************************************************/
// Load elements from array a with indices i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline f32x16 gather16f(const void* a) {
  constexpr std::array<int, 16> indexs{i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                       i8, i9, i10, i11, i12, i13, i14, i15};
  constexpr int imin = min_index(indexs);
  constexpr int imax = max_index(indexs);
  static_assert(imin >= 0, "Negative index in gather function");

  if constexpr (imax - imin <= 15) {
    // load one contiguous block and permute
    if constexpr (imax > 15) {
      // make sure we don't read past the end of the array
      f32x16 b = f32x16().load(reinterpret_cast<const f32*>(a) + imax - 15);
      return permute16<i0 - imax + 15, i1 - imax + 15, i2 - imax + 15, i3 - imax + 15,
                       i4 - imax + 15, i5 - imax + 15, i6 - imax + 15, i7 - imax + 15,
                       i8 - imax + 15, i9 - imax + 15, i10 - imax + 15, i11 - imax + 15,
                       i12 - imax + 15, i13 - imax + 15, i14 - imax + 15, i15 - imax + 15>(b);
    } else {
      f32x16 b = f32x16().load(reinterpret_cast<const f32*>(a) + imin);
      return permute16<i0 - imin, i1 - imin, i2 - imin, i3 - imin, i4 - imin, i5 - imin, i6 - imin,
                       i7 - imin, i8 - imin, i9 - imin, i10 - imin, i11 - imin, i12 - imin,
                       i13 - imin, i14 - imin, i15 - imin>(b);
    }
  }
  if constexpr ((i0 < imin + 16 || i0 > imax - 16) && (i1 < imin + 16 || i1 > imax - 16) &&
                (i2 < imin + 16 || i2 > imax - 16) && (i3 < imin + 16 || i3 > imax - 16) &&
                (i4 < imin + 16 || i4 > imax - 16) && (i5 < imin + 16 || i5 > imax - 16) &&
                (i6 < imin + 16 || i6 > imax - 16) && (i7 < imin + 16 || i7 > imax - 16) &&
                (i8 < imin + 16 || i8 > imax - 16) && (i9 < imin + 16 || i9 > imax - 16) &&
                (i10 < imin + 16 || i10 > imax - 16) && (i11 < imin + 16 || i11 > imax - 16) &&
                (i12 < imin + 16 || i12 > imax - 16) && (i13 < imin + 16 || i13 > imax - 16) &&
                (i14 < imin + 16 || i14 > imax - 16) && (i15 < imin + 16 || i15 > imax - 16)) {
    // load two contiguous blocks and blend
    f32x16 b = f32x16().load(reinterpret_cast<const f32*>(a) + imin);
    f32x16 c = f32x16().load(reinterpret_cast<const f32*>(a) + imax - 15);
    const int j0 = i0 < imin + 16 ? i0 - imin : 31 - imax + i0;
    const int j1 = i1 < imin + 16 ? i1 - imin : 31 - imax + i1;
    const int j2 = i2 < imin + 16 ? i2 - imin : 31 - imax + i2;
    const int j3 = i3 < imin + 16 ? i3 - imin : 31 - imax + i3;
    const int j4 = i4 < imin + 16 ? i4 - imin : 31 - imax + i4;
    const int j5 = i5 < imin + 16 ? i5 - imin : 31 - imax + i5;
    const int j6 = i6 < imin + 16 ? i6 - imin : 31 - imax + i6;
    const int j7 = i7 < imin + 16 ? i7 - imin : 31 - imax + i7;
    const int j8 = i8 < imin + 16 ? i8 - imin : 31 - imax + i8;
    const int j9 = i9 < imin + 16 ? i9 - imin : 31 - imax + i9;
    const int j10 = i10 < imin + 16 ? i10 - imin : 31 - imax + i10;
    const int j11 = i11 < imin + 16 ? i11 - imin : 31 - imax + i11;
    const int j12 = i12 < imin + 16 ? i12 - imin : 31 - imax + i12;
    const int j13 = i13 < imin + 16 ? i13 - imin : 31 - imax + i13;
    const int j14 = i14 < imin + 16 ? i14 - imin : 31 - imax + i14;
    const int j15 = i15 < imin + 16 ? i15 - imin : 31 - imax + i15;
    return blend16<j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15>(b, c);
  }
  // use gather instruction
  return _mm512_i32gather_ps(
    (i32x16(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15)),
    reinterpret_cast<const f32*>(a), 4);
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline f64x8 gather8d(const void* a) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  constexpr int imin = min_index(indexs);
  constexpr int imax = max_index(indexs);
  static_assert(imin >= 0, "Negative index in gather function");

  if constexpr (imax - imin <= 7) {
    // load one contiguous block and permute
    if constexpr (imax > 7) {
      // make sure we don't read past the end of the array
      f64x8 b = f64x8().load(reinterpret_cast<const f64*>(a) + imax - 7);
      return permute8<i0 - imax + 7, i1 - imax + 7, i2 - imax + 7, i3 - imax + 7, i4 - imax + 7,
                      i5 - imax + 7, i6 - imax + 7, i7 - imax + 7>(b);
    } else {
      f64x8 b = f64x8().load(reinterpret_cast<const f64*>(a) + imin);
      return permute8<i0 - imin, i1 - imin, i2 - imin, i3 - imin, i4 - imin, i5 - imin, i6 - imin,
                      i7 - imin>(b);
    }
  }
  if constexpr ((i0 < imin + 8 || i0 > imax - 8) && (i1 < imin + 8 || i1 > imax - 8) &&
                (i2 < imin + 8 || i2 > imax - 8) && (i3 < imin + 8 || i3 > imax - 8) &&
                (i4 < imin + 8 || i4 > imax - 8) && (i5 < imin + 8 || i5 > imax - 8) &&
                (i6 < imin + 8 || i6 > imax - 8) && (i7 < imin + 8 || i7 > imax - 8)) {
    // load two contiguous blocks and blend
    f64x8 b = f64x8().load(reinterpret_cast<const f64*>(a) + imin);
    f64x8 c = f64x8().load(reinterpret_cast<const f64*>(a) + imax - 7);
    const int j0 = i0 < imin + 8 ? i0 - imin : 15 - imax + i0;
    const int j1 = i1 < imin + 8 ? i1 - imin : 15 - imax + i1;
    const int j2 = i2 < imin + 8 ? i2 - imin : 15 - imax + i2;
    const int j3 = i3 < imin + 8 ? i3 - imin : 15 - imax + i3;
    const int j4 = i4 < imin + 8 ? i4 - imin : 15 - imax + i4;
    const int j5 = i5 < imin + 8 ? i5 - imin : 15 - imax + i5;
    const int j6 = i6 < imin + 8 ? i6 - imin : 15 - imax + i6;
    const int j7 = i7 < imin + 8 ? i7 - imin : 15 - imax + i7;
    return blend8<j0, j1, j2, j3, j4, j5, j6, j7>(b, c);
  }
  // use gather instruction
  return _mm512_i64gather_pd((i64x8(i0, i1, i2, i3, i4, i5, i6, i7)),
                             reinterpret_cast<const f64*>(a), 8);
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

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline void scatter(const f32x16 data, f32* array) {
  __m512i indx =
    _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
  CompactMask<16> mask(i0 >= 0, i1 >= 0, i2 >= 0, i3 >= 0, i4 >= 0, i5 >= 0, i6 >= 0, i7 >= 0,
                       i8 >= 0, i9 >= 0, i10 >= 0, i11 >= 0, i12 >= 0, i13 >= 0, i14 >= 0,
                       i15 >= 0);
  _mm512_mask_i32scatter_ps(array, mask, indx, data, 4);
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline void scatter(const f64x8 data, f64* array) {
  __m256i indx = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
  CompactMask<8> mask(i0 >= 0, i1 >= 0, i2 >= 0, i3 >= 0, i4 >= 0, i5 >= 0, i6 >= 0, i7 >= 0);
  _mm512_mask_i32scatter_pd(array, mask, indx, data, 8);
}

/*****************************************************************************
 *
 *          Scatter functions with variable indexes
 *
 *****************************************************************************/

inline void scatter(const i32x16 index, u32 limit, const f32x16 data, f32* destination) {
  CompactMask<16> mask = u32x16(index) < limit;
  _mm512_mask_i32scatter_ps(destination, mask, index, data, 4);
}

inline void scatter(const i64x8 index, u32 limit, const f64x8 data, f64* destination) {
  CompactMask<8> mask = u64x8(index) < u64(limit);
  _mm512_mask_i64scatter_pd(destination, u8(mask), index, data, 8);
}

inline void scatter(const i32x8 index, u32 limit, const f64x8 data, f64* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // __AVX512VL__, __AVX512DQ__
  __mmask8 mask = _mm256_cmplt_epu32_mask(index, (u32x8(limit)));
#else
  __mmask16 mask =
    _mm512_cmplt_epu32_mask(_mm512_castsi256_si512(index), _mm512_castsi256_si512(u32x8(limit)));
#endif
  _mm512_mask_i32scatter_pd(destination, (__mmask8)mask, index, data, 8);
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_512F_HPP
