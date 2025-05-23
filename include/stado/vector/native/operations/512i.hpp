#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_512I_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_512I_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/compact.hpp"
#include "stado/vector/native/divisor.hpp"
#include "stado/vector/native/operations/128i.hpp"
#include "stado/vector/native/operations/256i.hpp"
#include "stado/vector/native/operations/i32x16.hpp"
#include "stado/vector/native/operations/u32x16.hpp"
#include "stado/vector/native/operations/u64x08.hpp"
#include "stado/vector/native/types.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
// Permute vector of 8 64-bit integers.
// Index -1 gives 0, index any means don't care.
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline i64x8 permute8(const i64x8 a) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  __m512i y = a; // result
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<i64x8>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm512_setzero_si512(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed
    if constexpr (flags.largeblock) { // use larger permutation
      constexpr std::array<int, 4> arr = largeblock_perm<8>(indexs); // permutation pattern
      constexpr u8 ppat =
        (arr[0] & 3) | (arr[1] << 2 & 0xC) | (arr[2] << 4 & 0x30) | (arr[3] << 6 & 0xC0);
      y = _mm512_shuffle_i64x2(a, a, ppat);
    } else if constexpr (flags.same_pattern) { // same pattern in all lanes
      if constexpr (flags.punpckh) { // fits punpckhi
        y = _mm512_unpackhi_epi64(y, y);
      } else if constexpr (flags.punpckl) { // fits punpcklo
        y = _mm512_unpacklo_epi64(y, y);
      } else { // general permute
        y = _mm512_shuffle_epi32(a, (_MM_PERM_ENUM)flags.ipattern);
      }
    } else { // different patterns in all lanes
      if constexpr (flags.rotate_big) { // fits big rotate
        constexpr u8 rot = flags.rot_count; // rotation count
        y = _mm512_alignr_epi64(y, y, rot);
      } else if constexpr (flags.broadcast) { // broadcast one element
        constexpr int e = flags.rot_count;
        if constexpr (e != 0) {
          y = _mm512_alignr_epi64(y, y, e);
        }
        y = _mm512_broadcastq_epi64(_mm512_castsi512_si128(y));
      } else if constexpr (flags.compress) {
        y = _mm512_maskz_compress_epi64(__mmask8(compress_mask(indexs)), y); // compress
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (flags.expand) {
        y = _mm512_maskz_expand_epi64(__mmask8(expand_mask(indexs)), y); // expand
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (!flags.cross_lane) { // no lane crossing. Use pshufb
        constexpr std::array<i8, 64> bm = pshufb_mask<i64x8>(indexs);
        return _mm512_shuffle_epi8(y, i64x8().load(bm.data()));
      } else {
        // full permute needed
        const __m512i pmask = _mm512_setr_epi32(i0 & 7, 0, i1 & 7, 0, i2 & 7, 0, i3 & 7, 0, i4 & 7,
                                                0, i5 & 7, 0, i6 & 7, 0, i7 & 7, 0);
        y = _mm512_permutexvar_epi64(pmask, y);
      }
    }
  }
  if constexpr (flags.zeroing) { // additional zeroing needed
    y = _mm512_maskz_mov_epi64(zero_mask<8>(indexs), y);
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline u64x8 permute8(const u64x8 a) {
  return u64x8(permute8<i0, i1, i2, i3, i4, i5, i6, i7>(i64x8(a)));
}

// Permute vector of 16 32-bit integers.
// Index -1 gives 0, index any means don't care.
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline i32x16 permute16(const i32x16 a) {
  constexpr std::array<int, 16> indexs{// indexes as array
                                       i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                       i8, i9, i10, i11, i12, i13, i14, i15};
  __m512i y = a;
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<i32x16>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm512_setzero_si512(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed

    if constexpr (flags.largeblock) { // use larger permutation
      constexpr std::array<int, 8> arr = largeblock_perm<16>(indexs); // permutation pattern
      y = permute8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(i64x8(a));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.same_pattern) { // same pattern in all lanes
      if constexpr (flags.punpckh) { // fits punpckhi
        y = _mm512_unpackhi_epi32(y, y);
      } else if constexpr (flags.punpckl) { // fits punpcklo
        y = _mm512_unpacklo_epi32(y, y);
      } else { // general permute
        y = _mm512_shuffle_epi32(a, (_MM_PERM_ENUM)flags.ipattern);
      }
    } else { // different patterns in all lanes
      if constexpr (flags.rotate_big) { // fits big rotate
        constexpr u8 rot = flags.rot_count; // rotation count
        return _mm512_maskz_alignr_epi32(zero_mask<16>(indexs), y, y, rot);
      } else if constexpr (flags.broadcast) { // broadcast one element
        constexpr int e = flags.rot_count; // element index
        if constexpr (e != 0) {
          y = _mm512_alignr_epi32(y, y, e);
        }
        y = _mm512_broadcastd_epi32(_mm512_castsi512_si128(y));
      } else if constexpr (flags.zext) {
        y = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(y)); // zero extension
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (flags.compress) {
        y = _mm512_maskz_compress_epi32(__mmask16(compress_mask(indexs)), y); // compress
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (flags.expand) {
        y = _mm512_maskz_expand_epi32(__mmask16(expand_mask(indexs)), y); // expand
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (!flags.cross_lane) { // no lane crossing. Use pshufb
        constexpr std::array<stado::i8, 64> bm = pshufb_mask<i32x16>(indexs);
        return _mm512_shuffle_epi8(a, i32x16().load(bm.data()));
      } else {
        // full permute needed
        const __m512i pmask = _mm512_setr_epi32(
          i0 & 15, i1 & 15, i2 & 15, i3 & 15, i4 & 15, i5 & 15, i6 & 15, i7 & 15, i8 & 15, i9 & 15,
          i10 & 15, i11 & 15, i12 & 15, i13 & 15, i14 & 15, i15 & 15);
        return _mm512_maskz_permutexvar_epi32(zero_mask<16>(indexs), pmask, a);
      }
    }
  }
  if constexpr (flags.zeroing) { // additional zeroing needed
    y = _mm512_maskz_mov_epi32(zero_mask<16>(indexs), y);
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline u32x16 permute16(const u32x16 a) {
  return {
    permute16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15>(i32x16(a))};
}

/*****************************************************************************
 *
 *          NativeVector blend functions
 *
 *****************************************************************************/

// permute and blend i64x8
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline i64x8 blend8(const i64x8 a, const i64x8 b) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  __m512i y = a;
  // get flags for possibilities that fit the index pattern
  constexpr u64 flags = blend_flags<i64x8>(indexs);

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm512_setzero_si512(); // just return zero
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
    y = _mm512_mask_mov_epi64(a, mb, b);
  } else if constexpr ((flags & blend_rotate_big) != 0) { // full rotate
    constexpr auto rot = u8(flags >> blend_rotpattern); // rotate count
    if constexpr (rot < 8) {
      y = _mm512_alignr_epi64(b, a, rot);
    } else {
      y = _mm512_alignr_epi64(a, b, rot & 7);
    }
  } else if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 128-bit blocks
    constexpr std::array<int, 4> arr = largeblock_perm<8>(indexs); // get 128-bit blend pattern
    constexpr u8 shuf = (arr[0] & 3) | (arr[1] & 3) << 2 | (arr[2] & 3) << 4 | (arr[3] & 3) << 6;
    if constexpr (make_bit_mask<8, 0x103>(indexs) == 0) { // fits vshufi64x2 (a,b)
      y = _mm512_shuffle_i64x2(a, b, shuf);
    } else if constexpr (make_bit_mask<8, 0x203>(indexs) == 0) { // fits vshufi64x2 (b,a)
      y = _mm512_shuffle_i64x2(b, a, shuf);
    } else {
      constexpr std::array<i64, 8> bm = perm_mask_broad<i64, 8>(indexs); // full permute
      y = _mm512_permutex2var_epi64(a, i64x8().load(bm.data()), b);
    }
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm512_unpacklo_epi64(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm512_unpacklo_epi64(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm512_unpackhi_epi64(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm512_unpackhi_epi64(b, a);
  }
#if ALLOW_FP_PERMUTE // allow floating point permute instructions on integer vectors
  else if constexpr ((flags & blend_shufab) != 0) {
    // use floating point instruction shufpd
    y = _mm512_castpd_si512(_mm512_shuffle_pd(_mm512_castsi512_pd(a), _mm512_castsi512_pd(b),
                                              u8(flags >> blend_shufpattern)));
  } else if constexpr ((flags & blend_shufba) != 0) {
    // use floating point instruction shufpd
    y = _mm512_castpd_si512(_mm512_shuffle_pd(_mm512_castsi512_pd(b), _mm512_castsi512_pd(a),
                                              u8(flags >> blend_shufpattern)));
  }
#else
  // we might use 2 x _mm512_mask(z)_shuffle_epi32 like in blend16 below
#endif
  else {
    // No special cases
    constexpr std::array<i64, 8> bm = perm_mask_broad<i64, 8>(indexs); // full permute
    y = _mm512_permutex2var_epi64(a, i64x8().load(bm.data()), b);
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
    y = _mm512_maskz_mov_epi64(zero_mask<8>(indexs), y);
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline u64x8 blend8(const u64x8 a, const u64x8 b) {
  return u64x8(blend8<i0, i1, i2, i3, i4, i5, i6, i7>(i64x8(a), i64x8(b)));
}

// permute and blend i32x16
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline i32x16 blend16(const i32x16 a, const i32x16 b) {
  constexpr std::array<int, 16> indexs{i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                       i8, i9, i10, i11, i12, i13, i14, i15}; // indexes as array
  __m512i y = a;
  // get flags for possibilities that fit the index pattern
  constexpr u64 flags = blend_flags<i32x16>(indexs);

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm512_setzero_si512(); // just return zero
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
    y = _mm512_mask_mov_epi32(a, mb, b);
  } else if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 64-bit blocks
    constexpr std::array<int, 8> arr = largeblock_perm<16>(indexs); // get 64-bit blend pattern
    y = blend8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(i64x8(a), i64x8(b));
    if (!(flags & blend_addz)) {
      return y; // no remaining zeroing
    }
  } else if constexpr ((flags & blend_same_pattern) != 0) {
    // same pattern in all 128-bit lanes. check if pattern fits special cases
    if constexpr ((flags & blend_punpcklab) != 0) {
      y = _mm512_unpacklo_epi32(a, b);
    } else if constexpr ((flags & blend_punpcklba) != 0) {
      y = _mm512_unpacklo_epi32(b, a);
    } else if constexpr ((flags & blend_punpckhab) != 0) {
      y = _mm512_unpackhi_epi32(a, b);
    } else if constexpr ((flags & blend_punpckhba) != 0) {
      y = _mm512_unpackhi_epi32(b, a);
    }
#if ALLOW_FP_PERMUTE // allow floating point permute instructions on integer vectors
    else if constexpr ((flags & blend_shufab) != 0) { // use floating point instruction shufpd
      y = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(a), _mm512_castsi512_ps(b),
                                                u8(flags >> blend_shufpattern)));
    } else if constexpr ((flags & blend_shufba) != 0) { // use floating point instruction shufpd
      y = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(b), _mm512_castsi512_ps(a),
                                                u8(flags >> blend_shufpattern)));
    }
#endif
    else {
      // Use vpshufd twice. This generates two instructions in the dependency chain,
      // but we are avoiding the slower lane-crossing instruction, and saving 64
      // bytes of data cache.
      auto shuf = [](const std::array<int, 16>& a) constexpr { // get pattern for vpshufd
        int pat[4]{-1, -1, -1, -1};
        for (int i = 0; i < 16; i++) {
          int ix = a[i];
          if (ix >= 0 && pat[i & 3] < 0) {
            pat[i & 3] = ix;
          }
        }
        return (pat[0] & 3) | (pat[1] & 3) << 2 | (pat[2] & 3) << 4 | (pat[3] & 3) << 6;
      };
      constexpr u8 pattern = u8(shuf(indexs)); // permute pattern
      constexpr auto froma = u16(make_bit_mask<16, 0x004>(indexs)); // elements from a
      constexpr auto fromb = u16(make_bit_mask<16, 0x304>(indexs)); // elements from b
      y = _mm512_maskz_shuffle_epi32(froma, a, (_MM_PERM_ENUM)pattern);
      y = _mm512_mask_shuffle_epi32(y, fromb, b, (_MM_PERM_ENUM)pattern);
      return y; // we have already zeroed any unused elements
    }
  } else if constexpr ((flags & blend_rotate_big) != 0) { // full rotate
    constexpr auto rot = u8(flags >> blend_rotpattern); // rotate count
    if constexpr (rot < 16) {
      y = _mm512_alignr_epi32(b, a, rot);
    } else {
      y = _mm512_alignr_epi32(a, b, rot & 0x0F);
    }
  } else {
    // No special cases
    constexpr std::array<i32, 16> bm = perm_mask_broad<i32, 16>(indexs); // full permute
    y = _mm512_permutex2var_epi32(a, i32x16().load(bm.data()), b);
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
    y = _mm512_maskz_mov_epi32(zero_mask<16>(indexs), y);
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline u32x16 blend16(const u32x16 a, const u32x16 b) {
  return {blend16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15>(i32x16(a),
                                                                                        i32x16(b))};
}

inline i32x16 lookup16(const i32x16 index, const i32x16 table) {
  return _mm512_permutexvar_epi32(index, table);
}

inline i32x16 lookup32(const i32x16 index, const i32x16 table1, const i32x16 table2) {
  return _mm512_permutex2var_epi32(table1, index, table2);
}

inline i32x16 lookup64(const i32x16 index, const i32x16 table1, const i32x16 table2,
                       const i32x16 table3, const i32x16 table4) {
  i32x16 d12 = _mm512_permutex2var_epi32(table1, index, table2);
  i32x16 d34 = _mm512_permutex2var_epi32(table3, index, table4);
  return select((index >> 5) != i32x16(0), d34, d12);
}

inline i32x16 lookup(const i32x16 index, const i32* table) {
  return _mm512_i32gather_epi32(index, table, 4);
  // return  _mm512_i32gather_epi32(index, table, _MM_UPCONV_EPI32_NONE, 4, 0);
}

template<std::size_t n>
inline i32x16 lookup_bounded(const i32x16 index, const i32* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 16) {
    i32x16 table1 = i32x16().load(table);
    return lookup16(index, table1);
  }
  if constexpr (n <= 32) {
    i32x16 table1 = i32x16().load(table);
    i32x16 table2 = i32x16().load(reinterpret_cast<const i8*>(table) + 64);
    return _mm512_permutex2var_epi32(table1, index, table2);
  }
  // n > 32
  return lookup(index, table);
}

inline i64x8 lookup8(const i64x8 index, const i64x8 table) {
  return _mm512_permutexvar_epi64(index, table);
}

inline i64x8 lookup(const i64x8 index, const i64* table) {
  return _mm512_i64gather_epi64(index, table, 8);
}

template<std::size_t n>
inline i64x8 lookup_bounded(const i64x8 index, const i64* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 8) {
    i64x8 table1 = i64x8().load(table);
    return lookup8(index, table1);
  }
  if constexpr (n <= 16) {
    i64x8 table1 = i64x8().load(table);
    i64x8 table2 = i64x8().load(reinterpret_cast<const i8*>(table) + 64);
    return _mm512_permutex2var_epi64(table1, index, table2);
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
inline i32x16 gather16i(const void* a) {
  constexpr std::array<int, 16> indexs{i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                       i8, i9, i10, i11, i12, i13, i14, i15};
  constexpr int imin = min_index(indexs);
  constexpr int imax = max_index(indexs);
  static_assert(imin >= 0, "Negative index in gather function");

  if constexpr (imax - imin <= 15) {
    // load one contiguous block and permute
    if constexpr (imax > 15) {
      // make sure we don't read past the end of the array
      i32x16 b = i32x16().load(reinterpret_cast<const i32*>(a) + imax - 15);
      return permute16<i0 - imax + 15, i1 - imax + 15, i2 - imax + 15, i3 - imax + 15,
                       i4 - imax + 15, i5 - imax + 15, i6 - imax + 15, i7 - imax + 15,
                       i8 - imax + 15, i9 - imax + 15, i10 - imax + 15, i11 - imax + 15,
                       i12 - imax + 15, i13 - imax + 15, i14 - imax + 15, i15 - imax + 15>(b);
    } else {
      i32x16 b = i32x16().load(reinterpret_cast<const i32*>(a) + imin);
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
    i32x16 b = i32x16().load(reinterpret_cast<const i32*>(a) + imin);
    i32x16 c = i32x16().load(reinterpret_cast<const i32*>(a) + imax - 15);
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
  return _mm512_i32gather_epi32(
    (i32x16(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15)),
    reinterpret_cast<const i32*>(a), 4);
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline i64x8 gather8q(const void* a) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  constexpr int imin = min_index(indexs);
  constexpr int imax = max_index(indexs);
  static_assert(imin >= 0, "Negative index in gather function");

  if constexpr (imax - imin <= 7) {
    // load one contiguous block and permute
    if constexpr (imax > 7) {
      // make sure we don't read past the end of the array
      i64x8 b = i64x8().load(reinterpret_cast<const i64*>(a) + imax - 7);
      return permute8<i0 - imax + 7, i1 - imax + 7, i2 - imax + 7, i3 - imax + 7, i4 - imax + 7,
                      i5 - imax + 7, i6 - imax + 7, i7 - imax + 7>(b);
    } else {
      i64x8 b = i64x8().load(reinterpret_cast<const i64*>(a) + imin);
      return permute8<i0 - imin, i1 - imin, i2 - imin, i3 - imin, i4 - imin, i5 - imin, i6 - imin,
                      i7 - imin>(b);
    }
  }
  if constexpr ((i0 < imin + 8 || i0 > imax - 8) && (i1 < imin + 8 || i1 > imax - 8) &&
                (i2 < imin + 8 || i2 > imax - 8) && (i3 < imin + 8 || i3 > imax - 8) &&
                (i4 < imin + 8 || i4 > imax - 8) && (i5 < imin + 8 || i5 > imax - 8) &&
                (i6 < imin + 8 || i6 > imax - 8) && (i7 < imin + 8 || i7 > imax - 8)) {
    // load two contiguous blocks and blend
    i64x8 b = i64x8().load(reinterpret_cast<const i64*>(a) + imin);
    i64x8 c = i64x8().load(reinterpret_cast<const i64*>(a) + imax - 7);
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
  return _mm512_i64gather_epi64((i64x8(i0, i1, i2, i3, i4, i5, i6, i7)),
                                reinterpret_cast<const i64*>(a), 8);
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
inline void scatter(const i32x16 data, void* array) {
  __m512i indx =
    _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
  CompactMask<16> mask(i0 >= 0, i1 >= 0, i2 >= 0, i3 >= 0, i4 >= 0, i5 >= 0, i6 >= 0, i7 >= 0,
                       i8 >= 0, i9 >= 0, i10 >= 0, i11 >= 0, i12 >= 0, i13 >= 0, i14 >= 0,
                       i15 >= 0);
  _mm512_mask_i32scatter_epi32(reinterpret_cast<i32*>(array), mask, indx, data, 4);
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline void scatter(const i64x8 data, void* array) {
  __m256i indx = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
  CompactMask<8> mask(i0 >= 0, i1 >= 0, i2 >= 0, i3 >= 0, i4 >= 0, i5 >= 0, i6 >= 0, i7 >= 0);
  _mm512_mask_i32scatter_epi64(reinterpret_cast<i64*>(array), mask, indx, data, 8);
}

/*****************************************************************************
 *
 *          Scatter functions with variable indexes
 *
 *****************************************************************************/

inline void scatter(const i32x16 index, u32 limit, const i32x16 data, void* destination) {
  CompactMask<16> mask = u32x16(index) < limit;
  _mm512_mask_i32scatter_epi32(reinterpret_cast<i32*>(destination), mask, index, data, 4);
}

inline void scatter(const i64x8 index, u32 limit, const i64x8 data, void* destination) {
  CompactMask<8> mask = u64x8(index) < u64(limit);
  _mm512_mask_i64scatter_epi64(reinterpret_cast<i64*>(destination), u8(mask), index, data, 8);
}

inline void scatter(const i32x8 index, u32 limit, const i64x8 data, void* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL //  __AVX512VL__
  __mmask16 mask = _mm256_cmplt_epu32_mask(index, (u32x8(limit)));
#else
  __mmask16 mask = _mm512_mask_cmplt_epu32_mask(0xFFu, _mm512_castsi256_si512(index),
                                                _mm512_castsi256_si512(u32x8(limit)));
#endif
  _mm512_mask_i32scatter_epi64(reinterpret_cast<i64*>(destination), u8(mask), index, data, 8);
}

/*****************************************************************************
 *
 *          Functions for conversion between integer sizes
 *
 *****************************************************************************/

// Extend 32-bit integers to 64-bit integers, signed and unsigned

// Function extend_low : extends the low 8 elements to 64 bits with sign extension
inline i64x8 extend_low(const i32x16 a) {
  return _mm512_cvtepi32_epi64(a.get_low());
}

// Function extend_high : extends the high 8 elements to 64 bits with sign extension
inline i64x8 extend_high(const i32x16 a) {
  return _mm512_cvtepi32_epi64(a.get_high());
}

// Function extend_low : extends the low 8 elements to 64 bits with zero extension
inline u64x8 extend_low(const u32x16 a) {
  return _mm512_cvtepu32_epi64(a.get_low());
}

// Function extend_high : extends the high 8 elements to 64 bits with zero extension
inline u64x8 extend_high(const u32x16 a) {
  return _mm512_cvtepu32_epi64(a.get_high());
}

// Compress 64-bit integers to 32-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Overflow wraps around
inline i32x16 compress(const i64x8 low, const i64x8 high) {
  i32x8 low2 = _mm512_cvtepi64_epi32(low);
  i32x8 high2 = _mm512_cvtepi64_epi32(high);
  return {low2, high2};
}
inline u32x16 compress(const u64x8 low, const u64x8 high) {
  return {compress(i64x8(low), i64x8(high))};
}

// Function compress_saturated : packs two vectors of 64-bit integers into one vector of 32-bit
// integers Signed, with saturation
inline i32x16 compress_saturated(const i64x8 low, const i64x8 high) {
  i32x8 low2 = _mm512_cvtsepi64_epi32(low);
  i32x8 high2 = _mm512_cvtsepi64_epi32(high);
  return {low2, high2};
}

// Function compress_saturated : packs two vectors of 64-bit integers into one vector of 32-bit
// integers Unsigned, with saturation
inline u32x16 compress_saturated(const u64x8 low, const u64x8 high) {
  u32x8 low2 = _mm512_cvtusepi64_epi32(low);
  u32x8 high2 = _mm512_cvtusepi64_epi32(high);
  return {low2, high2};
}

// vector operator / : divide each element by divisor

// vector of 16 32-bit signed integers
inline i32x16 operator/(const i32x16 a, const DivisorI32 d) {
  __m512i m = _mm512_broadcast_i32x4(d.getm()); // broadcast multiplier
  __m512i sgn = _mm512_broadcast_i32x4(d.getsign()); // broadcast sign of d
  __m512i t1 = _mm512_mul_epi32(a, m); // 32x32->64 bit signed multiplication of even elements of a
  __m512i t3 = _mm512_srli_epi64(a, 32); // get odd elements of a into position for multiplication
  __m512i t4 = _mm512_mul_epi32(t3, m); // 32x32->64 bit signed multiplication of odd elements
  __m512i t2 = _mm512_srli_epi64(t1, 32); // dword of even index results
  __m512i t7 = _mm512_mask_mov_epi32(t2, 0xAAAA, t4); // blend two results
  __m512i t8 = _mm512_add_epi32(t7, a); // add
  __m512i t9 = _mm512_sra_epi32(t8, d.gets1()); // shift right artihmetic
  __m512i t10 = _mm512_srai_epi32(a, 31); // sign of a
  __m512i t11 = _mm512_sub_epi32(t10, sgn); // sign of a - sign of d
  __m512i t12 = _mm512_sub_epi32(t9, t11); // + 1 if a < 0, -1 if d < 0
  return _mm512_xor_si512(t12, sgn); // change sign if divisor negative
}

// vector of 16 32-bit unsigned integers
inline u32x16 operator/(const u32x16 a, const DivisorU32 d) {
  // broadcast multiplier
  __m512i m = _mm512_broadcast_i32x4(d.getm());
  // 32x32->64 bit unsigned multiplication of even elements of a
  __m512i t1 = _mm512_mul_epu32(a, m);
  // get odd elements of a into position for multiplication
  __m512i t3 = _mm512_srli_epi64(a, 32);
  // 32x32->64 bit unsigned multiplication of odd elements
  __m512i t4 = _mm512_mul_epu32(t3, m);
  // high dword of even index results
  __m512i t2 = _mm512_srli_epi64(t1, 32);
  // blend two results
  __m512i t7 = _mm512_mask_mov_epi32(t2, 0xAAAA, t4);
  // subtract
  __m512i t8 = _mm512_sub_epi32(a, t7);
  // shift right logical
  __m512i t9 = _mm512_srl_epi32(t8, d.gets1());
  // add
  __m512i t10 = _mm512_add_epi32(t7, t9);
  // shift right logical
  return _mm512_srl_epi32(t10, d.gets2());
}

// vector operator /= : divide
inline i32x16& operator/=(i32x16& a, const DivisorI32 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline u32x16& operator/=(u32x16& a, const DivisorU32 d) {
  a = a / d;
  return a;
}

// Divide i32x16 by compile-time constant
template<i32 d>
inline i32x16 divide_by_i(const i32x16 x) {
  static_assert(d != 0, "Integer division by zero");
  if constexpr (d == 1) {
    return x;
  }
  if constexpr (d == -1) {
    return -x;
  }
  if constexpr (u32(d) == 0x80000000) {
    // avoid overflow of abs(d). return (x == 0x80000000) ? 1 : 0;
    return _mm512_maskz_set1_epi32(x == i32x16(i32(0x80000000)), 1);
  }
  // compile-time abs(d). (force compiler to treat d as 32 bits, not 64 bits)
  constexpr u32 d1 = d > 0 ? u32(d) : u32(-d);
  if constexpr ((d1 & (d1 - 1)) == 0) {
    // d1 is a power of 2. use shift
    constexpr int k = bit_scan_reverse_const(d1);
    __m512i sign;
    if constexpr (k > 1) {
      sign = _mm512_srai_epi32(x, k - 1);
    } else {
      sign = x; // k copies of sign bit
    }
    __m512i bias = _mm512_srli_epi32(sign, 32 - k); // bias = x >= 0 ? 0 : k-1
    __m512i xpbias = _mm512_add_epi32(x, bias); // x + bias
    __m512i q = _mm512_srai_epi32(xpbias, k); // (x + bias) >> k
    if (d > 0) {
      return q; // d > 0: return  q
    }
    return _mm512_sub_epi32(_mm512_setzero_epi32(), q); // d < 0: return -q
  }
  // general case
  // ceil(log2(d1)) - 1. (d1 < 2 handled by power of 2 case)
  constexpr i32 sh = bit_scan_reverse_const(u32(d1) - 1);
  constexpr i32 mult = int(1 + (u64(1) << (32 + sh)) / u32(d1) - (i64(1) << 32)); // multiplier
  const DivisorI32 div(mult, sh, d < 0 ? -1 : 0);
  return x / div;
}

// define i32x8 a / const_int(d)
template<i32 d>
inline i32x16 operator/(const i32x16 a, ConstInt<d> /*b*/) {
  return divide_by_i<d>(a);
}

// define i32x16 a / const_uint(d)
template<u32 d>
inline i32x16 operator/(const i32x16 a, ConstUint<d> /*b*/) {
  static_assert(d < 0x80000000, "Dividing signed integer by overflowing unsigned");
  return divide_by_i<i32(d)>(a); // signed divide
}

// vector operator /= : divide
template<i32 d>
inline i32x16& operator/=(i32x16& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<u32 d>
inline i32x16& operator/=(i32x16& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// Divide u32x16 by compile-time constant
template<u32 d>
inline u32x16 divide_by_ui(const u32x16 x) {
  static_assert(d != 0, "Integer division by zero");
  if constexpr (d == 1) {
    return x; // divide by 1
  }
  constexpr int b = bit_scan_reverse_const(d); // floor(log2(d))
  if constexpr ((d & (d - 1)) == 0) {
    // d is a power of 2. use shift
    return _mm512_srli_epi32(x, b); // x >> b
  }
  // general case (d > 2)
  constexpr u32 mult = u32((u64(1) << (b + 32)) / d); // multiplier = 2^(32+b) / d
  constexpr u64 rem = (u64(1) << (b + 32)) - u64(d) * mult; // remainder 2^(32+b) % d
  constexpr bool round_down = (2 * rem < d); // check if fraction is less than 0.5
  constexpr u32 mult1 = round_down ? mult : mult + 1;

  // do 32*32->64 bit unsigned multiplication and get high part of result
  const __m512i multv = _mm512_maskz_set1_epi32(0x5555, mult1); // zero-extend mult and broadcast
  __m512i t1 = _mm512_mul_epu32(x, multv); // 32x32->64 bit unsigned multiplication of even elements
  if constexpr (round_down) {
    // compensate for rounding error. (x+1)*m replaced by x*m+m to avoid overflow
    t1 = _mm512_add_epi64(t1, multv);
  }
  // high dword of result 0 and 2
  __m512i t2 = _mm512_srli_epi64(t1, 32);
  // get odd elements into position for multiplication
  __m512i t3 = _mm512_srli_epi64(x, 32);
  // 32x32->64 bit unsigned multiplication of x[1] and x[3]
  __m512i t4 = _mm512_mul_epu32(t3, multv);
  if constexpr (round_down) {
    // compensate for rounding error. (x+1)*m replaced by x*m+m to avoid overflow
    t4 = _mm512_add_epi64(t4, multv);
  }
  __m512i t7 = _mm512_mask_mov_epi32(t2, 0xAAAA, t4); // blend two results
  u32x16 q = _mm512_srli_epi32(t7, b); // shift right by b
  return q; // no overflow possible
}

// define u32x8 a / const_uint(d)
template<u32 d>
inline u32x16 operator/(const u32x16 a, ConstUint<d> /*b*/) {
  return divide_by_ui<d>(a);
}

// define u32x8 a / const_int(d)
template<i32 d>
inline u32x16 operator/(const u32x16 a, ConstInt<d> /*b*/) {
  static_assert(d >= 0, "Dividing unsigned integer by negative is ambiguous");
  return divide_by_ui<d>(a); // unsigned divide
}

// vector operator /= : divide
template<u32 d>
inline u32x16& operator/=(u32x16& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<i32 d>
inline u32x16& operator/=(u32x16& a, ConstInt<d> b) {
  a = a / b;
  return a;
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_512I_HPP
