#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_256I_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_256I_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/divisor.hpp"
#include "stado/vector/native/operations/128i.hpp"
#include "stado/vector/native/operations/i16x16.hpp"
#include "stado/vector/native/operations/i32x08.hpp"
#include "stado/vector/native/operations/i64x04.hpp"
#include "stado/vector/native/operations/u16x16.hpp"
#include "stado/vector/native/operations/u64x04.hpp"
#include "stado/vector/native/types.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
// Permute vector of 4 64-bit integers.
template<int i0, int i1, int i2, int i3>
inline i64x4 permute4(const i64x4 a) {
  constexpr std::array<int, 4> indexs{i0, i1, i2, i3}; // indexes as array
  __m256i y = a; // result
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<i64x4>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm256_setzero_si256(); // just return zero
  }

  if constexpr (flags.largeblock) { // permute 128-bit blocks
    constexpr std::array<int, 2> arr = largeblock_perm<4>(indexs); // get 128-bit permute pattern
    constexpr int j0 = arr[0];
    constexpr int j1 = arr[1];
#ifndef ZEXT_MISSING
    if constexpr (j0 == 0 && j1 == -1 && !flags.addz) { // zero extend
      return _mm256_zextsi128_si256(_mm256_castsi256_si128(y));
    }
    if constexpr (j0 == 1 && j1 < 0 && !flags.addz) { // extract upper part, zero extend
      return _mm256_zextsi128_si256(_mm256_extracti128_si256(y, 1));
    }
#endif
    if constexpr (flags.perm && !flags.zeroing) {
      return _mm256_permute2x128_si256(y, y, (j0 & 1) | (j1 & 1) << 4);
    }
  }
  if constexpr (flags.perm) { // permutation needed
    if constexpr (flags.same_pattern) { // same pattern in both lanes
      // try to fit various instructions
      if constexpr (flags.punpckh) { // fits punpckhi
        y = _mm256_unpackhi_epi64(y, y);
      } else if constexpr (flags.punpckl) { // fits punpcklo
        y = _mm256_unpacklo_epi64(y, y);
      } else { // general permute
        y = _mm256_shuffle_epi32(a, flags.ipattern);
      }
    } else if constexpr (flags.broadcast && flags.rot_count == 0) {
      y = _mm256_broadcastq_epi64(_mm256_castsi256_si128(y)); // broadcast first element
    } else { // different patterns in two lanes
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
      if constexpr (flags.rotate_big) { // fits big rotate
        constexpr u8 rot = flags.rot_count; // rotation count
        return _mm256_maskz_alignr_epi64(zero_mask<4>(indexs), y, y, rot);
      } else { // full permute
        constexpr u8 mms = (i0 & 3) | (i1 & 3) << 2 | (i2 & 3) << 4 | (i3 & 3) << 6;
        constexpr __mmask8 mmz =
          zero_mask<4>(indexs); //(i0 >= 0) | (i1 >= 0) << 1 | (i2 >= 0) << 2 | (i3 >= 0) << 3;
        return _mm256_maskz_permutex_epi64(mmz, a, mms);
      }
#else
      // full permute
      constexpr int ms = (i0 & 3) | (i1 & 3) << 2 | (i2 & 3) << 4 | (i3 & 3) << 6;
      y = _mm256_permute4x64_epi64(a, ms);
#endif
    }
  }
  if constexpr (flags.zeroing) {
    // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_epi64(zero_mask<4>(indexs), y);
#else // use broad mask
    constexpr std::array<i64, 4> bm = zero_mask_broad<i64, 4>(indexs);
    y = _mm256_and_si256(i64x4().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3>
inline u64x4 permute4(const u64x4 a) {
  return u64x4(permute4<i0, i1, i2, i3>(i64x4(a)));
}

// Permute vector of 8 32-bit integers.
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline i32x8 permute8(const i32x8 a) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  __m256i y = a; // result
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<i32x8>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm256_setzero_si256(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed

    if constexpr (flags.largeblock) { // use larger permutation
      constexpr std::array<int, 4> arr = largeblock_perm<8>(indexs); // permutation pattern
      y = permute4<arr[0], arr[1], arr[2], arr[3]>(i64x4(a));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.same_pattern) { // same pattern in both lanes
      // try to fit various instructions
      if constexpr (flags.punpckh) { // fits punpckhi
        y = _mm256_unpackhi_epi32(y, y);
      } else if constexpr (flags.punpckl) { // fits punpcklo
        y = _mm256_unpacklo_epi32(y, y);
      } else { // general permute
        y = _mm256_shuffle_epi32(a, flags.ipattern);
      }
    }
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    else if constexpr (flags.broadcast && !flags.zeroing) {
      constexpr u8 e = flags.rot_count; // broadcast one element
      if constexpr (e > 0) {
        y = _mm256_alignr_epi32(y, y, e);
      }
      return _mm256_broadcastd_epi32(_mm256_castsi256_si128(y));
#else
    else if constexpr (flags.broadcast && !flags.zeroing && flags.rot_count == 0) {
      return _mm256_broadcastd_epi32(_mm256_castsi256_si128(y)); // broadcast first element
#endif
    } else if constexpr (flags.zext) {
      y = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(y)); // zero extension
      if constexpr (!flags.addz2) {
        return y;
      }
    }
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    else if constexpr (flags.compress) {
      y = _mm256_maskz_compress_epi32(__mmask8(compress_mask(indexs)), y); // compress
      if constexpr (!flags.addz2) {
        return y;
      }
    } else if constexpr (flags.expand) {
      y = _mm256_maskz_expand_epi32(__mmask8(expand_mask(indexs)), y); // expand
      if constexpr (!flags.addz2) {
        return y;
      }
    }
#endif
    else { // different patterns in two lanes
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
      if constexpr (flags.rotate_big) { // fits big rotate
        constexpr u8 rot = flags.rot_count; // rotation count
        return _mm256_maskz_alignr_epi32(zero_mask<8>(indexs), y, y, rot);
      } else
#endif
        if constexpr (!flags.cross_lane) { // no lane crossing. Use pshufb
        constexpr std::array<i8, 32> bm = pshufb_mask<i32x8>(indexs);
        return _mm256_shuffle_epi8(a, i32x8().load(bm.data()));
      }
      // full permute needed
      __m256i permmask =
        _mm256_setr_epi32(i0 & 7, i1 & 7, i2 & 7, i3 & 7, i4 & 7, i5 & 7, i6 & 7, i7 & 7);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
      return _mm256_maskz_permutexvar_epi32(zero_mask<8>(indexs), permmask, y);
#else
      y = _mm256_permutevar8x32_epi32(y, permmask);
#endif
    }
  }
  if constexpr (flags.zeroing) {
    // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_epi32(zero_mask<8>(indexs), y);
#else // use broad mask
    constexpr std::array<i32, 8> bm = zero_mask_broad<i32, 8>(indexs);
    y = _mm256_and_si256(i32x8().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline u32x8 permute8(const u32x8 a) {
  return u32x8(permute8<i0, i1, i2, i3, i4, i5, i6, i7>(i32x8(a)));
}

// Permute vector of 16 16-bit integers.
// Index -1 gives 0, index any means don't care.
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline i16x16 permute16(const i16x16 a) {
  constexpr std::array<int, 16> indexs{// indexes as array
                                       i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                       i8, i9, i10, i11, i12, i13, i14, i15};
  __m256i y = a; // result
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<i16x16>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm256_setzero_si256(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed

    if constexpr (flags.largeblock) { // use larger permutation
      constexpr std::array<int, 8> arr = largeblock_perm<16>(indexs); // permutation pattern
      y = permute8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(i32x8(a));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.same_pattern) { // same pattern in both lanes
      // try to fit various instructions
      if constexpr (flags.punpckh) { // fits punpckhi
        y = _mm256_unpackhi_epi16(y, y);
      } else if constexpr (flags.punpckl) { // fits punpcklo
        y = _mm256_unpacklo_epi16(y, y);
      } else if constexpr (flags.rotate) { // fits palignr. rotate within lanes
        y = _mm256_alignr_epi8(a, a, flags.rot_count);
      } else {
        // flags for 16 bit permute instructions
        constexpr u64 flags16 = perm16_flags<i16x16>(indexs);
        constexpr bool l2l = (flags16 & 1) != 0; // from low  to low  64-bit part
        constexpr bool h2h = (flags16 & 2) != 0; // from high to high 64-bit part
        constexpr bool h2l = (flags16 & 4) != 0; // from high to low  64-bit part
        constexpr bool l2h = (flags16 & 8) != 0; // from low  to high 64-bit part
        constexpr u8 pl2l = u8(flags16 >> 32); // low  to low  permute pattern
        constexpr u8 ph2h = u8(flags16 >> 40); // high to high permute pattern
        constexpr u8 noperm = 0xE4; // pattern for no permute
        if constexpr (!h2l && !l2h) { // simple case. no crossing of 64-bit boundary
          if constexpr (l2l && pl2l != noperm) {
            y = _mm256_shufflelo_epi16(y, pl2l); // permute low 64-bits
          }
          if constexpr (h2h && ph2h != noperm) {
            y = _mm256_shufflehi_epi16(y, ph2h); // permute high 64-bits
          }
        } else { // use pshufb
          constexpr std::array<stado::i8, 32> bm = pshufb_mask<i16x16>(indexs);
          return _mm256_shuffle_epi8(a, i16x16().load(bm.data()));
        }
      }
    } else { // different patterns in two lanes
      if constexpr (flags.zext) { // fits zero extension
        y = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(y)); // zero extension
        if constexpr (!flags.addz2) {
          return y;
        }
      }
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI2__)
      else if constexpr (flags.compress) {
        y = _mm256_maskz_compress_epi16(__mmask16(compress_mask(indexs)), y); // compress
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (flags.expand) {
        y = _mm256_maskz_expand_epi16(__mmask16(expand_mask(indexs)), y); // expand
        if constexpr (!flags.addz2) {
          return y;
        }
      }
#endif // AVX512VBMI2
      else if constexpr (!flags.cross_lane) { // no lane crossing. Use pshufb
        constexpr std::array<stado::i8, 32> bm = pshufb_mask<i16x16>(indexs);
        return _mm256_shuffle_epi8(a, i16x16().load(bm.data()));
      } else if constexpr (flags.rotate_big) { // fits full rotate
        constexpr u8 rot = flags.rot_count * 2; // rotate count
        __m256i swap = _mm256_permute4x64_epi64(a, 0x4E); // swap 128-bit halves
        if (rot <= 16) {
          y = _mm256_alignr_epi8(swap, y, rot);
        } else {
          y = _mm256_alignr_epi8(y, swap, rot & 15);
        }
      } else if constexpr (flags.broadcast && flags.rot_count == 0) {
        y = _mm256_broadcastw_epi16(_mm256_castsi256_si128(y)); // broadcast first element
      } else { // full permute needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
        constexpr std::array<i16, 16> bm = perm_mask_broad<i16, 16>(indexs);
        y = _mm256_permutexvar_epi16(i16x16().load(bm.data()), y);
#else // no full permute instruction available
        __m256i swap = _mm256_permute4x64_epi64(y, 0x4E); // swap high and low 128-bit lane
        constexpr std::array<stado::i8, 32> bm1 = pshufb_mask<i16x16, 1>(indexs);
        constexpr std::array<stado::i8, 32> bm2 = pshufb_mask<i16x16, 0>(indexs);
        __m256i r1 = _mm256_shuffle_epi8(swap, i16x16().load(bm1.data()));
        __m256i r2 = _mm256_shuffle_epi8(y, i16x16().load(bm2.data()));
        return _mm256_or_si256(r1, r2);
#endif
      }
    }
  }
  if constexpr (flags.zeroing) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_epi16(zero_mask<16>(indexs), y);
#else // use broad mask
    constexpr std::array<i16, 16> bm = zero_mask_broad<i16, 16>(indexs);
    y = _mm256_and_si256(i16x16().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline u16x16 permute16(const u16x16 a) {
  return u16x16(
    permute16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15>(i16x16(a)));
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15, int i16, int i17, int i18, int i19, int i20,
         int i21, int i22, int i23, int i24, int i25, int i26, int i27, int i28, int i29, int i30,
         int i31>
inline i8x32 permute32(const i8x32 a) {
  constexpr std::array<int, 32> indexs{// indexes as array
                                       i0,  i1,  i2,  i3,  i4,  i5,  i6,  i7,  i8,  i9,  i10,
                                       i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21,
                                       i22, i23, i24, i25, i26, i27, i28, i29, i30, i31};

  __m256i y = a; // result
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<i8x32>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm256_setzero_si256(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed

    if constexpr (flags.largeblock) { // use larger permutation
      constexpr std::array<int, 16> arr = largeblock_perm<32>(indexs); // permutation pattern
      y = permute16<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
                    arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(i16x16(a));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.same_pattern) { // same pattern in both lanes
      if constexpr (flags.punpckh) { // fits punpckhi
        y = _mm256_unpackhi_epi8(y, y);
      } else if constexpr (flags.punpckl) { // fits punpcklo
        y = _mm256_unpacklo_epi8(y, y);
      } else if constexpr (flags.rotate) { // fits palignr. rotate within lanes
        y = _mm256_alignr_epi8(a, a, flags.rot_count);
      } else { // use pshufb
        constexpr std::array<stado::i8, 32> bm = pshufb_mask<i8x32>(indexs);
        return _mm256_shuffle_epi8(a, i8x32().load(bm.data()));
      }
    } else { // different patterns in two lanes
      if constexpr (flags.zext) { // fits zero extension
        y = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y)); // zero extension
        if constexpr (!flags.addz2) {
          return y;
        }
      }
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI2__)
      else if constexpr (flags.compress) {
        y = _mm256_maskz_compress_epi8(__mmask32(compress_mask(indexs)), y); // compress
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (flags.expand) {
        y = _mm256_maskz_expand_epi8(__mmask32(expand_mask(indexs)), y); // expand
        if constexpr (!flags.addz2) {
          return y;
        }
      }
#endif // AVX512VBMI2
      else if constexpr (!flags.cross_lane) { // no lane crossing. Use pshufb
        constexpr std::array<stado::i8, 32> bm = pshufb_mask<i8x32>(indexs);
        return _mm256_shuffle_epi8(a, i8x32().load(bm.data()));
      } else if constexpr (flags.rotate_big) { // fits full rotate
        constexpr u8 rot = flags.rot_count; // rotate count
        __m256i swap = _mm256_permute4x64_epi64(a, 0x4E); // swap 128-bit halves
        if (rot <= 16) {
          y = _mm256_alignr_epi8(swap, y, rot);
        } else {
          y = _mm256_alignr_epi8(y, swap, rot & 15);
        }
      } else if constexpr (flags.broadcast && flags.rot_count == 0) {
        y = _mm256_broadcastb_epi8(_mm256_castsi256_si128(y)); // broadcast first element
      } else { // full permute needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI__) // AVX512VBMI
        constexpr std::array<stado::i8, 32> bm = perm_mask_broad<i8, 32>(indexs);
        y = _mm256_permutexvar_epi8(i8x32().load(bm.data()), y);
#else
        // no full permute instruction available
        __m256i swap = _mm256_permute4x64_epi64(y, 0x4E); // swap high and low 128-bit lane
        constexpr std::array<stado::i8, 32> bm1 = pshufb_mask<i8x32, 1>(indexs);
        constexpr std::array<stado::i8, 32> bm2 = pshufb_mask<i8x32, 0>(indexs);
        __m256i r1 = _mm256_shuffle_epi8(swap, i8x32().load(bm1.data()));
        __m256i r2 = _mm256_shuffle_epi8(y, i8x32().load(bm2.data()));
        return _mm256_or_si256(r1, r2);
#endif
      }
    }
  }
  if constexpr (flags.zeroing) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_epi8(zero_mask<32>(indexs), y);
#else // use broad mask
    constexpr std::array<stado::i8, 32> bm = zero_mask_broad<stado::i8, 32>(indexs);
    y = _mm256_and_si256(i8x32().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15, int i16, int i17, int i18, int i19, int i20,
         int i21, int i22, int i23, int i24, int i25, int i26, int i27, int i28, int i29, int i30,
         int i31>
inline u8x32 permute32(const u8x32 a) {
  return u8x32(
    permute32<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18,
              i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31>(i8x32(a)));
}

/*****************************************************************************
 *
 *          NativeVector blend functions
 *
 *****************************************************************************/

// permute and blend i64x4
template<int i0, int i1, int i2, int i3>
inline i64x4 blend4(const i64x4 a, const i64x4 b) {
  constexpr std::array<int, 4> indexs{i0, i1, i2, i3}; // indexes as array
  __m256i y = a; // result
  constexpr u64 flags =
    blend_flags<i64x4>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm256_setzero_si256(); // just return zero
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute4<i0, i1, i2, i3>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    return permute4<(i0 < 0 ? i0 : i0 & 3), (i1 < 0 ? i1 : i1 & 3), (i2 < 0 ? i2 : i2 & 3),
                    (i3 < 0 ? i3 : i3 & 3)>(b);
  }
  if constexpr ((flags & (blend_perma | blend_permb)) == 0) { // no permutation, only blending
    constexpr u8 mb = (u8)make_bit_mask<4, 0x302>(indexs); // blend mask
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm256_mask_mov_epi64(a, mb, b);
#else // AVX2
    y = _mm256_blend_epi32(
      a, b, ((mb & 1) | (mb & 2) << 1 | (mb & 4) << 2 | (mb & 8) << 3) * 3); // duplicate each bit
#endif
  } else if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 128-bit blocks
    constexpr std::array<int, 2> arr = largeblock_perm<4>(indexs); // get 128-bit blend pattern
    constexpr u8 pp = (arr[0] & 0xF) | u8(arr[1] & 0xF) << 4;
    y = _mm256_permute2x128_si256(a, b, pp);
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm256_unpacklo_epi64(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm256_unpacklo_epi64(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm256_unpackhi_epi64(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm256_unpackhi_epi64(b, a);
  } else if constexpr ((flags & blend_rotateab) != 0) {
    y = _mm256_alignr_epi8(a, b, flags >> blend_rotpattern);
  } else if constexpr ((flags & blend_rotateba) != 0) {
    y = _mm256_alignr_epi8(b, a, flags >> blend_rotpattern);
  }
#if ALLOW_FP_PERMUTE // allow floating point permute instructions on integer vectors
  else if constexpr ((flags & blend_shufab) != 0) { // use floating point instruction shufpd
    y = _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b),
                                              (flags >> blend_shufpattern) & 0xF));
  } else if constexpr ((flags & blend_shufba) != 0) { // use floating point instruction shufpd
    y = _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(b), _mm256_castsi256_pd(a),
                                              (flags >> blend_shufpattern) & 0xF));
  }
#endif
  else {
    // No special cases
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL. use vpermi2q
    const __m256i maskp = _mm256_setr_epi32(i0 & 15, 0, i1 & 15, 0, i2 & 15, 0, i3 & 15, 0);
    return _mm256_maskz_permutex2var_epi64(zero_mask<4>(indexs), a, maskp, b);
#else // permute a and b separately, then blend.
    constexpr std::array<int, 8> arr = blend_perm_indexes<4, 0>(indexs); // get permutation indexes
    __m256i ya = permute4<arr[0], arr[1], arr[2], arr[3]>(a);
    __m256i yb = permute4<arr[4], arr[5], arr[6], arr[7]>(b);
    constexpr u8 mb = (u8)make_bit_mask<4, 0x302>(indexs); // blend mask
    y = _mm256_blend_epi32(ya, yb,
                           ((mb & 1) | (mb & 2) << 1 | (mb & 4) << 2 | (mb & 8) << 3) *
                             3); // duplicate each bit
#endif
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_epi64(zero_mask<4>(indexs), y);
#else // use broad mask
    constexpr std::array<i64, 4> bm = zero_mask_broad<i64, 4>(indexs);
    y = _mm256_and_si256(i64x4().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3>
inline u64x4 blend4(const u64x4 a, const u64x4 b) {
  return u64x4(blend4<i0, i1, i2, i3>(i64x4(a), i64x4(b)));
}

// permute and blend i32x8
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline i32x8 blend8(const i32x8 a, const i32x8 b) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  __m256i y = a; // result
  constexpr u64 flags =
    blend_flags<i32x8>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm256_setzero_si256(); // just return zero
  }

  if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 32-bit blocks
    constexpr std::array<int, 4> arr = largeblock_perm<8>(indexs); // get 32-bit blend pattern
    y = blend4<arr[0], arr[1], arr[2], arr[3]>(i64x4(a), i64x4(b));
    if (!(flags & blend_addz)) {
      return y; // no remaining zeroing
    }
  } else if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute8<i0, i1, i2, i3, i4, i5, i6, i7>(a);
  } else if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    constexpr std::array<int, 16> arr = blend_perm_indexes<8, 2>(indexs); // get permutation indexes
    return permute8<arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(b);
  } else if constexpr ((flags & (blend_perma | blend_permb)) ==
                       0) { // no permutation, only blending
    constexpr u8 mb = (u8)make_bit_mask<8, 0x303>(indexs); // blend mask
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm256_mask_mov_epi32(a, mb, b);
#else // AVX2
    y = _mm256_blend_epi32(a, b, mb);
#endif
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm256_unpacklo_epi32(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm256_unpacklo_epi32(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm256_unpackhi_epi32(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm256_unpackhi_epi32(b, a);
  } else if constexpr ((flags & blend_rotateab) != 0) {
    y = _mm256_alignr_epi8(a, b, flags >> blend_rotpattern);
  } else if constexpr ((flags & blend_rotateba) != 0) {
    y = _mm256_alignr_epi8(b, a, flags >> blend_rotpattern);
  }
#if ALLOW_FP_PERMUTE // allow floating point permute instructions on integer vectors
  else if constexpr ((flags & blend_shufab) != 0) { // use floating point instruction shufpd
    y = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b),
                                              u8(flags >> blend_shufpattern)));
  } else if constexpr ((flags & blend_shufba) != 0) { // use floating point instruction shufpd
    y = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(b), _mm256_castsi256_ps(a),
                                              u8(flags >> blend_shufpattern)));
  }
#endif
  else {
    // No special cases
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL. use vpermi2d
    const __m256i maskp =
      _mm256_setr_epi32(i0 & 15, i1 & 15, i2 & 15, i3 & 15, i4 & 15, i5 & 15, i6 & 15, i7 & 15);
    return _mm256_maskz_permutex2var_epi32(zero_mask<8>(indexs), a, maskp, b);
#else // permute a and b separately, then blend.
    constexpr std::array<int, 16> arr = blend_perm_indexes<8, 0>(indexs); // get permutation indexes
    __m256i ya = permute8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(a);
    __m256i yb = permute8<arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(b);
    constexpr u8 mb = (u8)make_bit_mask<8, 0x303>(indexs); // blend mask
    y = _mm256_blend_epi32(ya, yb, mb);
#endif
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_epi32(zero_mask<8>(indexs), y);
#else // use broad mask
    constexpr std::array<i32, 8> bm = zero_mask_broad<i32, 8>(indexs);
    y = _mm256_and_si256(i32x8().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline u32x8 blend8(const u32x8 a, const u32x8 b) {
  return u32x8(blend8<i0, i1, i2, i3, i4, i5, i6, i7>(i32x8(a), i32x8(b)));
}

// permute and blend i16x16
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline i16x16 blend16(const i16x16 a, const i16x16 b) {
  constexpr std::array<int, 16> indexs{i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                       i8, i9, i10, i11, i12, i13, i14, i15}; // indexes as array
  __m256i y = a; // result
  constexpr u64 flags =
    blend_flags<i16x16>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm256_setzero_si256(); // just return zero
  }

  if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 32-bit blocks
    constexpr std::array<int, 8> arr = largeblock_perm<16>(indexs); // get 32-bit blend pattern
    y = blend8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(i32x8(a), i32x8(b));
    if (!(flags & blend_addz)) {
      return y; // no remaining zeroing
    }
  } else if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15>(a);
  } else if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    // get permutation indexes
    constexpr std::array<int, 32> arr = blend_perm_indexes<16, 2>(indexs);
    return permute16<arr[16], arr[17], arr[18], arr[19], arr[20], arr[21], arr[22], arr[23],
                     arr[24], arr[25], arr[26], arr[27], arr[28], arr[29], arr[30], arr[31]>(b);
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm256_unpacklo_epi16(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm256_unpacklo_epi16(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm256_unpackhi_epi16(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm256_unpackhi_epi16(b, a);
  } else if constexpr ((flags & blend_rotateab) != 0) {
    y = _mm256_alignr_epi8(a, b, flags >> blend_rotpattern);
  } else if constexpr ((flags & blend_rotateba) != 0) {
    y = _mm256_alignr_epi8(b, a, flags >> blend_rotpattern);
  } else {
    // No special cases
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL. use vpermi2w
    if constexpr ((flags & (blend_perma | blend_permb)) != 0) {
      constexpr std::array<i16, 16> bm = perm_mask_broad<i16, 16>(indexs);
      return _mm256_maskz_permutex2var_epi16(zero_mask<16>(indexs), a, i16x16().load(bm.data()), b);
    }
#endif
    // permute a and b separately, then blend.
    // a and b permuted
    i16x16 ya = a;
    i16x16 yb = b;
    // get permutation indexes
    constexpr std::array<int, 32> arr = blend_perm_indexes<16, 0>(indexs);
    if constexpr ((flags & blend_perma) != 0) {
      ya = permute16<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
                     arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(ya);
    }
    if constexpr ((flags & blend_permb) != 0) {
      yb = permute16<arr[16], arr[17], arr[18], arr[19], arr[20], arr[21], arr[22], arr[23],
                     arr[24], arr[25], arr[26], arr[27], arr[28], arr[29], arr[30], arr[31]>(yb);
    }
    constexpr u16 mb = (u16)make_bit_mask<16, 0x304>(indexs); // blend mask
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm256_mask_mov_epi16(ya, mb, yb);
#else // AVX2
    if ((flags & blend_same_pattern) != 0) { // same blend pattern in both 128-bit lanes
      y = _mm256_blend_epi16(ya, yb, (u8)mb);
    } else {
      constexpr std::array<i16, 16> bm = make_broad_mask<i16, 16>(mb);
      y = _mm256_blendv_epi8(ya, yb, i16x16().load(bm.data()));
    }
#endif
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_epi16(zero_mask<16>(indexs), y);
#else // use broad mask
    constexpr std::array<i16, 16> bm = zero_mask_broad<i16, 16>(indexs);
    y = _mm256_and_si256(i16x16().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline u16x16 blend16(const u16x16 a, const u16x16 b) {
  return u16x16(blend16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15>(
    i16x16(a), i16x16(b)));
}

// permute and blend i8x32
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15, int i16, int i17, int i18, int i19, int i20,
         int i21, int i22, int i23, int i24, int i25, int i26, int i27, int i28, int i29, int i30,
         int i31>
inline i8x32 blend32(const i8x32 a, const i8x32 b) {
  constexpr std::array<int, 32> indexs{
    i0,  i1,  i2,  i3,  i4,  i5,  i6,  i7,  i8,  i9,  i10, i11, i12, i13, i14, i15,
    i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31}; // indexes as
                                                                                     // array
  __m256i y = a; // result
  constexpr u64 flags =
    blend_flags<i8x32>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm256_setzero_si256(); // just return zero
  }

  if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 16-bit blocks
    constexpr std::array<int, 16> arr = largeblock_perm<32>(indexs); // get 16-bit blend pattern
    y = blend16<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
                arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(i16x16(a), i16x16(b));
    if (!(flags & blend_addz)) {
      return y; // no remaining zeroing
    }
  } else if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute32<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17,
                     i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31>(a);
  } else if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    // get permutation indexes
    constexpr std::array<int, 64> arr = blend_perm_indexes<32, 2>(indexs);
    return permute32<arr[32], arr[33], arr[34], arr[35], arr[36], arr[37], arr[38], arr[39],
                     arr[40], arr[41], arr[42], arr[43], arr[44], arr[45], arr[46], arr[47],
                     arr[48], arr[49], arr[50], arr[51], arr[52], arr[53], arr[54], arr[55],
                     arr[56], arr[57], arr[58], arr[59], arr[60], arr[61], arr[62], arr[63]>(b);
  } else {
    // No special cases
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI__)
    // AVX512VL + AVX512VBMI. use vpermi2b
    if constexpr ((flags & (blend_perma | blend_permb)) != 0) {
      constexpr std::array<stado::i8, 32> bm = perm_mask_broad<stado::i8, 32>(indexs);
      return _mm256_maskz_permutex2var_epi8(zero_mask<32>(indexs), a, i8x32().load(bm.data()), b);
    }
#endif
    // permute a and b separately, then blend.
    // a and b permuted
    i8x32 ya = a;
    i8x32 yb = b;
    // get permutation indexes
    constexpr std::array<int, 64> arr = blend_perm_indexes<32, 0>(indexs);
    if constexpr ((flags & blend_perma) != 0) {
      ya = permute32<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
                     arr[10], arr[11], arr[12], arr[13], arr[14], arr[15], arr[16], arr[17],
                     arr[18], arr[19], arr[20], arr[21], arr[22], arr[23], arr[24], arr[25],
                     arr[26], arr[27], arr[28], arr[29], arr[30], arr[31]>(ya);
    }
    if constexpr ((flags & blend_permb) != 0) {
      yb = permute32<arr[32], arr[33], arr[34], arr[35], arr[36], arr[37], arr[38], arr[39],
                     arr[40], arr[41], arr[42], arr[43], arr[44], arr[45], arr[46], arr[47],
                     arr[48], arr[49], arr[50], arr[51], arr[52], arr[53], arr[54], arr[55],
                     arr[56], arr[57], arr[58], arr[59], arr[60], arr[61], arr[62], arr[63]>(yb);
    }
    constexpr u32 mb = (u32)make_bit_mask<32, 0x305>(indexs); // blend mask
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm256_mask_mov_epi8(ya, mb, yb);
#else // AVX2
    constexpr std::array<stado::i8, 32> bm = make_broad_mask<stado::i8, 32>(mb);
    y = _mm256_blendv_epi8(ya, yb, i8x32().load(bm.data()));
#endif
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm256_maskz_mov_epi8(zero_mask<32>(indexs), y);
#else // use broad mask
    constexpr std::array<stado::i8, 32> bm = zero_mask_broad<stado::i8, 32>(indexs);
    y = _mm256_and_si256(i8x32().load(bm.data()), y);
#endif
  }
  return y;
}

template<int... i0>
inline u8x32 blend32(const u8x32 a, const u8x32 b) {
  return u8x32(blend32<i0...>(i8x32(a), i8x32(b)));
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

template<AnyInt8x32 TVec>
inline TVec lookup32(const TVec index, const TVec table) {
#ifdef __XOP__ // AMD XOP instruction set. Use VPPERM
  using Half = TVec;
  typename TVec::Half t0 = _mm_perm_epi8(table.get_low(), table.get_high(), index.get_low());
  typename TVec::Half t1 = _mm_perm_epi8(table.get_low(), table.get_high(), index.get_high());
  return {t0, t1};
#else
  TVec f0 = _mm256_setr_epi32(0, 0, 0, 0, 0x10101010, 0x10101010, 0x10101010, 0x10101010);
  TVec f1 = _mm256_setr_epi32(0x10101010, 0x10101010, 0x10101010, 0x10101010, 0, 0, 0, 0);
  TVec tablef = _mm256_permute4x64_epi64(table, 0x4E); // low and high parts swapped
  TVec r0 = _mm256_shuffle_epi8(table, (index ^ f0) + TVec(0x70));
  TVec r1 = _mm256_shuffle_epi8(tablef, (index ^ f1) + TVec(0x70));
  return r0 | r1;
#endif
}

inline u8x32 lookup(const u8x32 index, const u8* table) {
  u32x8 mask0(0x000000FF); // mask 8 bits
  u8x32 t0 =
    _mm256_i32gather_epi32(reinterpret_cast<const int*>(table), __m256i(mask0 & u32x8(index)),
                           1); // positions 0, 4, 8,  ...
  u8x32 t1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(table),
                                    __m256i(mask0 & _mm256_srli_epi32(index, 8)),
                                    1); // positions 1, 5, 9,  ...
  u8x32 t2 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(table),
                                    __m256i(mask0 & _mm256_srli_epi32(index, 16)),
                                    1); // positions 2, 6, 10, ...
  u8x32 t3 =
    _mm256_i32gather_epi32(reinterpret_cast<const int*>(table), _mm256_srli_epi32(index, 24),
                           1); // positions 3, 7, 11, ...
  t0 = u8x32(t0 & u8x32(mask0));
  t1 = _mm256_slli_epi32(t1 & u8x32(mask0), 8);
  t2 = _mm256_slli_epi32(t2 & u8x32(mask0), 16);
  t3 = _mm256_slli_epi32(t3, 24);
  return (t0 | t3) | (t1 | t2);
}

template<std::size_t n>
inline u8x32 lookup_bounded(const u8x32 index, const u8* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 16) {
    u8x16 tt = u8x16().load(table);
    u8x16 r0 = lookup16(index.get_low(), tt);
    u8x16 r1 = lookup16(index.get_high(), tt);
    return {r0, r1};
  }
  if constexpr (n <= 32) {
    return lookup32(index, u8x32().load(table));
  }
  // n > 32
  return lookup(index, table);
}

inline i8x32 lookup(const i8x32 index, const i8* table) {
  return lookup(u8x32(index), reinterpret_cast<const u8*>(table));
}
template<std::size_t n>
inline i8x32 lookup_bounded(const i8x32 index, const i8* table) {
  return lookup_bounded<n>(u8x32(index), reinterpret_cast<const u8*>(table));
}

inline i16x16 lookup16(const i16x16 index, const i16x16 table) {
  return i16x16(lookup32(i8x32(index * i16x16(0x202) + 0x100), i8x32(table)));
}

inline i16x16 lookup(const i16x16 index, const i16* table) {
  // even positions
  i16x16 t1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(table),
                                     __m256i(u32x8(index) & u32x8(0x0000FFFF)), 2);
  // odd positions
  i16x16 t2 =
    _mm256_i32gather_epi32(reinterpret_cast<const int*>(table), _mm256_srli_epi32(index, 16), 2);
  return blend16<0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30>(t1, t2);
}

template<std::size_t n>
inline i16x16 lookup_bounded(const i16x16 index, const i16* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 8) {
    i16x8 table1 = i16x8().load(table);
    return i16x16(lookup8(index.get_low(), table1), lookup8(index.get_high(), table1));
  }
  if constexpr (n <= 16) {
    return lookup16(index, i16x16().load(table));
  }
  // n > 16
  return lookup(index, table);
}

inline i32x8 lookup8(const i32x8 index, const i32x8 table) {
  return _mm256_permutevar8x32_epi32(table, index);
}

template<AnyInt32x8 TIdx, AnyInt32 TVal>
inline NativeVector<TVal, 8> lookup(const TIdx index, const TVal* table) {
  return _mm256_i32gather_epi32(reinterpret_cast<const i32*>(table), index, 4);
}

template<std::size_t n, AnyInt32x8 TIdx, AnyInt32 TVal>
inline NativeVector<TVal, 8> lookup_bounded(const TIdx index, const TVal* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 8) {
    i32x8 table1 = i32x8().load(table);
    return lookup8(index, table1);
  }
  if constexpr (n <= 16) {
    i32x8 table1 = i32x8().load(table);
    i32x8 table2 = i32x8().load((const i32*)table + 8);
    i32x8 y1 = lookup8(index, table1);
    i32x8 y2 = lookup8(index, table2);
    Mask<32, 8> s = index > 7;
    return select(s, y2, y1);
  }
  // n > 16
  return lookup(index, table);
}

inline i64x4 lookup4(const i64x4 index, const i64x4 table) {
  return i64x4(lookup8(i32x8(index * 0x200000002LL + 0x100000000LL), i32x8(table)));
}

template<AnyInt64x4 TIdx, AnyInt64 TVal>
inline NativeVector<TVal, 4> lookup(const TIdx index, const TVal* table) {
  return _mm256_i64gather_epi64(reinterpret_cast<const i64*>(table), index, 8);
}

template<std::size_t n>
inline i64x4 lookup_bounded(const i64x4 index, const i64* table) {
  if constexpr (n == 0) {
    return 0;
  }
  // n > 0
  return lookup(index, table);
}

/*****************************************************************************
 *
 *          Byte shifts
 *
 *****************************************************************************/

// Function shift_bytes_up: shift whole vector left by b bytes.
template<unsigned int b>
inline i8x32 shift_bytes_up(const i8x32 a) {
  __m256i ahi;
  __m256i alo;
  if constexpr (b == 0) {
    return a;
  }
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  else if constexpr ((b & 3) == 0) { // b is divisible by 4
    return _mm256_alignr_epi32(a, _mm256_setzero_si256(), (8 - (b >> 2)) & 7);
  }
#endif
  else if constexpr (b < 16) {
    alo = a;
    ahi = _mm256_inserti128_si256(_mm256_setzero_si256(), _mm256_castsi256_si128(a),
                                  1); // shift a 16 bytes up, zero lower part
  } else if constexpr (b < 32) {
    alo = _mm256_inserti128_si256(_mm256_setzero_si256(), _mm256_castsi256_si128(a),
                                  1); // shift a 16 bytes up, zero lower part
    ahi = _mm256_setzero_si256();
  } else {
    return _mm256_setzero_si256(); // zero
  }
  if constexpr ((b & 0xF) == 0) {
    return alo; // modulo 16. no more shift needeed
  }
  return _mm256_alignr_epi8(alo, ahi, 16 - (b & 0xF)); // shift within 16-bytes lane
}

// Function shift_bytes_down: shift whole vector right by b bytes
template<unsigned int b>
inline i8x32 shift_bytes_down(const i8x32 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  if constexpr ((b & 3) == 0) { // b is divisible by 4
    return _mm256_alignr_epi32(_mm256_setzero_si256(), a, (b >> 2) & 7);
  }
#endif
  __m256i ahi;
  __m256i alo;
  if constexpr (b < 16) {
    // shift a 16 bytes down, zero upper part
    alo = _mm256_inserti128_si256(
      _mm256_setzero_si256(), _mm256_extracti128_si256(a, 1),
      0); // make sure the upper part is zero (otherwise, an optimizing compiler can mess it up)
    ahi = a;
  } else if constexpr (b < 32) {
    alo = _mm256_setzero_si256(); // zero
    ahi = _mm256_inserti128_si256(_mm256_setzero_si256(), _mm256_extracti128_si256(a, 1),
                                  0); // shift a 16 bytes down, zero upper part
  } else {
    return _mm256_setzero_si256(); // zero
  }
  if constexpr ((b & 0xF) == 0) {
    return ahi; // modulo 16. no more shift needeed
  }
  return _mm256_alignr_epi8(alo, ahi, b & 0xF); // shift within 16-bytes lane
}

/*****************************************************************************
 *
 *          Gather functions with fixed indexes
 *
 *****************************************************************************/
// Load elements from array a with indices i0, i1, i2, i3, i4, i5, i6, i7
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline i32x8 gather8i(const void* a) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  constexpr int imin = min_index(indexs);
  constexpr int imax = max_index(indexs);
  static_assert(imin >= 0, "Negative index in gather function");

  if constexpr (imax - imin <= 7) {
    // load one contiguous block and permute
    if constexpr (imax > 7) {
      // make sure we don't read past the end of the array
      i32x8 b = i32x8().load(reinterpret_cast<const i32*>(a) + imax - 7);
      return permute8<i0 - imax + 7, i1 - imax + 7, i2 - imax + 7, i3 - imax + 7, i4 - imax + 7,
                      i5 - imax + 7, i6 - imax + 7, i7 - imax + 7>(b);
    } else {
      i32x8 b = i32x8().load(reinterpret_cast<const i32*>(a) + imin);
      return permute8<i0 - imin, i1 - imin, i2 - imin, i3 - imin, i4 - imin, i5 - imin, i6 - imin,
                      i7 - imin>(b);
    }
  }
  if constexpr ((i0 < imin + 8 || i0 > imax - 8) && (i1 < imin + 8 || i1 > imax - 8) &&
                (i2 < imin + 8 || i2 > imax - 8) && (i3 < imin + 8 || i3 > imax - 8) &&
                (i4 < imin + 8 || i4 > imax - 8) && (i5 < imin + 8 || i5 > imax - 8) &&
                (i6 < imin + 8 || i6 > imax - 8) && (i7 < imin + 8 || i7 > imax - 8)) {
    // load two contiguous blocks and blend
    i32x8 b = i32x8().load(reinterpret_cast<const i32*>(a) + imin);
    i32x8 c = i32x8().load(reinterpret_cast<const i32*>(a) + imax - 7);
    constexpr int j0 = i0 < imin + 8 ? i0 - imin : 15 - imax + i0;
    constexpr int j1 = i1 < imin + 8 ? i1 - imin : 15 - imax + i1;
    constexpr int j2 = i2 < imin + 8 ? i2 - imin : 15 - imax + i2;
    constexpr int j3 = i3 < imin + 8 ? i3 - imin : 15 - imax + i3;
    constexpr int j4 = i4 < imin + 8 ? i4 - imin : 15 - imax + i4;
    constexpr int j5 = i5 < imin + 8 ? i5 - imin : 15 - imax + i5;
    constexpr int j6 = i6 < imin + 8 ? i6 - imin : 15 - imax + i6;
    constexpr int j7 = i7 < imin + 8 ? i7 - imin : 15 - imax + i7;
    return blend8<j0, j1, j2, j3, j4, j5, j6, j7>(b, c);
  }
  // use AVX2 gather
  return _mm256_i32gather_epi32(reinterpret_cast<const i32*>(a),
                                (i32x8(i0, i1, i2, i3, i4, i5, i6, i7)), 4);
}

template<int i0, int i1, int i2, int i3>
inline i64x4 gather4q(const void* a) {
  constexpr std::array<int, 4> indexs{i0, i1, i2, i3}; // indexes as array
  constexpr int imin = min_index(indexs);
  constexpr int imax = max_index(indexs);
  static_assert(imin >= 0, "Negative index in gather function");

  if constexpr (imax - imin <= 3) {
    // load one contiguous block and permute
    if constexpr (imax > 3) {
      // make sure we don't read past the end of the array
      i64x4 b = i64x4().load(reinterpret_cast<const i64*>(a) + imax - 3);
      return permute4<i0 - imax + 3, i1 - imax + 3, i2 - imax + 3, i3 - imax + 3>(b);
    } else {
      i64x4 b = i64x4().load(reinterpret_cast<const i64*>(a) + imin);
      return permute4<i0 - imin, i1 - imin, i2 - imin, i3 - imin>(b);
    }
  }
  if constexpr ((i0 < imin + 4 || i0 > imax - 4) && (i1 < imin + 4 || i1 > imax - 4) &&
                (i2 < imin + 4 || i2 > imax - 4) && (i3 < imin + 4 || i3 > imax - 4)) {
    // load two contiguous blocks and blend
    i64x4 b = i64x4().load(reinterpret_cast<const i64*>(a) + imin);
    i64x4 c = i64x4().load(reinterpret_cast<const i64*>(a) + imax - 3);
    const int j0 = i0 < imin + 4 ? i0 - imin : 7 - imax + i0;
    const int j1 = i1 < imin + 4 ? i1 - imin : 7 - imax + i1;
    const int j2 = i2 < imin + 4 ? i2 - imin : 7 - imax + i2;
    const int j3 = i3 < imin + 4 ? i3 - imin : 7 - imax + i3;
    return blend4<j0, j1, j2, j3>(b, c);
  }
  // use AVX2 gather
  return _mm256_i32gather_epi64(reinterpret_cast<const i64*>(a), (i32x4(i0, i1, i2, i3)), 8);
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

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline void scatter(const i32x8 data, void* array) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  __m256i indx = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
  __mmask8 mask = (i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3) |
                  ((i4 >= 0) << 4) | ((i5 >= 0) << 5) | ((i6 >= 0) << 6) | ((i7 >= 0) << 7);
  _mm256_mask_i32scatter_epi32(reinterpret_cast<i32*>(array), mask, indx, data, 4);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  __m512i indx = _mm512_castsi256_si512(_mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7));
  __mmask16 mask = (i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3) |
                   ((i4 >= 0) << 4) | ((i5 >= 0) << 5) | ((i6 >= 0) << 6) | ((i7 >= 0) << 7);
  _mm512_mask_i32scatter_epi32(reinterpret_cast<i32*>(array), mask, indx,
                               _mm512_castsi256_si512(data), 4);
#else
  i32* arr = reinterpret_cast<i32*>(array);
  const std::array<int, 8> index{i0, i1, i2, i3, i4, i5, i6, i7};
  for (std::size_t i = 0; i < 8; ++i) {
    if (index[i] >= 0) {
      arr[index[i]] = data[i];
    }
  }
#endif
}

template<int i0, int i1, int i2, int i3>
inline void scatter(const i64x4 data, void* array) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  __m128i indx = _mm_setr_epi32(i0, i1, i2, i3);
  __mmask8 mask = (i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3);
  _mm256_mask_i32scatter_epi64(reinterpret_cast<i64*>(array), mask, indx, data, 8);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  __m256i indx = _mm256_castsi128_si256(_mm_setr_epi32(i0, i1, i2, i3));
  __mmask16 mask = (i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3);
  _mm512_mask_i32scatter_epi64(reinterpret_cast<i64*>(array), (__mmask8)mask, indx,
                               _mm512_castsi256_si512(data), 8);
#else
  i64* arr = reinterpret_cast<i64*>(array);
  const std::array<int, 4> index{i0, i1, i2, i3};
  for (std::size_t i = 0; i < 4; ++i) {
    if (index[i] >= 0) {
      arr[index[i]] = data[i];
    }
  }
#endif
}

/*****************************************************************************
 *
 *          Scatter functions with variable indexes
 *
 *****************************************************************************/

inline void scatter(const i32x8 index, u32 limit, const i32x8 data, void* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  __mmask8 mask = _mm256_cmplt_epu32_mask(index, (u32x8(limit)));
  _mm256_mask_i32scatter_epi32(reinterpret_cast<i32*>(destination), mask, index, data, 4);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  // 16 bit mask, upper 8 bits are 0. Usually, we can rely on the upper bit of an extended vector to
  // be zero, but we will mask then off the be sure
  //__mmask16 mask = _mm512_cmplt_epu32_mask(_mm512_castsi256_si512(index),
  //_mm512_castsi256_si512(u32x8(limit)));
  __mmask16 mask = _mm512_mask_cmplt_epu32_mask(0xFF, _mm512_castsi256_si512(index),
                                                _mm512_castsi256_si512(u32x8(limit)));
  _mm512_mask_i32scatter_epi32(reinterpret_cast<i32*>(destination), mask,
                               _mm512_castsi256_si512(index), _mm512_castsi256_si512(data), 4);
#else
  i32* arr = reinterpret_cast<i32*>(destination);
  for (std::size_t i = 0; i < 8; ++i) {
    if (u32(index[i]) < limit) {
      arr[index[i]] = data[i];
    }
  }
#endif
}

inline void scatter(const i64x4 index, u32 limit, const i64x4 data, void* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  __mmask8 mask = _mm256_cmplt_epu64_mask(index, (u64x4(u64(limit))));
  _mm256_mask_i64scatter_epi64(reinterpret_cast<i64*>(destination), mask, index, data, 8);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  // 16 bit mask. upper 12 bits are 0
  __mmask16 mask = _mm512_mask_cmplt_epu64_mask(0xF, _mm512_castsi256_si512(index),
                                                _mm512_castsi256_si512(u64x4(u64(limit))));
  _mm512_mask_i64scatter_epi64(reinterpret_cast<i64*>(destination), (__mmask8)mask,
                               _mm512_castsi256_si512(index), _mm512_castsi256_si512(data), 8);
#else
  i64* arr = reinterpret_cast<i64*>(destination);
  for (std::size_t i = 0; i < 4; ++i) {
    if (u64(index[i]) < u64(limit)) {
      arr[index[i]] = data[i];
    }
  }
#endif
}

inline void scatter(const i32x4 index, u32 limit, const i64x4 data, void* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  __mmask8 mask = _mm_cmplt_epu32_mask(index, (u32x4(limit)));
  _mm256_mask_i32scatter_epi64(reinterpret_cast<i64*>(destination), mask, index, data, 8);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  // 16 bit mask. upper 12 bits are 0
  __mmask16 mask = _mm512_mask_cmplt_epu32_mask(0xF, _mm512_castsi128_si512(index),
                                                _mm512_castsi128_si512(u32x4(limit)));
  _mm512_mask_i32scatter_epi64(reinterpret_cast<i64*>(destination), (__mmask8)mask,
                               _mm256_castsi128_si256(index), _mm512_castsi256_si512(data), 8);
#else
  i64* arr = reinterpret_cast<i64*>(destination);
  for (std::size_t i = 0; i < 4; ++i) {
    if (u32(index[i]) < limit) {
      arr[index[i]] = data[i];
    }
  }
#endif
}

/*****************************************************************************
 *
 *          Functions for conversion between integer sizes
 *
 *****************************************************************************/

// Extend 8-bit integers to 16-bit integers, signed and unsigned

// Function extend_low : extends the low 16 elements to 16 bits with sign extension
inline i16x16 extend_low(const i8x32 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0x10); // get bits 64-127 to position 128-191
  __m256i sign = _mm256_cmpgt_epi8(_mm256_setzero_si256(), a2); // 0 > a2
  return _mm256_unpacklo_epi8(a2, sign); // interleave with sign extensions
}

// Function extend_high : extends the high 16 elements to 16 bits with sign extension
inline i16x16 extend_high(const i8x32 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0xC8); // get bits 128-191 to position 64-127
  __m256i sign = _mm256_cmpgt_epi8(_mm256_setzero_si256(), a2); // 0 > a2
  return _mm256_unpackhi_epi8(a2, sign); // interleave with sign extensions
}

// Function extend_low : extends the low 16 elements to 16 bits with zero extension
inline u16x16 extend_low(const u8x32 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0x10); // get bits 64-127 to position 128-191
  return _mm256_unpacklo_epi8(a2, _mm256_setzero_si256()); // interleave with zero extensions
}

// Function extend_high : extends the high 19 elements to 16 bits with zero extension
inline u16x16 extend_high(const u8x32 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0xC8); // get bits 128-191 to position 64-127
  return _mm256_unpackhi_epi8(a2, _mm256_setzero_si256()); // interleave with zero extensions
}

// Extend 16-bit integers to 32-bit integers, signed and unsigned

// Function extend_low : extends the low 8 elements to 32 bits with sign extension
inline i32x8 extend_low(const i16x16 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0x10); // get bits 64-127 to position 128-191
  __m256i sign = _mm256_srai_epi16(a2, 15); // sign bit
  return _mm256_unpacklo_epi16(a2, sign); // interleave with sign extensions
}

// Function extend_high : extends the high 8 elements to 32 bits with sign extension
inline i32x8 extend_high(const i16x16 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0xC8); // get bits 128-191 to position 64-127
  __m256i sign = _mm256_srai_epi16(a2, 15); // sign bit
  return _mm256_unpackhi_epi16(a2, sign); // interleave with sign extensions
}

// Function extend_low : extends the low 8 elements to 32 bits with zero extension
inline u32x8 extend_low(const u16x16 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0x10); // get bits 64-127 to position 128-191
  return _mm256_unpacklo_epi16(a2, _mm256_setzero_si256()); // interleave with zero extensions
}

// Function extend_high : extends the high 8 elements to 32 bits with zero extension
inline u32x8 extend_high(const u16x16 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0xC8); // get bits 128-191 to position 64-127
  return _mm256_unpackhi_epi16(a2, _mm256_setzero_si256()); // interleave with zero extensions
}

// Extend 32-bit integers to 64-bit integers, signed and unsigned

// Function extend_low : extends the low 4 elements to 64 bits with sign extension
inline i64x4 extend_low(const i32x8 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0x10); // get bits 64-127 to position 128-191
  __m256i sign = _mm256_srai_epi32(a2, 31); // sign bit
  return _mm256_unpacklo_epi32(a2, sign); // interleave with sign extensions
}

// Function extend_high : extends the high 4 elements to 64 bits with sign extension
inline i64x4 extend_high(const i32x8 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0xC8); // get bits 128-191 to position 64-127
  __m256i sign = _mm256_srai_epi32(a2, 31); // sign bit
  return _mm256_unpackhi_epi32(a2, sign); // interleave with sign extensions
}

// Function extend_low : extends the low 4 elements to 64 bits with zero extension
inline u64x4 extend_low(const u32x8 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0x10); // get bits 64-127 to position 128-191
  return _mm256_unpacklo_epi32(a2, _mm256_setzero_si256()); // interleave with zero extensions
}

// Function extend_high : extends the high 4 elements to 64 bits with zero extension
inline u64x4 extend_high(const u32x8 a) {
  __m256i a2 = _mm256_permute4x64_epi64(a, 0xC8); // get bits 128-191 to position 64-127
  return _mm256_unpackhi_epi32(a2, _mm256_setzero_si256()); // interleave with zero extensions
}

// Compress 16-bit integers to 8-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Overflow wraps around
inline i8x32 compress(const i16x16 low, const i16x16 high) {
  __m256i mask = _mm256_set1_epi32(0x00FF00FF); // mask for low bytes
  __m256i lowm = _mm256_and_si256(low, mask); // bytes of low
  __m256i highm = _mm256_and_si256(high, mask); // bytes of high
  __m256i pk = _mm256_packus_epi16(lowm, highm); // unsigned pack
  return _mm256_permute4x64_epi64(pk, 0xD8); // put in right place
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Signed, with saturation
inline i8x32 compress_saturated(const i16x16 low, const i16x16 high) {
  __m256i pk = _mm256_packs_epi16(low, high); // packed with signed saturation
  return _mm256_permute4x64_epi64(pk, 0xD8); // put in right place
}

// Function compress : packs two vectors of 16-bit integers to one vector of 8-bit integers
// Unsigned, overflow wraps around
inline u8x32 compress(const u16x16 low, const u16x16 high) {
  return u8x32(compress((i16x16)low, (i16x16)high));
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Unsigned, with saturation
inline u8x32 compress_saturated(const u16x16 low, const u16x16 high) {
  __m256i maxval = _mm256_set1_epi32(0x00FF00FF); // maximum value
  __m256i low1 = _mm256_min_epu16(low, maxval); // upper limit
  __m256i high1 = _mm256_min_epu16(high, maxval); // upper limit
  __m256i pk = _mm256_packus_epi16(
    low1, high1); // this instruction saturates from signed 32 bit to unsigned 16 bit
  return _mm256_permute4x64_epi64(pk, 0xD8); // put in right place
}

// Compress 32-bit integers to 16-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
inline i16x16 compress(const i32x8 low, const i32x8 high) {
  __m256i mask = _mm256_set1_epi32(0x0000FFFF); // mask for low words
  __m256i lowm = _mm256_and_si256(low, mask); // words of low
  __m256i highm = _mm256_and_si256(high, mask); // words of high
  __m256i pk = _mm256_packus_epi32(lowm, highm); // unsigned pack
  return _mm256_permute4x64_epi64(pk, 0xD8); // put in right place
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Signed with saturation
inline i16x16 compress_saturated(const i32x8 low, const i32x8 high) {
  __m256i pk = _mm256_packs_epi32(low, high); // pack with signed saturation
  return _mm256_permute4x64_epi64(pk, 0xD8); // put in right place
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
inline u16x16 compress(const u32x8 low, const u32x8 high) {
  return u16x16(compress((i32x8)low, (i32x8)high));
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Unsigned, with saturation
inline u16x16 compress_saturated(const u32x8 low, const u32x8 high) {
  __m256i maxval = _mm256_set1_epi32(0x0000FFFF); // maximum value
  __m256i low1 = _mm256_min_epu32(low, maxval); // upper limit
  __m256i high1 = _mm256_min_epu32(high, maxval); // upper limit
  __m256i pk = _mm256_packus_epi32(
    low1, high1); // this instruction saturates from signed 32 bit to unsigned 16 bit
  return _mm256_permute4x64_epi64(pk, 0xD8); // put in right place
}

// Compress 64-bit integers to 32-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Overflow wraps around
inline i32x8 compress(const i64x4 low, const i64x4 high) {
  __m256i low2 = _mm256_shuffle_epi32(low, 0xD8); // low dwords of low  to pos. 0 and 32
  __m256i high2 = _mm256_shuffle_epi32(high, 0xD8); // low dwords of high to pos. 0 and 32
  __m256i pk = _mm256_unpacklo_epi64(low2, high2); // interleave
  return _mm256_permute4x64_epi64(pk, 0xD8); // put in right place
}

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Signed, with saturation
inline i32x8 compress_saturated(const i64x4 a, const i64x4 b) {
  i64x4 maxval = _mm256_setr_epi32(0x7FFFFFFF, 0, 0x7FFFFFFF, 0, 0x7FFFFFFF, 0, 0x7FFFFFFF, 0);
  i64x4 minval =
    _mm256_setr_epi32(i32(0x80000000), i32(0xFFFFFFFF), i32(0x80000000), i32(0xFFFFFFFF),
                      i32(0x80000000), i32(0xFFFFFFFF), i32(0x80000000), i32(0xFFFFFFFF));
  i64x4 a1 = min(a, maxval);
  i64x4 b1 = min(b, maxval);
  i64x4 a2 = max(a1, minval);
  i64x4 b2 = max(b1, minval);
  return compress(a2, b2);
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
inline u32x8 compress(const u64x4 low, const u64x4 high) {
  return u32x8(compress((i64x4)low, (i64x4)high));
}

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Unsigned, with saturation
inline u32x8 compress_saturated(const u64x4 low, const u64x4 high) {
  __m256i zero = _mm256_setzero_si256(); // 0
  __m256i lowzero = _mm256_cmpeq_epi32(low, zero); // for each dword is zero
  __m256i highzero = _mm256_cmpeq_epi32(high, zero); // for each dword is zero
  __m256i mone = _mm256_set1_epi32(-1); // FFFFFFFF
  __m256i lownz = _mm256_xor_si256(lowzero, mone); // for each dword is nonzero
  __m256i highnz = _mm256_xor_si256(highzero, mone); // for each dword is nonzero
  __m256i lownz2 = _mm256_srli_epi64(lownz, 32); // shift down to low dword
  __m256i highnz2 = _mm256_srli_epi64(highnz, 32); // shift down to low dword
  __m256i lowsatur = _mm256_or_si256(low, lownz2); // low, saturated
  __m256i hisatur = _mm256_or_si256(high, highnz2); // high, saturated
  return u32x8(compress(i64x4(lowsatur), i64x4(hisatur)));
}

/*****************************************************************************
 *
 *          Integer division operators
 *
 *          Please see the file vectori128.h for explanation.
 *
 *****************************************************************************/

// vector operator / : divide each element by divisor

// vector of 8 32-bit signed integers
inline i32x8 operator/(const i32x8 a, const DivisorI32 d) {
  __m256i m = _mm256_broadcastq_epi64(d.getm()); // broadcast multiplier
  __m256i sgn = _mm256_broadcastq_epi64(d.getsign()); // broadcast sign of d
  __m256i t1 = _mm256_mul_epi32(a, m); // 32x32->64 bit signed multiplication of even elements of a
  __m256i t2 = _mm256_srli_epi64(t1, 32); // high dword of even numbered results
  __m256i t3 = _mm256_srli_epi64(a, 32); // get odd elements of a into position for multiplication
  __m256i t4 = _mm256_mul_epi32(t3, m); // 32x32->64 bit signed multiplication of odd elements
  __m256i t7 = _mm256_blend_epi32(t2, t4, 0xAA);
  __m256i t8 = _mm256_add_epi32(t7, a); // add
  __m256i t9 = _mm256_sra_epi32(t8, d.gets1()); // shift right artihmetic
  __m256i t10 = _mm256_srai_epi32(a, 31); // sign of a
  __m256i t11 = _mm256_sub_epi32(t10, sgn); // sign of a - sign of d
  __m256i t12 = _mm256_sub_epi32(t9, t11); // + 1 if a < 0, -1 if d < 0
  return _mm256_xor_si256(t12, sgn); // change sign if divisor negative
}

// vector of 8 32-bit unsigned integers
inline u32x8 operator/(const u32x8 a, const DivisorU32 d) {
  __m256i m = _mm256_broadcastq_epi64(d.getm()); // broadcast multiplier
  __m256i t1 =
    _mm256_mul_epu32(a, m); // 32x32->64 bit unsigned multiplication of even elements of a
  __m256i t2 = _mm256_srli_epi64(t1, 32); // high dword of even numbered results
  __m256i t3 = _mm256_srli_epi64(a, 32); // get odd elements of a into position for multiplication
  __m256i t4 = _mm256_mul_epu32(t3, m); // 32x32->64 bit unsigned multiplication of odd elements
  __m256i t7 = _mm256_blend_epi32(t2, t4, 0xAA);
  __m256i t8 = _mm256_sub_epi32(a, t7); // subtract
  __m256i t9 = _mm256_srl_epi32(t8, d.gets1()); // shift right logical
  __m256i t10 = _mm256_add_epi32(t7, t9); // add
  return _mm256_srl_epi32(t10, d.gets2()); // shift right logical
}

// vector of 16 16-bit signed integers
inline i16x16 operator/(const i16x16 a, const DivisorI16 d) {
  __m256i m = _mm256_broadcastq_epi64(d.getm()); // broadcast multiplier
  __m256i sgn = _mm256_broadcastq_epi64(d.getsign()); // broadcast sign of d
  __m256i t1 = _mm256_mulhi_epi16(a, m); // multiply high signed words
  __m256i t2 = _mm256_add_epi16(t1, a); // + a
  __m256i t3 = _mm256_sra_epi16(t2, d.gets1()); // shift right artihmetic
  __m256i t4 = _mm256_srai_epi16(a, 15); // sign of a
  __m256i t5 = _mm256_sub_epi16(t4, sgn); // sign of a - sign of d
  __m256i t6 = _mm256_sub_epi16(t3, t5); // + 1 if a < 0, -1 if d < 0
  return _mm256_xor_si256(t6, sgn); // change sign if divisor negative
}

// vector of 16 16-bit unsigned integers
inline u16x16 operator/(const u16x16 a, const DivisorU16 d) {
  __m256i m = _mm256_broadcastq_epi64(d.getm()); // broadcast multiplier
  __m256i t1 = _mm256_mulhi_epu16(a, m); // multiply high signed words
  __m256i t2 = _mm256_sub_epi16(a, t1); // subtract
  __m256i t3 = _mm256_srl_epi16(t2, d.gets1()); // shift right logical
  __m256i t4 = _mm256_add_epi16(t1, t3); // add
  return _mm256_srl_epi16(t4, d.gets2()); // shift right logical
}

// vector of 32 8-bit signed integers
inline i8x32 operator/(const i8x32 a, const DivisorI16 d) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  // sign-extend even-numbered and odd-numbered elements to 16 bits
  i16x16 even = _mm256_srai_epi16(_mm256_slli_epi16(a, 8), 8);
  i16x16 odd = _mm256_srai_epi16(a, 8);
  i16x16 evend = even / d; // divide even-numbered elements
  i16x16 oddd = odd / d; // divide odd-numbered  elements
  oddd = _mm256_slli_epi16(oddd, 8); // shift left to put back in place
  __m256i res = _mm256_mask_mov_epi8(evend, 0xAAAAAAAA, oddd); // interleave even and odd
  return res;
#else
  // expand into two i16x16
  i16x16 low = extend_low(a) / d;
  i16x16 high = extend_high(a) / d;
  return compress(low, high);
#endif
}

// vector of 32 8-bit unsigned integers
inline u8x32 operator/(const u8x32 a, const DivisorU16 d) {
  // zero-extend even-numbered and odd-numbered elements to 16 bits
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  u16x16 even = _mm256_maskz_mov_epi8(__mmask32(0x55555555), a);
  u16x16 odd = _mm256_srli_epi16(a, 8);
  u16x16 evend = even / d; // divide even-numbered elements
  u16x16 oddd = odd / d; // divide odd-numbered  elements
  oddd = _mm256_slli_epi16(oddd, 8); // shift left to put back in place
  __m256i res = _mm256_mask_mov_epi8(evend, 0xAAAAAAAA, oddd); // interleave even and odd
  return res;
#else
  // expand into two i16x16
  u16x16 low = extend_low(a) / d;
  u16x16 high = extend_high(a) / d;
  return compress(low, high);
#endif
}

// vector operator /= : divide
inline i32x8& operator/=(i32x8& a, const DivisorI32 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline u32x8& operator/=(u32x8& a, const DivisorU32 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline i16x16& operator/=(i16x16& a, const DivisorI16 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline u16x16& operator/=(u16x16& a, const DivisorU16 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline i8x32& operator/=(i8x32& a, const DivisorI16 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline u8x32& operator/=(u8x32& a, const DivisorU16 d) {
  a = a / d;
  return a;
}

/*****************************************************************************
 *
 *          Integer division 2: divisor is a compile-time constant
 *
 *****************************************************************************/

// Divide i32x8 by compile-time constant
template<i32 d>
inline i32x8 divide_by_i(const i32x8 x) {
  static_assert(d != 0, "Integer division by zero");
  if constexpr (d == 1) {
    return x;
  }
  if constexpr (d == -1) {
    return -x;
  }
  if constexpr (u32(d) == 0x80000000) {
    return i32x8(x == i32x8(i32(0x80000000))) & 1; // prevent overflow when changing sign
  }
  constexpr u32 d1 =
    d > 0 ? u32(d)
          : u32(-d); // compile-time abs(d). (force GCC compiler to treat d as 32 bits, not 64 bits)
  if constexpr ((d1 & (d1 - 1)) == 0) {
    // d1 is a power of 2. use shift
    constexpr int k = bit_scan_reverse_const(d1);
    __m256i sign;
    if constexpr (k > 1) {
      sign = _mm256_srai_epi32(x, k - 1);
    } else {
      sign = x; // k copies of sign bit
    }
    __m256i bias = _mm256_srli_epi32(sign, 32 - k); // bias = x >= 0 ? 0 : k-1
    __m256i xpbias = _mm256_add_epi32(x, bias); // x + bias
    __m256i q = _mm256_srai_epi32(xpbias, k); // (x + bias) >> k
    if (d > 0) {
      return q; // d > 0: return  q
    }
    return _mm256_sub_epi32(_mm256_setzero_si256(), q); // d < 0: return -q
  }
  // general case
  constexpr i32 sh =
    bit_scan_reverse_const(u32(d1) - 1); // ceil(log2(d1)) - 1. (d1 < 2 handled by power of 2 case)
  constexpr i32 mult = int(1 + (u64(1) << (32 + sh)) / u32(d1) - (i64(1) << 32)); // multiplier
  const DivisorI32 div(mult, sh, d < 0 ? -1 : 0);
  return x / div;
}

// define i32x8 a / const_int(d)
template<i32 d>
inline i32x8 operator/(const i32x8 a, ConstInt<d> /*b*/) {
  return divide_by_i<d>(a);
}

// define i32x8 a / const_uint(d)
template<u32 d>
inline i32x8 operator/(const i32x8 a, ConstUint<d> /*b*/) {
  static_assert(d < 0x80000000, "Dividing signed integer by overflowing unsigned");
  return divide_by_i<i32(d)>(a); // signed divide
}

// vector operator /= : divide
template<i32 d>
inline i32x8& operator/=(i32x8& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<u32 d>
inline i32x8& operator/=(i32x8& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// Divide u32x8 by compile-time constant
template<u32 d>
inline u32x8 divide_by_ui(const u32x8 x) {
  static_assert(d != 0, "Integer division by zero");
  if constexpr (d == 1) {
    return x; // divide by 1
  }
  constexpr int b = bit_scan_reverse_const(d); // floor(log2(d))
  if constexpr ((d & (d - 1)) == 0) {
    // d is a power of 2. use shift
    return _mm256_srli_epi32(x, b); // x >> b
  }
  // general case (d > 2)
  constexpr u32 mult = u32((u64(1) << (b + 32)) / d); // multiplier = 2^(32+b) / d
  constexpr u64 rem = (u64(1) << (b + 32)) - u64(d) * mult; // remainder 2^(32+b) % d
  constexpr bool round_down = (2 * rem < d); // check if fraction is less than 0.5
  constexpr u32 mult1 = round_down ? mult : mult + 1;
  // do 32*32->64 bit unsigned multiplication and get high part of result
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  const __m256i multv = _mm256_maskz_set1_epi32(0x55, mult1); // zero-extend mult and broadcast
#else
  const __m256i multv = u64x4(u64(mult1)); // zero-extend mult and broadcast
#endif
  __m256i t1 = _mm256_mul_epu32(x, multv); // 32x32->64 bit unsigned multiplication of x[0] and x[2]
  if constexpr (round_down) {
    t1 = _mm256_add_epi64(
      t1, multv); // compensate for rounding error. (x+1)*m replaced by x*m+m to avoid overflow
  }
  __m256i t2 = _mm256_srli_epi64(t1, 32); // high dword of result 0 and 2
  __m256i t3 = _mm256_srli_epi64(x, 32); // get x[1] and x[3] into position for multiplication
  __m256i t4 =
    _mm256_mul_epu32(t3, multv); // 32x32->64 bit unsigned multiplication of x[1] and x[3]
  if constexpr (round_down) {
    t4 = _mm256_add_epi64(
      t4, multv); // compensate for rounding error. (x+1)*m replaced by x*m+m to avoid overflow
  }
  __m256i t7 = _mm256_blend_epi32(t2, t4, 0xAA);
  u32x8 q = _mm256_srli_epi32(t7, b); // shift right by b
  return q; // no overflow possible
}

// define u32x8 a / const_uint(d)
template<u32 d>
inline u32x8 operator/(const u32x8 a, ConstUint<d> /*b*/) {
  return divide_by_ui<d>(a);
}

// define u32x8 a / const_int(d)
template<i32 d>
inline u32x8 operator/(const u32x8 a, ConstInt<d> /*b*/) {
  static_assert(d >= 0, "Dividing unsigned integer by negative is ambiguous");
  return divide_by_ui<d>(a); // unsigned divide
}

// vector operator /= : divide
template<u32 d>
inline u32x8& operator/=(u32x8& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<i32 d>
inline u32x8& operator/=(u32x8& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// Divide i16x16 by compile-time constant
template<int d>
inline i16x16 divide_by_i(const i16x16 x) {
  constexpr i16 d0 = i16(d); // truncate d to 16 bits
  static_assert(d0 != 0, "Integer division by zero");
  if constexpr (d0 == 1) {
    return x; // divide by  1
  }
  if constexpr (d0 == -1) {
    return -x; // divide by -1
  }
  if constexpr (u16(d0) == 0x8000U) {
    return i16x16(x == i16x16(i16(0x8000))) & 1; // prevent overflow when changing sign
  }
  constexpr u16 d1 = d0 > 0 ? d0 : -d0; // compile-time abs(d0)
  if constexpr ((d1 & (d1 - 1)) == 0) {
    // d is a power of 2. use shift
    constexpr int k = bit_scan_reverse_const(u32(d1));
    __m256i sign;
    if constexpr (k > 1) {
      sign = _mm256_srai_epi16(x, k - 1);
    } else {
      sign = x; // k copies of sign bit
    }
    __m256i bias = _mm256_srli_epi16(sign, 16 - k); // bias = x >= 0 ? 0 : k-1
    __m256i xpbias = _mm256_add_epi16(x, bias); // x + bias
    __m256i q = _mm256_srai_epi16(xpbias, k); // (x + bias) >> k
    if constexpr (d0 > 0) {
      return q; // d0 > 0: return  q
    }
    return _mm256_sub_epi16(_mm256_setzero_si256(), q); // d0 < 0: return -q
  }
  // general case
  // ceil(log2(d)). (d < 2 handled above)
  constexpr int arr = bit_scan_reverse_const(u16(d1 - 1)) + 1;
  constexpr i16 mult = i16(1 + (1U << (15 + arr)) / u32(d1) - 0x10000); // multiplier
  constexpr int shift1 = arr - 1;
  const DivisorI16 div(mult, shift1, d0 > 0 ? 0 : -1);
  return x / div;
}

// define i16x16 a / const_int(d)
template<int d>
inline i16x16 operator/(const i16x16 a, ConstInt<d> /*b*/) {
  return divide_by_i<d>(a);
}

// define i16x16 a / const_uint(d)
template<u32 d>
inline i16x16 operator/(const i16x16 a, ConstUint<d> /*b*/) {
  static_assert(d < 0x8000, "Dividing signed integer by overflowing unsigned");
  return divide_by_i<int(d)>(a); // signed divide
}

// vector operator /= : divide
template<i32 d>
inline i16x16& operator/=(i16x16& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<u32 d>
inline i16x16& operator/=(i16x16& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// Divide u16x16 by compile-time constant
template<u32 d>
inline u16x16 divide_by_ui(const u16x16 x) {
  constexpr u16 d0 = u16(d); // truncate d to 16 bits
  static_assert(d0 != 0, "Integer division by zero");
  if constexpr (d0 == 1) {
    return x; // divide by 1
  }
  constexpr int b = bit_scan_reverse_const((u32)d0); // floor(log2(d))
  if constexpr ((d0 & (d0 - 1)) == 0) {
    // d is a power of 2. use shift
    return _mm256_srli_epi16(x, b); // x >> b
  }
  // general case (d > 2)
  constexpr u16 mult = u16((u32(1) << (b + 16)) / d0); // multiplier = 2^(32+b) / d
  constexpr u32 rem = (u32(1) << (b + 16)) - u32(d0) * mult; // remainder 2^(32+b) % d
  constexpr bool round_down = (2 * rem < d0); // check if fraction is less than 0.5
  u16x16 x1 = x;
  if constexpr (round_down) {
    x1 = x1 + u16x16(1); // round down mult and compensate by adding 1 to x
  }
  constexpr u16 mult1 = round_down ? mult : mult + 1;
  const __m256i multv = _mm256_set1_epi16(i16(mult1)); // broadcast mult
  __m256i xm = _mm256_mulhi_epu16(x1, multv); // high part of 16x16->32 bit unsigned multiplication
  u16x16 q = _mm256_srli_epi16(xm, b); // shift right by b
  if constexpr (round_down) {
    Mask<16, 16> overfl = (x1 == u16x16(_mm256_setzero_si256())); // check for overflow of x+1
    // deal with overflow (rarely needed)
    return select(overfl, u16x16(u16(mult1 >> (u16)b)), q);
  } else {
    return q; // no overflow possible
  }
}

// define u16x16 a / const_uint(d)
template<u32 d>
inline u16x16 operator/(const u16x16 a, ConstUint<d> /*b*/) {
  return divide_by_ui<d>(a);
}

// define u16x16 a / const_int(d)
template<int d>
inline u16x16 operator/(const u16x16 a, ConstInt<d> /*b*/) {
  static_assert(d >= 0, "Dividing unsigned integer by negative is ambiguous");
  return divide_by_ui<d>(a); // unsigned divide
}

// vector operator /= : divide
template<u32 d>
inline u16x16& operator/=(u16x16& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<i32 d>
inline u16x16& operator/=(u16x16& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// define i8x32 a / const_int(d)
template<int d>
inline i8x32 operator/(const i8x32 a, ConstInt<d> /*b*/) {
  // expand into two i16x16
  i16x16 low = extend_low(a) / ConstInt<d>();
  i16x16 high = extend_high(a) / ConstInt<d>();
  return compress(low, high);
}

// define i8x32 a / const_uint(d)
template<u32 d>
inline i8x32 operator/(const i8x32 a, ConstUint<d> /*b*/) {
  static_assert(u8(d) < 0x80, "Dividing signed integer by overflowing unsigned");
  return a / ConstInt<d>(); // signed divide
}

// vector operator /= : divide
template<i32 d>
inline i8x32& operator/=(i8x32& a, ConstInt<d> b) {
  a = a / b;
  return a;
}
// vector operator /= : divide
template<u32 d>
inline i8x32& operator/=(i8x32& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// define u8x32 a / const_uint(d)
template<u32 d>
inline u8x32 operator/(const u8x32 a, ConstUint<d> /*b*/) {
  // expand into two u16x16
  u16x16 low = extend_low(a) / ConstUint<d>();
  u16x16 high = extend_high(a) / ConstUint<d>();
  return compress(low, high);
}

// define u8x32 a / const_int(d)
template<int d>
inline u8x32 operator/(const u8x32 a, ConstInt<d> /*b*/) {
  static_assert(i8(d) >= 0, "Dividing unsigned integer by negative is ambiguous");
  return a / ConstUint<d>(); // unsigned divide
}

// vector operator /= : divide
template<u32 d>
inline u8x32& operator/=(u8x32& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<i32 d>
inline u8x32& operator/=(u8x32& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

/*****************************************************************************
 *
 *          Boolean <-> bitfield conversion functions
 *
 *****************************************************************************/

#if STADO_INSTRUCTION_SET < STADO_AVX512SKL // compact boolean vectors, other sizes
// to_bits: convert boolean vector to integer bitfield
inline u32 to_bits(const Mask<8, 32> x) {
  return u32(_mm256_movemask_epi8(x));
}

inline u16 to_bits(const Mask<16, 16> x) {
  __m128i a = _mm_packs_epi16(x.get_low(), x.get_high()); // 16-bit words to bytes
  return u16(_mm_movemask_epi8(a));
}
#endif
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_256I_HPP
