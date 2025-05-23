#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_128I_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_128I_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/base.hpp"
#include "stado/mask/compact.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/divisor.hpp"
#include "stado/vector/native/operations/i08x16.hpp"
#include "stado/vector/native/operations/i16x08.hpp"
#include "stado/vector/native/operations/i64x02.hpp"
#include "stado/vector/native/operations/u16x08.hpp"
#include "stado/vector/native/types.hpp"

namespace stado {
// find lowest and highest index
template<std::size_t tSize>
constexpr int min_index(const std::array<int, tSize>& a) {
  int ix = a[0];
  for (std::size_t i = 1; i < tSize; ++i) {
    if (a[i] < ix) {
      ix = a[i];
    }
  }
  return ix;
}

template<std::size_t tSize>
constexpr int max_index(const std::array<int, tSize>& a) {
  int ix = a[0];
  for (std::size_t i = 1; i < tSize; ++i) {
    if (a[i] > ix) {
      ix = a[i];
    }
  }
  return ix;
}

// permute i64x2
template<int i0, int i1>
inline i64x2 permute2(const i64x2 a) {
  static constexpr std::array<int, 2> indexs{i0, i1}; // indexes as array
  __m128i y = a; // result
  // get flags for possibilities that fit the permutation pattern
  static constexpr PermFlags flags = perm_flags<i64x2>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm_setzero_si128(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed
    // try to fit various instructions

    if constexpr (flags.shleft && flags.zeroing) {
      // pslldq does both permutation and zeroing. if zeroing not needed use punpckl instead
      return _mm_bslli_si128(a, 8);
    }
    if constexpr (flags.shright && flags.zeroing) {
      // psrldq does both permutation and zeroing. if zeroing not needed use punpckh instead
      return _mm_bsrli_si128(a, 8);
    }
    if constexpr (flags.punpckh) { // fits punpckhi
      y = _mm_unpackhi_epi64(a, a);
    } else if constexpr (flags.punpckl) { // fits punpcklo
      y = _mm_unpacklo_epi64(a, a);
    } else { // needs general permute
      y = _mm_shuffle_epi32(a, i0 * 0x0A + i1 * 0xA0 + 0x44);
    }
  }
  if constexpr (flags.zeroing) {
    // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm_maskz_mov_epi64(zero_mask<2>(indexs), y);
#else // use unpack to avoid using data cache
    if constexpr (i0 == -1) {
      y = _mm_unpackhi_epi64(_mm_setzero_si128(), y);
    } else if constexpr (i1 == -1) {
      y = _mm_unpacklo_epi64(y, _mm_setzero_si128());
    }
#endif
  }
  return y;
}

template<int i0, int i1>
inline u64x2 permute2(const u64x2 a) {
  return u64x2(permute2<i0, i1>((i64x2)a));
}

// permute i32x4
template<int i0, int i1, int i2, int i3>
inline i32x4 permute4(const i32x4 a) {
  static constexpr std::array<int, 4> indexs{i0, i1, i2, i3}; // indexes as array
  __m128i y = a; // result

  // get flags for possibilities that fit the permutation pattern
  static constexpr PermFlags flags = perm_flags<i32x4>(indexs);

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm_setzero_si128();
  }

  if constexpr (flags.perm) { // permutation needed
    if constexpr (flags.largeblock) {
      // use larger permutation
      static constexpr std::array<int, 2> arr = largeblock_perm<4>(indexs); // permutation pattern
      y = permute2<arr[0], arr[1]>(i64x2(a));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.shleft) { // fits pslldq
      y = _mm_bslli_si128(a, 16 - flags.rot_count);
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.shright) { // fits psrldq
      y = _mm_bsrli_si128(a, flags.rot_count);
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3 && STADO_INSTRUCTION_SET < STADO_AVX512SKL
    // SSSE3, but no compact mask
    else if constexpr (flags.zeroing) {
      // Do both permutation and zeroing with PSHUFB instruction
      // (bm is constexpr rather than const to make sure it is calculated at compile time)
      static constexpr std::array<i8, 16> bm = pshufb_mask<i32x4>(indexs);
      return _mm_shuffle_epi8(a, i32x4().load(bm.data()));
    }
#endif
    else if constexpr (flags.punpckh) { // fits punpckhi
      y = _mm_unpackhi_epi32(a, a);
    } else if constexpr (flags.punpckl) { // fits punpcklo
      y = _mm_unpacklo_epi32(a, a);
    }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
    else if constexpr (flags.rotate) { // fits palignr
      y = _mm_alignr_epi8(a, a, flags.rot_count);
    }
#endif
    else { // needs general permute
      y = _mm_shuffle_epi32(a, (i0 & 3U) | (i1 & 3U) << 2U | (i2 & 3U) << 4U | (i3 & 3U) << 6U);
    }
  }
  if constexpr (flags.zeroing) {
    // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    // The mask-zero operation can be merged into the preceding instruction, whatever that is.
    // A good optimizing compiler will do this automatically.
    // I don't want to clutter all the branches above with this
    y = _mm_maskz_mov_epi32(zero_mask<4>(indexs), y);
#else // use broad mask
    static constexpr std::array<i32, 4> bm = zero_mask_broad<i32, 4>(indexs);
    y = _mm_and_si128(i32x4().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3>
inline u32x4 permute4(const u32x4 a) {
  return u32x4(permute4<i0, i1, i2, i3>(i32x4(a)));
}

// permute i16x8
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline i16x8 permute8(const i16x8 a) {
  // indexes as array
  static constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7};
  // get flags for possibilities that fit the permutation pattern
  static constexpr PermFlags flags = perm_flags<i16x8>(indexs);
  static constexpr u64 flags16 = perm16_flags<i16x8>(indexs);

  static constexpr bool fit_zeroing = flags.zeroing; // needs additional zeroing
  static constexpr bool l2l = (flags16 & 1U) != 0; // from low  to low  64-bit part
  static constexpr bool h2h = (flags16 & 2U) != 0; // from high to high 64-bit part
  static constexpr bool h2l = (flags16 & 4U) != 0; // from high to low  64-bit part
  static constexpr bool l2h = (flags16 & 8U) != 0; // from low  to high 64-bit part
  static constexpr u8 pl2l = u8(flags16 >> 32U); // low  to low  permute pattern
  static constexpr u8 ph2h = u8(flags16 >> 40U); // high to high permute pattern
  static constexpr u8 noperm = 0xE4; // pattern for no permute

  __m128i y = a; // result

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm_setzero_si128();
  }

  if constexpr (flags.perm) {
    // permutation needed
    if constexpr (flags.largeblock) {
      // use larger permutation
      static constexpr std::array<int, 4> arr = largeblock_perm<8>(indexs); // permutation pattern
      y = permute4<arr[0], arr[1], arr[2], arr[3]>(i32x4(a));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.shleft && !flags.addz) { // fits pslldq
      return _mm_bslli_si128(a, 16 - flags.rot_count);
    } else if constexpr (flags.shright && !flags.addz) { // fits psrldq
      return _mm_bsrli_si128(a, flags.rot_count);
    } else if constexpr (flags.broadcast && !flags.zeroing && flags.rot_count == 0) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
      return _mm_broadcastw_epi16(y);
#else
      y = _mm_shufflelo_epi16(a, 0); // broadcast of first element
      return _mm_unpacklo_epi64(y, y);
#endif
    }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3 && STADO_INSTRUCTION_SET < STADO_AVX512SKL
    // SSSE3, but no compact mask
    else if constexpr (fit_zeroing) {
      // Do both permutation and zeroing with PSHUFB instruction
      static constexpr std::array<i8, 16> bm = pshufb_mask<i16x8>(indexs);
      return _mm_shuffle_epi8(a, i16x8().load(bm.data()));
    }
#endif
    else if constexpr (flags.punpckh) { // fits punpckhi
      y = _mm_unpackhi_epi16(a, a);
    } else if constexpr (flags.punpckl) { // fits punpcklo
      y = _mm_unpacklo_epi16(a, a);
    }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
    else if constexpr (flags.rotate) { // fits palignr
      y = _mm_alignr_epi8(a, a, flags.rot_count);
    }
#endif
    else if constexpr (!h2l && !l2h) { // no crossing of 64-bit boundary
      if constexpr (l2l && pl2l != noperm) {
        y = _mm_shufflelo_epi16(y, pl2l); // permute low 64-bits
      }
      if constexpr (h2h && ph2h != noperm) {
        y = _mm_shufflehi_epi16(y, ph2h); // permute high 64-bits
      }
    }
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI2__)
    else if constexpr (flags.compress) {
      y = _mm_maskz_compress_epi16(__mmask8(compress_mask(indexs)), y); // compress
      if constexpr (!flags.addz2) {
        return y;
      }
    } else if constexpr (flags.expand) {
      y = _mm_maskz_expand_epi16(__mmask8(expand_mask(indexs)), y); // expand
      if constexpr (!flags.addz2) {
        return y;
      }
    }
#endif // AVX512VBMI2
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
    else { // needs general permute
      static constexpr std::array<i8, 16> bm = pshufb_mask<i16x8>(indexs);
      y = _mm_shuffle_epi8(a, i16x8().load(bm.data()));
      return y; // _mm_shuffle_epi8 also does zeroing
    }
  }
  if constexpr (fit_zeroing) {
    // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm_maskz_mov_epi16(zero_mask<8>(indexs), y);
#else // use broad mask
    static constexpr std::array<i16, 8> bm = zero_mask_broad<i16, 8>(indexs);
    y = _mm_and_si128(i16x8().load(bm.data()), y);
#endif
  }
  return y;
#else
    else {
      // Difficult case. Use permutations of low and high half separately
      static constexpr u8 ph2l = u8(flags16 >> 48); // high to low  permute pattern
      static constexpr u8 pl2h = u8(flags16 >> 56); // low  to high permute pattern
      __m128i yswap = _mm_shuffle_epi32(y, 0x4E); // swap low and high 64-bits
      if constexpr (h2l && ph2l != noperm) {
        yswap = _mm_shufflelo_epi16(yswap, ph2l); // permute low 64-bits
      }
      if constexpr (l2h && pl2h != noperm) {
        yswap = _mm_shufflehi_epi16(yswap, pl2h); // permute high 64-bits
      }
      if constexpr (l2l && pl2l != noperm) {
        y = _mm_shufflelo_epi16(y, pl2l); // permute low 64-bits
      }
      if constexpr (h2h && ph2h != noperm) {
        y = _mm_shufflehi_epi16(y, ph2h); // permute high 64-bits
      }
      if constexpr (h2h || l2l) { // merge data from y and yswap
        static constexpr auto selb =
          make_bit_mask<8, 0x102>(indexs); // blend by bit 2. invert upper half
        static constexpr std::array<i16, 8> bm =
          make_broad_mask<i16, 8>(selb); // convert to broad mask
        y = selectb(i16x8().load(bm.data()), yswap, y);
      } else {
        y = yswap;
      }
    }
  }
  if constexpr (fit_zeroing) {
    // additional zeroing needed
    static constexpr std::array<i16, 8> bm = zero_mask_broad<i16, 8>(indexs);
    y = _mm_and_si128(i16x8().load(bm.data()), y);
  }
  return y;
#endif
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline u16x8 permute8(const u16x8 a) {
  return u16x8(permute8<i0, i1, i2, i3, i4, i5, i6, i7>(i16x8(a)));
}

// permute i8x16
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline i8x16 permute16(const i8x16 a) {
  // indexes as array
  static constexpr std::array<int, 16> indexs{i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                              i8, i9, i10, i11, i12, i13, i14, i15};
  // get flags for possibilities that fit the permutation pattern
  static constexpr PermFlags flags = perm_flags<i8x16>(indexs);

  static constexpr bool fit_zeroing = flags.zeroing; // needs additional zeroing

  __m128i y = a; // result

  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm_setzero_si128();
  }

  if constexpr (flags.perm) {
    // permutation needed

    if constexpr (flags.largeblock) {
      // use larger permutation
      static constexpr std::array<int, 8> arr = largeblock_perm<16>(indexs); // permutation pattern
      y = permute8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(i16x8(a));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.shleft) { // fits pslldq
      y = _mm_bslli_si128(a, 16 - flags.rot_count);
      if (!flags.addz) {
        return y;
      }
    } else if constexpr (flags.shright) { // fits psrldq
      y = _mm_bsrli_si128(a, flags.rot_count);
      if (!flags.addz) {
        return y;
      }
    }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3 && STADO_INSTRUCTION_SET < STADO_AVX512SKL
    // SSSE3, but no compact mask
    else if constexpr (fit_zeroing) {
      // Do both permutation and zeroing with PSHUFB instruction
      static constexpr std::array<stado::i8, 16> bm = pshufb_mask<i8x16>(indexs);
      return _mm_shuffle_epi8(a, i8x16().load(bm.data()));
    }
#endif
    else if constexpr (flags.punpckh) { // fits punpckhi
      y = _mm_unpackhi_epi8(a, a);
    } else if constexpr (flags.punpckl) { // fits punpcklo
      y = _mm_unpacklo_epi8(a, a);
    }
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI2__)
    else if constexpr (flags.compress) {
      y = _mm_maskz_compress_epi8(__mmask16(compress_mask(indexs)), y); // compress
      if constexpr (!flags.addz2) {
        return y;
      }
    } else if constexpr (flags.expand) {
      y = _mm_maskz_expand_epi8(__mmask16(expand_mask(indexs)), y); // expand
      if constexpr (!flags.addz2) {
        return y;
      }
    }
#endif // AVX512VBMI2
#if STADO_INSTRUCTION_SET >= STADO_AVX2
    else if constexpr (flags.broadcast && (!flags.zeroing || !fit_zeroing) == 0 &&
                       flags.rot_count == 0) {
      return _mm_broadcastb_epi8(y);
    }
#endif
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
    else if constexpr (flags.rotate) { // fits palignr
      y = _mm_alignr_epi8(a, a, flags.rot_count);
    } else { // needs general permute
      static constexpr std::array<stado::i8, 16> bm = pshufb_mask<i8x16>(indexs);
      y = _mm_shuffle_epi8(a, i8x16().load(bm.data()));
      return y; // _mm_shuffle_epi8 also does zeroing
    }
  }
#else
    else {
      // Difficult case. Use permutations of low and high half separately
      i8x16 swapped, te2e, te2o, to2e, to2o, combeven, combodd;

      // get permutation indexes for four 16-bit permutes:
      // k = 0: e2e: even bytes of source to even bytes of destination
      // k = 1: o2e: odd  bytes of source to even bytes of destination
      // k = 2: e2o: even bytes of source to odd  bytes of destination
      // k = 3: o2o: odd  bytes of source to odd  bytes of destination
      auto eoperm = [](const u8 k, const std::array<int, 16>& indexs) {
        u8 ix = 0; // index element
        u64 r = 0; // return value
        u8 i = (k >> 1) & 1; // look at odd indexes if destination is odd
        for (; i < 16; i += 2) {
          ix = (indexs[i] >= 0 && ((indexs[i] ^ k) & 1) == 0) ? (u8)indexs[i] / 2U : 0xFFU;
          r |= u64(ix) << (i / 2U * 8U);
        }
        return r;
      };
      static constexpr u64 ixe2e = eoperm(0, indexs);
      static constexpr u64 ixo2e = eoperm(1, indexs);
      static constexpr u64 ixe2o = eoperm(2, indexs);
      static constexpr u64 ixo2o = eoperm(3, indexs);

      // even bytes of source to odd  bytes of destination
      static constexpr bool e2e = ixe2e != -1;
      // even bytes of source to odd  bytes of destination
      static constexpr bool e2o = ixe2o != -1;
      // odd  bytes of source to even bytes of destination
      static constexpr bool o2e = ixo2e != -1;
      // odd  bytes of source to odd  bytes of destination
      static constexpr bool o2o = ixo2o != -1;

      if constexpr (e2o || o2e) {
        swapped = (__m128i)rotate_left(i16x8(a), 8); // swap odd and even bytes
      }

      if constexpr (e2e) {
        te2e = permute8<stado::i8(ixe2e), stado::i8(ixe2e >> 8), stado::i8(ixe2e >> 16),
                        stado::i8(ixe2e >> 24), stado::i8(ixe2e >> 32), stado::i8(ixe2e >> 40),
                        stado::i8(ixe2e >> 48), stado::i8(ixe2e >> 56)>(i16x8(a));
      }

      if constexpr (e2o) {
        te2o = permute8<stado::i8(ixe2o), stado::i8(ixe2o >> 8), stado::i8(ixe2o >> 16),
                        stado::i8(ixe2o >> 24), stado::i8(ixe2o >> 32), stado::i8(ixe2o >> 40),
                        stado::i8(ixe2o >> 48), stado::i8(ixe2o >> 56)>(i16x8(swapped));
      }

      if constexpr (o2e) {
        to2e = permute8<stado::i8(ixo2e), stado::i8(ixo2e >> 8), stado::i8(ixo2e >> 16),
                        stado::i8(ixo2e >> 24), stado::i8(ixo2e >> 32), stado::i8(ixo2e >> 40),
                        stado::i8(ixo2e >> 48), stado::i8(ixo2e >> 56)>(i16x8(swapped));
      }

      if constexpr (o2o) {
        to2o = permute8<stado::i8(ixo2o), stado::i8(ixo2o >> 8), stado::i8(ixo2o >> 16),
                        stado::i8(ixo2o >> 24), stado::i8(ixo2o >> 32), stado::i8(ixo2o >> 40),
                        stado::i8(ixo2o >> 48), stado::i8(ixo2o >> 56)>(i16x8(a));
      }

      if constexpr (e2e && o2e) {
        combeven = {te2e | to2e};
      } else if constexpr (e2e) {
        combeven = te2e;
      } else if constexpr (o2e) {
        combeven = to2e;
      } else {
        combeven = _mm_setzero_si128();
      }

      if constexpr (e2o && o2o) {
        combodd = {te2o | to2o};
      } else if constexpr (e2o) {
        combodd = te2o;
      } else if constexpr (o2o) {
        combodd = to2o;
      } else {
        combodd = _mm_setzero_si128();
      }

      // mask used even bytes
      __m128i maske = _mm_setr_epi32((i0 < 0 ? 0 : 0xFF) | (i2 < 0 ? 0 : 0xFF0000),
                                     (i4 < 0 ? 0 : 0xFF) | (i6 < 0 ? 0 : 0xFF0000),
                                     (i8 < 0 ? 0 : 0xFF) | (i10 < 0 ? 0 : 0xFF0000),
                                     (i12 < 0 ? 0 : 0xFF) | (i14 < 0 ? 0 : 0xFF0000));
      // mask used odd bytes
      __m128i masko = _mm_setr_epi32((i1 < 0 ? 0 : 0xFF00) | (i3 < 0 ? 0 : 0xFF000000),
                                     (i5 < 0 ? 0 : 0xFF00) | (i7 < 0 ? 0 : 0xFF000000),
                                     (i9 < 0 ? 0 : 0xFF00) | (i11 < 0 ? 0 : 0xFF000000),
                                     (i13 < 0 ? 0 : 0xFF00) | (i15 < 0 ? 0 : 0xFF000000));

      return _mm_or_si128( // combine even and odd bytes
        _mm_and_si128(combeven, maske), _mm_and_si128(combodd, masko));
    }
  }
#endif
  if constexpr (fit_zeroing) {
    // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm_maskz_mov_epi8(zero_mask<16>(indexs), y);
#else // use broad mask
    static constexpr std::array<stado::i8, 16> bm = zero_mask_broad<i8, 16>(indexs);
    y = _mm_and_si128(i8x16().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline u8x16 permute16(const u8x16 a) {
  return u8x16(
    permute16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15>(i8x16(a)));
}

/*****************************************************************************
 *
 *          NativeVector blend functions
 *
 ******************************************************************************
 *
 * These blend functions can mix elements from two different vectors of N
 * elements eadh and optionally set some elements to zero.
 *
 * The N indexes are inserted as template parameters in <>.
 * These indexes must be compile-time constants. Each template parameter
 * selects an element from one of the input vectors a and b.
 * An index in the range 0 .. N-1 selects the corresponding element from a.
 * An index in the range N .. 2*N-1 selects an element from b.
 * An index with the value -1 gives zero in the corresponding element of
 * the result.
 * An index with the value any means don't care. The code will select the
 * optimal sequence of instructions that fits the remaining indexes.
 *
 * Example:
 * i32x4 a(100,101,102,103);         // a is (100, 101, 102, 103)
 * i32x4 b(200,201,202,203);         // b is (200, 201, 202, 203)
 * i32x4 c;
 * c = blend4<1,4,-1,7> (a,b);       // c is (101, 200,   0, 203)
 *
 * A lot of the code here is metaprogramming aiming to find the instructions
 * that best fit the template parameters and instruction set. The metacode
 * will be reduced out to leave only a few vector instructions in release
 * mode with optimization on.
 *****************************************************************************/

// permute and blend i64x2
template<int i0, int i1>
inline i64x2 blend2(const i64x2 a, const i64x2 b) {
  static constexpr std::array<int, 2> indexs{i0, i1}; // indexes as array
  __m128i y = a; // result
  static constexpr u64 flags =
    blend_flags<i64x2>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm_setzero_si128();
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute2<i0, i1>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    return permute2<(i0 < 0 ? i0 : i0 & 1), (i1 < 0 ? i1 : i1 & 1)>(b);
  }

  if constexpr ((flags & (blend_perma | blend_permb)) == 0) { // no permutation, only blending
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm_mask_mov_epi64(a, (u8)make_bit_mask<2, 0x301>(indexs), b);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
    y = _mm_blend_epi16(a, b, ((i0 & 2) ? 0x0F : 0) | ((i1 & 2) ? 0xF0 : 0));
#else // SSE2
        static constexpr std::array<i64, 2> bm =
          make_broad_mask<i64, 2>(make_bit_mask<2, 0x301>(indexs));
        y = selectb(i64x2().load(bm.data()), b, a);
#endif
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm_unpacklo_epi64(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm_unpacklo_epi64(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm_unpackhi_epi64(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm_unpackhi_epi64(b, a);
  }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  else if constexpr ((flags & blend_rotateab) != 0) {
    y = _mm_alignr_epi8(a, b, flags >> blend_rotpattern);
  } else if constexpr ((flags & blend_rotateba) != 0) {
    y = _mm_alignr_epi8(b, a, flags >> blend_rotpattern);
  }
#endif
#if ALLOW_FP_PERMUTE // allow floating point permute instructions on integer vectors
  else if constexpr ((flags & blend_shufab) != 0) { // use floating point instruction shufpd
    y = _mm_castpd_si128(
      _mm_shuffle_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b), (flags >> blend_shufpattern) & 3));
  } else if constexpr ((flags & blend_shufba) != 0) { // use floating point instruction shufpd
    y = _mm_castpd_si128(
      _mm_shuffle_pd(_mm_castsi128_pd(b), _mm_castsi128_pd(a), (flags >> blend_shufpattern) & 3));
  }
#endif
  else { // No special cases. permute a and b separately, then blend.
         // This will not occur if ALLOW_FP_PERMUTE is true
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
    static constexpr bool dozero = false;
#else // SSE2
    static constexpr bool dozero = true;
#endif
    // get permutation indexes
    static constexpr std::array<int, 4> arr = blend_perm_indexes<2, (int)dozero>(indexs);
    __m128i ya = permute2<arr[0], arr[1]>(a);
    __m128i yb = permute2<arr[2], arr[3]>(b);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm_mask_mov_epi64(ya, (u8)make_bit_mask<2, 0x301>(indexs), yb);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
    y = _mm_blend_epi16(ya, yb, ((i0 & 2) ? 0x0F : 0) | ((i1 & 2) ? 0xF0 : 0));
#else // SSE2
        return _mm_or_si128(ya, yb);
#endif
  }

  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm_maskz_mov_epi64(zero_mask<2>(indexs), y);
#else // use broad mask
    static constexpr std::array<i64, 2> bm = zero_mask_broad<i64, 2>(indexs);
    y = _mm_and_si128(i64x2().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1>
inline u64x2 blend2(const u64x2 a, const u64x2 b) {
  return u64x2(blend2<i0, i1>(i64x2(a), i64x2(b)));
}

// permute and blend i32x4
template<int i0, int i1, int i2, int i3>
inline i32x4 blend4(const i32x4 a, const i32x4 b) {
  static constexpr std::array<int, 4> indexs{i0, i1, i2, i3}; // indexes as array
  __m128i y = a; // result
  static constexpr u64 flags =
    blend_flags<i32x4>(indexs); // get flags for possibilities that fit the index pattern

  static constexpr bool blendonly =
    (flags & (blend_perma | blend_permb)) == 0; // no permutation, only blending

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm_setzero_si128();
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute4<i0, i1, i2, i3>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    return permute4<(i0 < 0 ? i0 : i0 & 3), (i1 < 0 ? i1 : i1 & 3), (i2 < 0 ? i2 : i2 & 3),
                    (i3 < 0 ? i3 : i3 & 3)>(b);
  }
  if constexpr ((flags & blend_largeblock) != 0) { // fits blending with larger block size
    static constexpr std::array<int, 2> arr = largeblock_indexes<4>(indexs);
    y = blend2<arr[0], arr[1]>(i64x2(a), i64x2(b));
    if constexpr ((flags & blend_addz) == 0) {
      return y; // any zeroing has been done by larger blend
    }
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm_unpacklo_epi32(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm_unpacklo_epi32(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm_unpackhi_epi32(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm_unpackhi_epi32(b, a);
  }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  else if constexpr ((flags & blend_rotateab) != 0) {
    y = _mm_alignr_epi8(a, b, flags >> blend_rotpattern);
  } else if constexpr ((flags & blend_rotateba) != 0) {
    y = _mm_alignr_epi8(b, a, flags >> blend_rotpattern);
  }
#endif
#if ALLOW_FP_PERMUTE // allow floating point permute instructions on integer vectors
  else if constexpr ((flags & blend_shufab) != 0 &&
                     !blendonly) { // use floating point instruction shufps
    y = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), u8(flags >> blend_shufpattern)));
  } else if constexpr ((flags & blend_shufba) != 0 &&
                       !blendonly) { // use floating point instruction shufps
    y = _mm_castps_si128(
      _mm_shuffle_ps(_mm_castsi128_ps(b), _mm_castsi128_ps(a), u8(flags >> blend_shufpattern)));
  }
#endif
  else { // No special cases. permute a and b separately, then blend.
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
    static constexpr bool dozero = false;
#else // SSE2
    static constexpr bool dozero = true;
#endif
    // a and b permuted
    i32x4 ya = a;
    i32x4 yb = b;
    static constexpr std::array<int, 8> arr =
      blend_perm_indexes<4, (int)dozero>(indexs); // get permutation indexes
    if constexpr ((flags & blend_perma) != 0 || dozero) {
      ya = permute4<arr[0], arr[1], arr[2], arr[3]>(a);
    }
    if constexpr ((flags & blend_permb) != 0 || dozero) {
      yb = permute4<arr[4], arr[5], arr[6], arr[7]>(b);
    }
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    y = _mm_mask_mov_epi32(ya, (u8)make_bit_mask<4, 0x302>(indexs), yb);
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
    static constexpr u8 mm =
      ((i0 & 4) ? 0x03 : 0) | ((i1 & 4) ? 0x0C : 0) | ((i2 & 4) ? 0x30 : 0) | ((i3 & 4) ? 0xC0 : 0);
    y = _mm_blend_epi16(ya, yb, mm);
#else // SSE2. dozero = true
        return _mm_or_si128(ya, yb);
#endif
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm_maskz_mov_epi32(zero_mask<4>(indexs), y);
#else // use broad mask
    static constexpr std::array<i32, 4> bm = zero_mask_broad<i32, 4>(indexs);
    y = _mm_and_si128(i32x4().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3>
inline u32x4 blend4(const u32x4 a, const u32x4 b) {
  return u32x4(blend4<i0, i1, i2, i3>(i32x4(a), i32x4(b)));
}

// permute and blend i16x8
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline i16x8 blend8(const i16x8 a, const i16x8 b) {
  constexpr std::array<int, 8> indexs{i0, i1, i2, i3, i4, i5, i6, i7}; // indexes as array
  __m128i y = a; // result
  constexpr u64 flags =
    blend_flags<i16x8>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm_setzero_si128();
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute8<i0, i1, i2, i3, i4, i5, i6, i7>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    return permute8<(i0 < 0 ? i0 : i0 & 7), (i1 < 0 ? i1 : i1 & 7), (i2 < 0 ? i2 : i2 & 7),
                    (i3 < 0 ? i3 : i3 & 7), (i4 < 0 ? i4 : i4 & 7), (i5 < 0 ? i5 : i5 & 7),
                    (i6 < 0 ? i6 : i6 & 7), (i7 < 0 ? i7 : i7 & 7)>(b);
  }
  if constexpr ((flags & blend_largeblock) != 0) { // fits blending with larger block size
    constexpr std::array<int, 4> arr = largeblock_indexes<8>(indexs);
    y = blend4<arr[0], arr[1], arr[2], arr[3]>(i32x4(a), i32x4(b));
    if constexpr ((flags & blend_addz) == 0) {
      return y; // any zeroing has been done by larger blend
    }
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm_unpacklo_epi16(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm_unpacklo_epi16(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm_unpackhi_epi16(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm_unpackhi_epi16(b, a);
  }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  else if constexpr ((flags & blend_rotateab) != 0) {
    y = _mm_alignr_epi8(a, b, flags >> blend_rotpattern);
  } else if constexpr ((flags & blend_rotateba) != 0) {
    y = _mm_alignr_epi8(b, a, flags >> blend_rotpattern);
  }
#endif
  else { // No special cases.
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
    constexpr std::array<i16, 8> bm = perm_mask_broad<i16, 8>(indexs);
    return _mm_maskz_permutex2var_epi16(zero_mask<8>(indexs), a, i16x8().load(bm.data()), b);
#endif
    // full blend instruction not available,
    // permute a and b separately, then blend.
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
    constexpr bool dozero = (flags & blend_zeroing) != 0;
#else // SSE2
    constexpr bool dozero = true;
#endif
    // a and b permuted
    i16x8 ya = a;
    i16x8 yb = b;
    // get permutation indexes
    constexpr std::array<int, 16> arr = blend_perm_indexes<8, (int)dozero>(indexs);
    if constexpr ((flags & blend_perma) != 0 || dozero) {
      ya = permute8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(a);
    }
    if constexpr ((flags & blend_permb) != 0 || dozero) {
      yb = permute8<arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(b);
    }
    if constexpr (dozero) { // unused elements are zero
      return _mm_or_si128(ya, yb);
    } else { // blend ya and yb

#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
      constexpr u8 mm = ((i0 & 8) ? 0x01 : 0) | ((i1 & 8) ? 0x02 : 0) | ((i2 & 8) ? 0x04 : 0) |
                        ((i3 & 8) ? 0x08 : 0) | ((i4 & 8) ? 0x10 : 0) | ((i5 & 8) ? 0x20 : 0) |
                        ((i6 & 8) ? 0x40 : 0) | ((i7 & 8) ? 0x80 : 0);
      y = _mm_blend_epi16(ya, yb, mm);
#endif
    }
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed after special cases
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm_maskz_mov_epi16(zero_mask<8>(indexs), y);
#else // use broad mask
    constexpr std::array<i16, 8> bm = zero_mask_broad<i16, 8>(indexs);
    y = _mm_and_si128(i16x8().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline u16x8 blend8(const u16x8 a, const u16x8 b) {
  return u16x8(blend8<i0, i1, i2, i3, i4, i5, i6, i7>(i16x8(a), i16x8(b)));
}

// permute and blend i8x16
template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline i8x16 blend16(const i8x16 a, const i8x16 b) {
  constexpr std::array<int, 16> indexs{i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                       i8, i9, i10, i11, i12, i13, i14, i15}; // indexes as array
  __m128i y = a; // result
  constexpr u64 flags =
    blend_flags<i8x16>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm_setzero_si128();
  }

  else if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15>(a);
  } else if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    // get permutation indexes
    constexpr std::array<int, 32> arr = blend_perm_indexes<16, 2>(indexs);
    return permute16<arr[16], arr[17], arr[18], arr[19], arr[20], arr[21], arr[22], arr[23],
                     arr[24], arr[25], arr[26], arr[27], arr[28], arr[29], arr[30], arr[31]>(b);
  }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  else if constexpr ((flags & blend_rotateab) != 0) {
    y = _mm_alignr_epi8(a, b, flags >> blend_rotpattern);
  } else if constexpr ((flags & blend_rotateba) != 0) {
    y = _mm_alignr_epi8(b, a, flags >> blend_rotpattern);
  }
#endif
  else if constexpr ((flags & blend_largeblock) != 0) { // fits blending with larger block size
    constexpr std::array<int, 8> arr = largeblock_indexes<16>(indexs);
    y = blend8<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]>(i16x8(a), i16x8(b));
    if constexpr ((flags & blend_addz) == 0) {
      return y; // any zeroing has been done by larger blend
    }
  }
  // check if pattern fits special cases
  else if constexpr ((flags & blend_punpcklab) != 0) {
    y = _mm_unpacklo_epi8(a, b);
  } else if constexpr ((flags & blend_punpcklba) != 0) {
    y = _mm_unpacklo_epi8(b, a);
  } else if constexpr ((flags & blend_punpckhab) != 0) {
    y = _mm_unpackhi_epi8(a, b);
  } else if constexpr ((flags & blend_punpckhba) != 0) {
    y = _mm_unpackhi_epi8(b, a);
  } else { // No special cases. Full permute needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI__) // AVX512VBMI
    constexpr std::array<stado::i8, 16> bm = perm_mask_broad<i8, 16>(indexs);
    return _mm_maskz_permutex2var_epi8(zero_mask<16>(indexs), a, i8x16().load(bm.data()), b);
#endif // __AVX512VBMI__

    // full blend instruction not available,
    // permute a and b separately, then blend.
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    constexpr bool dozero = (flags & blend_zeroing) != 0;
#else // SSE2
    constexpr bool dozero = true;
#endif
    // a and b permuted
    i8x16 ya = a;
    i8x16 yb = b;
    constexpr std::array<int, 32> arr =
      blend_perm_indexes<16, (int)dozero>(indexs); // get permutation indexes
    if constexpr ((flags & blend_perma) != 0 || dozero) {
      ya = permute16<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
                     arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(a);
    }
    if constexpr ((flags & blend_permb) != 0 || dozero) {
      yb = permute16<arr[16], arr[17], arr[18], arr[19], arr[20], arr[21], arr[22], arr[23],
                     arr[24], arr[25], arr[26], arr[27], arr[28], arr[29], arr[30], arr[31]>(b);
    }
    if constexpr (dozero) { // unused fields in ya and yb are zero
      return _mm_or_si128(ya, yb);
    } else {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
      y = _mm_mask_mov_epi8(ya, (__mmask16)make_bit_mask<16, 0x304>(indexs), yb);
#endif
    }
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // use compact mask
    y = _mm_maskz_mov_epi8(zero_mask<16>(indexs), y);
#else // use broad mask
    constexpr std::array<stado::i8, 16> bm = zero_mask_broad<i8, 16>(indexs);
    y = _mm_and_si128(i8x16().load(bm.data()), y);
#endif
  }
  return y;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10,
         int i11, int i12, int i13, int i14, int i15>
inline u8x16 blend16(const u8x16 a, const u8x16 b) {
  return u8x16(blend16<i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15>(
    i8x16(a), i8x16(b)));
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
 * This can be used for several purposes:
 *  - table lookup
 *  - permute or blend with variable indexes
 *  - blend from more than two sources
 *  - gather non-contiguous data
 *
 * An index out of range may produce any value - the actual value produced is
 * implementation dependent and may be different for different instruction
 * sets. An index out of range does not produce an error message or exception.
 *
 * Example:
 * i32x4 a(2,0,0,3);           // index a is (  2,   0,   0,   3)
 * i32x4 b(100,101,102,103);   // table b is (100, 101, 102, 103)
 * i32x4 c;
 * c = lookup4 (a,b);          // c is (102, 100, 100, 103)
 *
 *****************************************************************************/

template<AnyInt8x16 TVec>
inline TVec lookup16(const TVec index, const TVec table) {
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  return _mm_shuffle_epi8(table, index);
#else
  u8 ii[16];
  i8 tt[16];
  i8 rr[16];
  table.store(tt);
  index.store(ii);
  for (int j = 0; j < 16; j++) {
    rr[j] = tt[ii[j]];
  }
  return TVec{}.load(rr);
#endif
}

inline i8x16 lookup32(const i8x16 index, const i8x16 table0, const i8x16 table1) {
#ifdef __XOP__ // AMD XOP instruction set. Use VPPERM
  return (i8x16)_mm_perm_epi8(table0, table1, index);
#elif STADO_INSTRUCTION_SET >= STADO_SSSE3
  // make negative index for values >= 16
  i8x16 r0 = _mm_shuffle_epi8(table0, index + i8x16(0x70));
  // make negative index for values < 16
  i8x16 r1 = _mm_shuffle_epi8(table1, (index ^ i8x16(0x10)) + i8x16(0x70));
  return {r0 | r1};
#else
      u8 ii[16];
      i8 tt[32];
      i8 rr[16];
      table0.store(tt);
      table1.store(tt + 16);
      index.store(ii);
      for (int j = 0; j < 16; j++) {
        rr[j] = tt[ii[j]];
      }
      return i8x16().load(rr);
#endif
}

inline i8x16 lookup(const i8x16 index, const i8* table) {
  u8 ii[16];
  index.store(ii);
  i8 rr[16];
  for (int j = 0; j < 16; j++) {
    rr[j] = table[ii[j]];
  }
  return i8x16().load(rr);
}

template<std::size_t n>
inline i8x16 lookup_bounded(const i8x16 index, const i8* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 16) {
    return lookup16(index, i8x16().load(table));
  }
  if constexpr (n <= 32) {
    return lookup32(index, i8x16().load(table), i8x16().load(table + 16));
  }
  return lookup(index, table);
}

inline i16x8 lookup8(const i16x8 index, const i16x8 table) {
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  return _mm_shuffle_epi8(table, index * i16x8(0x202) + 0x100);
#else
  i16 ii[8], tt[8], rr[8];
  table.store(tt);
  index.store(ii);
  for (int j = 0; j < 8; j++) {
    rr[j] = tt[ii[j]];
  }
  return i16x8().load(rr);
#endif
}

inline i16x8 lookup16(const i16x8 index, const i16x8 table0, const i16x8 table1) {
#ifdef __XOP__ // AMD XOP instruction set. Use VPPERM
  return (i16x8)_mm_perm_epi8(table0, table1, index * 0x202 + 0x100);
#elif STADO_INSTRUCTION_SET >= STADO_SSSE3
  i16x8 r0 = _mm_shuffle_epi8(table0, i8x16(index * i16x8(0x202)) + i8x16(i16x8(0x7170)));
  i16x8 r1 = _mm_shuffle_epi8(table1, i8x16(index * i16x8(0x202) ^ 0x1010) + i8x16(i16x8(0x7170)));
  return i16x8(r0 | r1);
#else
      i16 ii[16], tt[32], rr[16];
      table0.store(tt);
      table1.store(tt + 8);
      index.store(ii);
      for (int j = 0; j < 16; j++) {
        rr[j] = tt[ii[j]];
      }
      return i16x8().load(rr);
#endif
}

inline i16x8 lookup(const i16x8 index, const i16* table) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  i16x8 t1 = _mm_i32gather_epi32(reinterpret_cast<const int*>(table),
                                 __m128i((i32x4(index)) & (i32x4(0x0000FFFF))),
                                 2); // even positions
  i16x8 t2 = _mm_i32gather_epi32(reinterpret_cast<const int*>(table), _mm_srli_epi32(index, 16),
                                 2); // odd  positions
  return blend8<0, 8, 2, 10, 4, 12, 6, 14>(t1, t2);
#else
  u16 ii[8];
  index.store(ii);
  return {table[ii[0]], table[ii[1]], table[ii[2]], table[ii[3]],
          table[ii[4]], table[ii[5]], table[ii[6]], table[ii[7]]};
#endif
}

template<std::size_t n>
inline i16x8 lookup_bounded(const i16x8 index, const i16* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 8) {
    return lookup8(index, i16x8().load(table));
  }
  if constexpr (n <= 16) {
    return lookup16(index, i16x8().load(table), i16x8().load(table + 8));
  }
  // n > 16
  lookup(index, table);
}

template<AnyInt32x4 TIdx, AnyInt32 TVal>
inline NativeVector<TVal, 4> lookup(const TIdx index, const TVal* table) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm_i32gather_epi32((const int*)table, index, 4);
#else
  u32 ii[4];
  index.store(ii);
  return {table[ii[0]], table[ii[1]], table[ii[2]], table[ii[3]]};
#endif
}

template<AnyInt32x4 TIdx, AnyInt32x4 TVal>
inline TVal lookup4(const TIdx index, const TVal table) {
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  return _mm_shuffle_epi8(table, index * 0x04040404 + 0x03020100);
#else
  return {table[index[0]], table[index[1]], table[index[2]], table[index[3]]};
#endif
}

template<AnyInt32x4 TIdx, AnyInt32x4 TVal>
inline TVal lookup8(const TIdx index, const TVal table0, const TVal table1) {
#ifdef __XOP__ // AMD XOP instruction set. Use VPPERM
  return _mm_perm_epi8(table0, table1, index * 0x04040404 + 0x03020100);
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  __m256i table01 = _mm256_inserti128_si256(_mm256_castsi128_si256(table0), table1,
                                            1); // join tables into 256 bit vector
  return _mm256_castsi256_si128(
    _mm256_permutevar8x32_epi32(table01, _mm256_castsi128_si256(index)));

#elif STADO_INSTRUCTION_SET >= STADO_SSSE3
      i32x4 r0 = _mm_shuffle_epi8(table0, i8x16(index * 0x04040404) + i8x16(i32x4(0x73727170)));
      i32x4 r1 =
        _mm_shuffle_epi8(table1, i8x16(index * 0x04040404 ^ 0x10101010) + i8x16(i32x4(0x73727170)));
      return r0 | r1;
#else // SSE2
      i32 ii[4], tt[8], rr[4];
      table0.store(tt);
      table1.store(tt + 4);
      index.store(ii);
      for (int j = 0; j < 4; j++) {
        rr[j] = tt[ii[j]];
      }
      return i32x4().load(rr);
#endif
}

template<AnyInt32x4 TIdx, AnyInt32x4 TVal>
inline TVal lookup16(const TIdx index, const TVal table0, const TVal table1, const TVal table2,
                     const TVal table3) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  __m256i table01 = _mm256_inserti128_si256(_mm256_castsi128_si256(table0), table1,
                                            1); // join tables into 256 bit vector
  __m256i table23 = _mm256_inserti128_si256(_mm256_castsi128_si256(table2), table3,
                                            1); // join tables into 256 bit vector
  TVal r0 =
    _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(table01, _mm256_castsi128_si256(index)));
  TVal r1 =
    _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(table23, _mm256_castsi128_si256(index ^ 8)));
  return select(index >= 8, r1, r0);
#elif defined(__XOP__) // AMD XOP instruction set. Use VPPERM
  i32x4 r0 = _mm_perm_epi8(table0, table1, ((index) * 0x04040404u + 0x63626160u) & 0X9F9F9F9Fu);
  i32x4 r1 = _mm_perm_epi8(table2, table3, ((index ^ 8) * 0x04040404u + 0x63626160u) & 0X9F9F9F9Fu);
  return r0 | r1;
#elif STADO_INSTRUCTION_SET >= STADO_SSSE3
      i8x16 aa = i8x16(i32x4(0x73727170));
      i32x4 r0 = _mm_shuffle_epi8(table0, i8x16((index) * 0x04040404) + aa);
      i32x4 r1 = _mm_shuffle_epi8(table1, i8x16((index ^ 4) * 0x04040404) + aa);
      i32x4 r2 = _mm_shuffle_epi8(table2, i8x16((index ^ 8) * 0x04040404) + aa);
      i32x4 r3 = _mm_shuffle_epi8(table3, i8x16((index ^ 12) * 0x04040404) + aa);
      return (r0 | r1) | (r2 | r3);
#else // SSE2
      i32 ii[4], tt[16], rr[4];
      table0.store(tt);
      table1.store(tt + 4);
      table2.store(tt + 8);
      table3.store(tt + 12);
      index.store(ii);
      for (int j = 0; j < 4; j++) {
        rr[j] = tt[ii[j]];
      }
      return i32x4().load(rr);
#endif
}

template<std::size_t n, AnyInt32x4 TIdx, AnyInt32 TVal>
inline NativeVector<TVal, 4> lookup_bounded(const TIdx index, const TVal* table) {
  if constexpr (n == 0) {
    return 0;
  }
  if constexpr (n <= 4) {
    return lookup4(index, TIdx{}.load(table));
  }
  if constexpr (n <= 8) {
    return lookup8(index, TIdx{}.load(table), TIdx{}.load(table + 4));
  }
  // n > 8
  lookup(index, table);
}

inline i64x2 lookup(const i64x2 index, const i64* table) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm_i64gather_epi64(reinterpret_cast<const long long*>(table), index, 8);
#else
  i64 ii[2];
  index.store(ii);
  return i64x2(table[ii[0]], table[ii[1]]);
#endif
}

inline i64x2 lookup2(const i64x2 index, const i64x2 table) {
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  return _mm_shuffle_epi8(table, index * i64x2(0x0808080808080808) + 0x0706050403020100);
#else
  i64 ii[2], tt[2];
  table.store(tt);
  index.store(ii);
  return i64x2(tt[int(ii[0])], tt[int(ii[1])]);
#endif
}

template<std::size_t n>
inline i64x2 lookup_bounded(const i64x2 index, const i64* table) {
  if constexpr (n == 0) {
    return 0;
  }
  // n > 0
  lookup(index, table);
}

/*****************************************************************************
 *
 *          Byte shifts
 *
 *****************************************************************************/

// Function shift_bytes_up: shift whole vector left by b bytes.
template<unsigned int b>
inline i8x16 shift_bytes_up(const i8x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  if (b < 16) {
    return _mm_alignr_epi8(a, _mm_setzero_si128(), 16 - b);
  }
  return _mm_setzero_si128(); // zero
#else
  i8 dat[32];
  if (b < 16) {
    i8x16(0).store(dat);
    a.store(dat + b);
    return i8x16().load(dat);
  }
  return 0;
#endif
}

// Function shift_bytes_down: shift whole vector right by b bytes
template<unsigned int b>
inline i8x16 shift_bytes_down(const i8x16 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
  if (b < 16) {
    return _mm_alignr_epi8(_mm_setzero_si128(), a, b);
  }
  return _mm_setzero_si128();
#else
  i8 dat[32];
  if (b < 16) {
    a.store(dat);
    i8x16(0).store(dat + 16);
    return i8x16().load(dat + b);
  }
  return 0;
#endif
}

/*****************************************************************************
 *
 *          Gather functions with fixed indexes
 *
 *****************************************************************************/

// Load elements from array a with indices i0, i1, i2, i3
template<int i0, int i1, int i2, int i3>
inline i32x4 gather4i(const void* a) {
  constexpr std::array<int, 4> indexs{i0, i1, i2, i3}; // indexes as array
  constexpr int imin = min_index(indexs);
  constexpr int imax = max_index(indexs);
  static_assert(imin >= 0, "Negative index in gather function");

  if constexpr (imax - imin <= 3) {
    // load one contiguous block and permute
    if constexpr (imax > 3) {
      // make sure we don't read past the end of the array
      i32x4 b = i32x4().load(reinterpret_cast<const i32*>(a) + imax - 3);
      return permute4<i0 - imax + 3, i1 - imax + 3, i2 - imax + 3, i3 - imax + 3>(b);
    } else {
      i32x4 b = i32x4().load(reinterpret_cast<const i32*>(a) + imin);
      return permute4<i0 - imin, i1 - imin, i2 - imin, i3 - imin>(b);
    }
  }
  if constexpr ((i0 < imin + 4 || i0 > imax - 4) && (i1 < imin + 4 || i1 > imax - 4) &&
                (i2 < imin + 4 || i2 > imax - 4) && (i3 < imin + 4 || i3 > imax - 4)) {
    // load two contiguous blocks and blend
    i32x4 b = i32x4().load(reinterpret_cast<const i32*>(a) + imin);
    i32x4 c = i32x4().load(reinterpret_cast<const i32*>(a) + imax - 3);
    constexpr int j0 = i0 < imin + 4 ? i0 - imin : 7 - imax + i0;
    constexpr int j1 = i1 < imin + 4 ? i1 - imin : 7 - imax + i1;
    constexpr int j2 = i2 < imin + 4 ? i2 - imin : 7 - imax + i2;
    constexpr int j3 = i3 < imin + 4 ? i3 - imin : 7 - imax + i3;
    return blend4<j0, j1, j2, j3>(b, c);
  }
  // use AVX2 gather if available
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  return _mm_i32gather_epi32(reinterpret_cast<const int*>(a), (i32x4(i0, i1, i2, i3)), 4);
#else
  return lookup_bounded<imax + 1>(i32x4(i0, i1, i2, i3), a);
#endif
}

// Load elements from array a with indices i0, i1
template<int i0, int i1>
inline i64x2 gather2q(const void* a) {
  constexpr int imin = i0 < i1 ? i0 : i1;
  constexpr int imax = i0 > i1 ? i0 : i1;
  static_assert(imin >= 0, "Negative index in gather function");

  if constexpr (imax - imin <= 1) {
    // load one contiguous block and permute
    if constexpr (imax > 1) {
      // make sure we don't read past the end of the array
      i64x2 b = i64x2().load(reinterpret_cast<const i64*>(a) + imax - 1);
      return permute2<i0 - imax + 1, i1 - imax + 1>(b);
    } else {
      i64x2 b = i64x2().load(reinterpret_cast<const i64*>(a) + imin);
      return permute2<i0 - imin, i1 - imin>(b);
    }
  }
  return i64x2(reinterpret_cast<const i64*>(a)[i0], reinterpret_cast<const i64*>(a)[i1]);
}

/*****************************************************************************
 *
 *          NativeVector scatter functions with fixed indexes
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
 * The scatter functions are useful if the data are distributed in a sparce
 * manner into the array. If the array is dense then it is more efficient
 * to permute the data into the right positions and then write the whole
 * permuted vector into the array.
 *
 * Example:
 * i64x8 a(10,11,12,13,14,15,16,17);
 * i64 b[16] = {0};
 * scatter<0,2,14,10,1,-1,5,9>(a,b); // b = (10,14,11,0,0,16,0,0,0,17,13,0,0,0,12,0)
 *
 *****************************************************************************/

template<int i0, int i1, int i2, int i3>
inline void scatter(const i32x4 data, void* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
  __m128i indx = _mm_setr_epi32(i0, i1, i2, i3);
  __mmask8 mask = u8((i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3));
  _mm_mask_i32scatter_epi32(destination, mask, indx, data, 4);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  __m512i indx = _mm512_castsi128_si512(_mm_setr_epi32(i0, i1, i2, i3));
  __mmask16 mask = u16((i0 >= 0) | ((i1 >= 0) << 1) | ((i2 >= 0) << 2) | ((i3 >= 0) << 3));
  _mm512_mask_i32scatter_epi32(destination, mask, indx, _mm512_castsi128_si512(data), 4);
#else
      i32* arr = reinterpret_cast<i32*>(destination);
      const int index[4] = {i0, i1, i2, i3};
      for (std::size_t i = 0; i < 4; ++i) {
        if (index[i] >= 0) {
          arr[index[i]] = data[i];
        }
      }
#endif
}

template<int i0, int i1>
inline void scatter(const i64x2 data, void* destination) {
  auto* arr = reinterpret_cast<i64*>(destination);
  if (i0 >= 0) {
    arr[i0] = data[0];
  }
  if (i1 >= 0) {
    arr[i1] = data[1];
  }
}

/*****************************************************************************
 *
 *          Scatter functions with variable indexes
 *
 *****************************************************************************/

inline void scatter(const i32x4 index, u32 limit, const i32x4 data, void* destination) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL //  AVX512VL
  __mmask8 mask = _mm_cmplt_epu32_mask(index, (u32x4(limit)));
  _mm_mask_i32scatter_epi32(destination, mask, index, data, 4);
#elif STADO_INSTRUCTION_SET >= STADO_AVX512F
  __mmask16 mask = _mm512_mask_cmplt_epu32_mask(0xF, _mm512_castsi128_si512(index),
                                                _mm512_castsi128_si512(u32x4(limit)));
  _mm512_mask_i32scatter_epi32((int*)destination, mask, _mm512_castsi128_si512(index),
                               _mm512_castsi128_si512(data), 4);
#else
      i32* arr = reinterpret_cast<i32*>(destination);
      for (std::size_t i = 0; i < 4; ++i) {
        if (u32(index[i]) < limit) {
          arr[index[i]] = data[i];
        }
      }
#endif
}

inline void scatter(const i64x2 index, u32 limit, const i64x2 data, void* destination) {
  auto* arr = reinterpret_cast<i64*>(destination);
  if (u64(index[0]) < u64(limit)) {
    arr[index[0]] = data[0];
  }
  if (u64(index[1]) < u64(limit)) {
    arr[index[1]] = data[1];
  }
}

/*****************************************************************************
 *
 *          Functions for conversion between integer sizes
 *
 *****************************************************************************/

// Extend 8-bit integers to 16-bit integers, signed and unsigned

// Function extend_low : extends the low 8 elements to 16 bits with sign extension
inline i16x8 extend_low(const i8x16 a) {
  __m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(), a); // 0 > a
  return _mm_unpacklo_epi8(a, sign); // interleave with sign extensions
}

// Function extend_high : extends the high 8 elements to 16 bits with sign extension
inline i16x8 extend_high(const i8x16 a) {
  __m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(), a); // 0 > a
  return _mm_unpackhi_epi8(a, sign); // interleave with sign extensions
}

// Function extend_low : extends the low 8 elements to 16 bits with zero extension
inline u16x8 extend_low(const u8x16 a) {
  return _mm_unpacklo_epi8(a, _mm_setzero_si128()); // interleave with zero extensions
}

// Function extend_high : extends the high 8 elements to 16 bits with zero extension
inline u16x8 extend_high(const u8x16 a) {
  return _mm_unpackhi_epi8(a, _mm_setzero_si128()); // interleave with zero extensions
}

// Extend 16-bit integers to 32-bit integers, signed and unsigned

// Function extend_low : extends the low 4 elements to 32 bits with sign extension
inline i32x4 extend_low(const i16x8 a) {
  __m128i sign = _mm_srai_epi16(a, 15); // sign bit
  return _mm_unpacklo_epi16(a, sign); // interleave with sign extensions
}

// Function extend_high : extends the high 4 elements to 32 bits with sign extension
inline i32x4 extend_high(const i16x8 a) {
  __m128i sign = _mm_srai_epi16(a, 15); // sign bit
  return _mm_unpackhi_epi16(a, sign); // interleave with sign extensions
}

// Function extend_low : extends the low 4 elements to 32 bits with zero extension
inline u32x4 extend_low(const u16x8 a) {
  return _mm_unpacklo_epi16(a, _mm_setzero_si128()); // interleave with zero extensions
}

// Function extend_high : extends the high 4 elements to 32 bits with zero extension
inline u32x4 extend_high(const u16x8 a) {
  return _mm_unpackhi_epi16(a, _mm_setzero_si128()); // interleave with zero extensions
}

// Extend 32-bit integers to 64-bit integers, signed and unsigned

// Function extend_low : extends the low 2 elements to 64 bits with sign extension
inline i64x2 extend_low(const i32x4 a) {
  __m128i sign = _mm_srai_epi32(a, 31); // sign bit
  return _mm_unpacklo_epi32(a, sign); // interleave with sign extensions
}

// Function extend_high : extends the high 2 elements to 64 bits with sign extension
inline i64x2 extend_high(const i32x4 a) {
  __m128i sign = _mm_srai_epi32(a, 31); // sign bit
  return _mm_unpackhi_epi32(a, sign); // interleave with sign extensions
}

// Function extend_low : extends the low 2 elements to 64 bits with zero extension
inline u64x2 extend_low(const u32x4 a) {
  return _mm_unpacklo_epi32(a, _mm_setzero_si128()); // interleave with zero extensions
}

// Function extend_high : extends the high 2 elements to 64 bits with zero extension
inline u64x2 extend_high(const u32x4 a) {
  return _mm_unpackhi_epi32(a, _mm_setzero_si128()); // interleave with zero extensions
}

// Compress 16-bit integers to 8-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Overflow wraps around
inline i8x16 compress(const i16x8 low, const i16x8 high) {
  __m128i mask = _mm_set1_epi32(0x00FF00FF); // mask for low bytes
  __m128i lowm = _mm_and_si128(low, mask); // bytes of low
  __m128i highm = _mm_and_si128(high, mask); // bytes of high
  return _mm_packus_epi16(lowm, highm); // unsigned pack
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Signed, with saturation
inline i8x16 compress_saturated(const i16x8 low, const i16x8 high) {
  return _mm_packs_epi16(low, high);
}

// Function compress : packs two vectors of 16-bit integers to one vector of 8-bit integers
// Unsigned, overflow wraps around
inline u8x16 compress(const u16x8 low, const u16x8 high) {
  return u8x16(compress((i16x8)low, (i16x8)high));
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Unsigned, with saturation
inline u8x16 compress_saturated(const u16x8 low, const u16x8 high) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  __m128i maxval = _mm_set1_epi32(0x00FF00FF); // maximum value
  __m128i low1 = _mm_min_epu16(low, maxval); // upper limit
  __m128i high1 = _mm_min_epu16(high, maxval); // upper limit
  return _mm_packus_epi16(
    low1, high1); // this instruction saturates from signed 32 bit to unsigned 16 bit
#else
  __m128i zero = _mm_setzero_si128();
  __m128i signlow = _mm_cmpgt_epi16(zero, low); // sign bit of low
  __m128i signhi = _mm_cmpgt_epi16(zero, high); // sign bit of high
  __m128i slow2 = _mm_srli_epi16(signlow, 8); // FF if low negative
  __m128i shigh2 = _mm_srli_epi16(signhi, 8); // FF if high negative
  __m128i maskns = _mm_set1_epi32(0x7FFF7FFF); // mask for removing sign bit
  __m128i lowns = _mm_and_si128(low, maskns); // low,  with sign bit removed
  __m128i highns = _mm_and_si128(high, maskns); // high, with sign bit removed
  __m128i lowo = _mm_or_si128(lowns, slow2); // low,  sign bit replaced by 00FF
  __m128i higho = _mm_or_si128(highns, shigh2); // high, sign bit replaced by 00FF
  return _mm_packus_epi16(lowo,
                          higho); // this instruction saturates from signed 16 bit to unsigned 8 bit
#endif
}

// Compress 32-bit integers to 16-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
inline i16x8 compress(const i32x4 low, const i32x4 high) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  __m128i mask = _mm_set1_epi32(0x0000FFFF); // mask for low words
  __m128i lowm = _mm_and_si128(low, mask); // bytes of low
  __m128i highm = _mm_and_si128(high, mask); // bytes of high
  return _mm_packus_epi32(lowm, highm); // unsigned pack
#else
  __m128i low1 = _mm_shufflelo_epi16(low, 0xD8); // low words in place
  __m128i high1 = _mm_shufflelo_epi16(high, 0xD8); // low words in place
  __m128i low2 = _mm_shufflehi_epi16(low1, 0xD8); // low words in place
  __m128i high2 = _mm_shufflehi_epi16(high1, 0xD8); // low words in place
  __m128i low3 = _mm_shuffle_epi32(low2, 0xD8); // low dwords of low  to pos. 0 and 32
  __m128i high3 = _mm_shuffle_epi32(high2, 0xD8); // low dwords of high to pos. 0 and 32
  return _mm_unpacklo_epi64(low3, high3); // interleave
#endif
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Signed with saturation
inline i16x8 compress_saturated(const i32x4 low, const i32x4 high) {
  return _mm_packs_epi32(low, high); // pack with signed saturation
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
inline u16x8 compress(const u32x4 low, const u32x4 high) {
  return u16x8(compress((i32x4)low, (i32x4)high));
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Unsigned, with saturation
inline u16x8 compress_saturated(const u32x4 low, const u32x4 high) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  __m128i maxval = _mm_set1_epi32(0x0000FFFF); // maximum value
  __m128i low1 = _mm_min_epu32(low, maxval); // upper limit
  __m128i high1 = _mm_min_epu32(high, maxval); // upper limit
  return _mm_packus_epi32(
    low1, high1); // this instruction saturates from signed 32 bit to unsigned 16 bit
#else
  __m128i zero = _mm_setzero_si128();
  __m128i lowzero = _mm_cmpeq_epi16(low, zero); // for each word is zero
  __m128i highzero = _mm_cmpeq_epi16(high, zero); // for each word is zero
  __m128i mone = _mm_set1_epi32(-1); // FFFFFFFF
  __m128i lownz = _mm_xor_si128(lowzero, mone); // for each word is nonzero
  __m128i highnz = _mm_xor_si128(highzero, mone); // for each word is nonzero
  __m128i lownz2 = _mm_srli_epi32(lownz, 16); // shift down to low dword
  __m128i highnz2 = _mm_srli_epi32(highnz, 16); // shift down to low dword
  __m128i lowsatur = _mm_or_si128(low, lownz2); // low, saturated
  __m128i hisatur = _mm_or_si128(high, highnz2); // high, saturated
  return u16x8(compress(i32x4(lowsatur), i32x4(hisatur)));
#endif
}

// Compress 64-bit integers to 32-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Overflow wraps around
inline i32x4 compress(const i64x2 low, const i64x2 high) {
  __m128i low2 = _mm_shuffle_epi32(low, 0xD8); // low dwords of low  to pos. 0 and 32
  __m128i high2 = _mm_shuffle_epi32(high, 0xD8); // low dwords of high to pos. 0 and 32
  return _mm_unpacklo_epi64(low2, high2); // interleave
}

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Signed, with saturation
// This function is very inefficient unless the SSE4.2 instruction set is supported
inline i32x4 compress_saturated(const i64x2 low, const i64x2 high) {
  i64x2 maxval = _mm_set_epi32(0, 0x7FFFFFFF, 0, 0x7FFFFFFF);
  i64x2 minval = _mm_set_epi32(-1, i32(0x80000000), -1, i32(0x80000000));
  i64x2 low1 = min(low, maxval);
  i64x2 high1 = min(high, maxval);
  i64x2 low2 = max(low1, minval);
  i64x2 high2 = max(high1, minval);
  return compress(low2, high2);
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
inline u32x4 compress(const u64x2 low, const u64x2 high) {
  return {compress((i64x2)low, (i64x2)high)};
}

// Function compress : packs two vectors of 64-bit integers into one vector of 32-bit integers
// Unsigned, with saturation
inline u32x4 compress_saturated(const u64x2 low, const u64x2 high) {
  __m128i zero = _mm_setzero_si128();
  __m128i lowzero = _mm_cmpeq_epi32(low, zero); // for each dword is zero
  __m128i highzero = _mm_cmpeq_epi32(high, zero); // for each dword is zero
  __m128i mone = _mm_set1_epi32(-1); // FFFFFFFF
  __m128i lownz = _mm_xor_si128(lowzero, mone); // for each dword is nonzero
  __m128i highnz = _mm_xor_si128(highzero, mone); // for each dword is nonzero
  __m128i lownz2 = _mm_srli_epi64(lownz, 32); // shift down to low dword
  __m128i highnz2 = _mm_srli_epi64(highnz, 32); // shift down to low dword
  __m128i lowsatur = _mm_or_si128(low, lownz2); // low, saturated
  __m128i hisatur = _mm_or_si128(high, highnz2); // high, saturated
  return {compress(i64x2(lowsatur), i64x2(hisatur))};
}

/*****************************************************************************
 *
 *          Integer division operators
 *
 ******************************************************************************
 *
 * The instruction set does not support integer vector division. Instead, we
 * are using a method for fast integer division based on multiplication and
 * shift operations. This method is faster than simple integer division if the
 * same divisor is used multiple times.
 *
 * All elements in a vector are divided by the same divisor. It is not possible
 * to divide different elements of the same vector by different divisors.
 *
 * The parameters used for fast division are stored in an object of a
 * Divisor class. This object can be created implicitly, for example in:
 *        i32x4 a, b; int c;
 *        a = b / c;
 * or explicitly as:
 *        a = b / DivisorI32(c);
 *
 * It takes more time to compute the parameters used for fast division than to
 * do the division. Therefore, it is advantageous to use the same divisor object
 * multiple times. For example, to divide 80 unsigned short integers by 10:
 *
 *        u16 dividends[80], quotients[80];         // numbers to work with
 *        DivisorU16 div10(10);                          // make divisor object for dividing by 10
 *        u16x8 temp;                                   // temporary vector
 *        for (std::size_t i = 0; i < 80; i += 8) {              // loop for 4 elements per
 *iteration temp.load(dividends+i);                    // load 4 elements temp /= div10; // divide
 *each element by 10 temp.store(quotients+i);                   // store 4 elements
 *        }
 *
 * The parameters for fast division can also be computed at compile time. This is
 * an advantage if the divisor is known at compile time. Use the const_int or const_uint
 * macro to do this. For example, for signed integers:
 *        i16x8 a, b;
 *        a = b / const_int(10);
 * Or, for unsigned integers:
 *        u16x8 a, b;
 *        a = b / const_uint(10);
 *
 * The division of a vector of 16-bit integers is faster than division of a vector
 * of other integer sizes.
 *
 *
 * Mathematical formula, used for signed division with fixed or variable divisor:
 * (From T. Granlund and P. L. Montgomery: Division by Invariant Integers Using Multiplication,
 * Proceedings of the SIGPLAN 1994 Conference on Programming Language Design and Implementation.
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.2556 )
 * x = dividend
 * d = abs(divisor)
 * w = integer word size, bits
 * L = ceil(log2(d)) = bit_scan_reverse(d-1)+1
 * L = max(L,1)
 * m = 1 + 2^(w+L-1)/d - 2^w                      [division should overflow to 0 if d = 1]
 * sh1 = L-1
 * q = x + (m*x >> w)                             [high part of signed multiplication with 2w bits]
 * q = (q >> sh1) - (x<0 ? -1 : 0)
 * if (divisor < 0) q = -q
 * result = trunc(x/d) = q
 *
 * Mathematical formula, used for unsigned division with variable divisor:
 * (Also from T. Granlund and P. L. Montgomery)
 * x = dividend
 * d = divisor
 * w = integer word size, bits
 * L = ceil(log2(d)) = bit_scan_reverse(d-1)+1
 * m = 1 + 2^w * (2^L-d) / d                      [2^L should overflow to 0 if L = w]
 * sh1 = min(L,1)
 * sh2 = max(L-1,0)
 * t = m*x >> w                                   [high part of unsigned multiplication with 2w
 *bits] result = floor(x/d) = (((x-t) >> sh1) + t) >> sh2
 *
 * Mathematical formula, used for unsigned division with fixed divisor:
 * (From Terje Mathisen, unpublished)
 * x = dividend
 * d = divisor
 * w = integer word size, bits
 * b = floor(log2(d)) = bit_scan_reverse(d)
 * f = 2^(w+b) / d                                [exact division]
 * If f is an integer then d is a power of 2 then go to case A
 * If the fractional part of f is < 0.5 then go to case B
 * If the fractional part of f is > 0.5 then go to case C
 * Case A:  [shift only]
 * result = x >> b
 * Case B:  [round down f and compensate by adding one to x]
 * result = ((x+1)*floor(f)) >> (w+b)             [high part of unsigned multiplication with 2w
 *bits] Case C:  [round up f, no compensation for rounding error] result = (x*ceil(f)) >> (w+b)
 *[high part of unsigned multiplication with 2w bits]
 *
 *
 *****************************************************************************/

// vector operator / : divide each element by divisor

// vector of 4 32-bit signed integers
inline i32x4 operator/(const i32x4 a, const DivisorI32 d) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  __m128i t1 = _mm_mul_epi32(a, d.getm()); // 32x32->64 bit signed multiplication of a[0] and a[2]
  __m128i t2 = _mm_srli_epi64(t1, 32); // high dword of result 0 and 2
  __m128i t3 = _mm_srli_epi64(a, 32); // get a[1] and a[3] into position for multiplication
  __m128i t4 = _mm_mul_epi32(t3, d.getm()); // 32x32->64 bit signed multiplication of a[1] and a[3]
  __m128i t7 = _mm_blend_epi16(t2, t4, 0xCC);
  __m128i t8 = _mm_add_epi32(t7, a); // add
  __m128i t9 = _mm_sra_epi32(t8, d.gets1()); // shift right arithmetic
  __m128i t10 = _mm_srai_epi32(a, 31); // sign of a
  __m128i t11 = _mm_sub_epi32(t10, d.getsign()); // sign of a - sign of d
  __m128i t12 = _mm_sub_epi32(t9, t11); // + 1 if a < 0, -1 if d < 0
  return _mm_xor_si128(t12, d.getsign()); // change sign if divisor negative
#else // not SSE4.1
  __m128i t1 = _mm_mul_epu32(a, d.getm()); // 32x32->64 bit unsigned multiplication of a[0] and a[2]
  __m128i t2 = _mm_srli_epi64(t1, 32); // high dword of result 0 and 2
  __m128i t3 = _mm_srli_epi64(a, 32); // get a[1] and a[3] into position for multiplication
  __m128i t4 =
    _mm_mul_epu32(t3, d.getm()); // 32x32->64 bit unsigned multiplication of a[1] and a[3]
  __m128i t5 = _mm_set_epi32(-1, 0, -1, 0); // mask of dword 1 and 3
  __m128i t6 = _mm_and_si128(t4, t5); // high dword of result 1 and 3
  __m128i t7 =
    _mm_or_si128(t2, t6); // combine all four results of unsigned high mul into one vector
  // convert unsigned to signed high multiplication (from: H S Warren: Hacker's delight, 2003, p.
  // 132)
  __m128i u1 = _mm_srai_epi32(a, 31); // sign of a
  __m128i u2 =
    _mm_srai_epi32(d.getm(), 31); // sign of m [ m is always negative, except for abs(d) = 1 ]
  __m128i u3 = _mm_and_si128(d.getm(), u1); // m * sign of a
  __m128i u4 = _mm_and_si128(a, u2); // a * sign of m
  __m128i u5 = _mm_add_epi32(u3, u4); // sum of sign corrections
  __m128i u6 = _mm_sub_epi32(t7, u5); // high multiplication result converted to signed
  __m128i t8 = _mm_add_epi32(u6, a); // add a
  __m128i t9 = _mm_sra_epi32(t8, d.gets1()); // shift right arithmetic
  __m128i t10 = _mm_sub_epi32(u1, d.getsign()); // sign of a - sign of d
  __m128i t11 = _mm_sub_epi32(t9, t10); // + 1 if a < 0, -1 if d < 0
  return _mm_xor_si128(t11, d.getsign()); // change sign if divisor negative
#endif
}

// vector of 4 32-bit unsigned integers
inline u32x4 operator/(const u32x4 a, const DivisorU32 d) {
  __m128i t1 = _mm_mul_epu32(a, d.getm()); // 32x32->64 bit unsigned multiplication of a[0] and a[2]
  __m128i t2 = _mm_srli_epi64(t1, 32); // high dword of result 0 and 2
  __m128i t3 = _mm_srli_epi64(a, 32); // get a[1] and a[3] into position for multiplication
  __m128i t4 =
    _mm_mul_epu32(t3, d.getm()); // 32x32->64 bit unsigned multiplication of a[1] and a[3]
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  __m128i t7 = _mm_blend_epi16(t2, t4, 0xCC); // blend two results
#else
  __m128i t5 = _mm_set_epi32(-1, 0, -1, 0); // mask of dword 1 and 3
  __m128i t6 = _mm_and_si128(t4, t5); // high dword of result 1 and 3
  __m128i t7 = _mm_or_si128(t2, t6); // combine all four results into one vector
#endif
  __m128i t8 = _mm_sub_epi32(a, t7); // subtract
  __m128i t9 = _mm_srl_epi32(t8, d.gets1()); // shift right logical
  __m128i t10 = _mm_add_epi32(t7, t9); // add
  return _mm_srl_epi32(t10, d.gets2()); // shift right logical
}

// vector of 8 16-bit signed integers
inline i16x8 operator/(const i16x8 a, const DivisorI16 d) {
  __m128i t1 = _mm_mulhi_epi16(a, d.getm()); // multiply high signed words
  __m128i t2 = _mm_add_epi16(t1, a); // + a
  __m128i t3 = _mm_sra_epi16(t2, d.gets1()); // shift right arithmetic
  __m128i t4 = _mm_srai_epi16(a, 15); // sign of a
  __m128i t5 = _mm_sub_epi16(t4, d.getsign()); // sign of a - sign of d
  __m128i t6 = _mm_sub_epi16(t3, t5); // + 1 if a < 0, -1 if d < 0
  return _mm_xor_si128(t6, d.getsign()); // change sign if divisor negative
}

// vector of 8 16-bit unsigned integers
inline u16x8 operator/(const u16x8 a, const DivisorU16 d) {
  __m128i t1 = _mm_mulhi_epu16(a, d.getm()); // multiply high unsigned words
  __m128i t2 = _mm_sub_epi16(a, t1); // subtract
  __m128i t3 = _mm_srl_epi16(t2, d.gets1()); // shift right logical
  __m128i t4 = _mm_add_epi16(t1, t3); // add
  return _mm_srl_epi16(t4, d.gets2()); // shift right logical
}

// vector of 16 8-bit signed integers
inline i8x16 operator/(const i8x16 a, const DivisorI16 d) {
  // expand into two i16x8
  i16x8 low = extend_low(a) / d;
  i16x8 high = extend_high(a) / d;
  return compress(low, high);
}

// vector of 16 8-bit unsigned integers
inline u8x16 operator/(const u8x16 a, const DivisorU16 d) {
  // expand into two i16x8
  u16x8 low = extend_low(a) / d;
  u16x8 high = extend_high(a) / d;
  return compress(low, high);
}

// vector operator /= : divide
inline i16x8& operator/=(i16x8& a, const DivisorI16 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline u16x8& operator/=(u16x8& a, const DivisorU16 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline i32x4& operator/=(i32x4& a, const DivisorI32 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline u32x4& operator/=(u32x4& a, const DivisorU32 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline i8x16& operator/=(i8x16& a, const DivisorI16 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline u8x16& operator/=(u8x16& a, const DivisorU16 d) {
  a = a / d;
  return a;
}

/*****************************************************************************
 *
 *          Integer division 2: divisor is a compile-time constant
 *
 *****************************************************************************/

// Divide i32x4 by compile-time constant
template<i32 d>
inline i32x4 divide_by_i(const i32x4 x) {
  static_assert(d != 0, "Integer division by zero"); // Error message if dividing by zero
  if constexpr (d == 1) {
    return x;
  }
  if constexpr (d == -1) {
    return -x;
  }
  if constexpr (u32(d) == 0x80000000U) {
    // prevent overflow when changing sign
    return {i32x4(x == i32x4(i32(0x80000000))) & i32x4(1)};
  }
  constexpr u32 d1 =
    d > 0 ? u32(d)
          : u32(-d); // compile-time abs(d). (force GCC compiler to treat d as 32 bits, not 64 bits)
  if constexpr ((d1 & (d1 - 1)) == 0) {
    // d1 is a power of 2. use shift
    constexpr int k = bit_scan_reverse_const(d1);
    __m128i sign;
    if constexpr (k > 1) {
      sign = _mm_srai_epi32(x, k - 1);
    } else {
      sign = x; // k copies of sign bit
    }
    __m128i bias = _mm_srli_epi32(sign, 32 - k); // bias = x >= 0 ? 0 : k-1
    __m128i xpbias = _mm_add_epi32(x, bias); // x + bias
    __m128i q = _mm_srai_epi32(xpbias, k); // (x + bias) >> k
    if (d > 0) {
      return q; // d > 0: return  q
    }
    return _mm_sub_epi32(_mm_setzero_si128(), q); // d < 0: return -q
  }
  // general case
  constexpr i32 sh =
    bit_scan_reverse_const(u32(d1) - 1); // ceil(log2(d1)) - 1. (d1 < 2 handled by power of 2 case)
  constexpr i32 mult = int(1 + (u64(1) << (32 + sh)) / u32(d1) - (i64(1) << 32)); // multiplier
  const DivisorI32 div(mult, sh, d < 0 ? -1 : 0);
  return x / div;
}

// define i32x4 a / const_int(d)
template<i32 d>
inline i32x4 operator/(const i32x4 a, ConstInt<d> /*b*/) {
  return divide_by_i<d>(a);
}

// define i32x4 a / const_uint(d)
template<u32 d>
inline i32x4 operator/(const i32x4 a, ConstUint<d> /*b*/) {
  static_assert(d < 0x80000000U, "Dividing signed by overflowing unsigned");
  return divide_by_i<i32(d)>(a); // signed divide
}

// vector operator /= : divide
template<i32 d>
inline i32x4& operator/=(i32x4& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<u32 d>
inline i32x4& operator/=(i32x4& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// Divide u32x4 by compile-time constant
template<u32 d>
inline u32x4 divide_by_ui(const u32x4 x) {
  static_assert(d != 0, "Integer division by zero"); // Error message if dividing by zero
  if constexpr (d == 1) {
    return x; // divide by 1
  }
  constexpr int b = bit_scan_reverse_const(d); // floor(log2(d))
  if constexpr ((d & (d - 1)) == 0) {
    // d is a power of 2. use shift
    return _mm_srli_epi32(x, b); // x >> b
  }
  // general case (d > 2)
  constexpr u32 mult = u32((u64(1) << (b + 32)) / d); // multiplier = 2^(32+b) / d
  constexpr u64 rem = (u64(1) << (b + 32)) - u64(d) * mult; // remainder 2^(32+b) % d
  constexpr bool round_down = (2 * rem < d); // check if fraction is less than 0.5
  constexpr u32 mult1 = round_down ? mult : mult + 1;
  // do 32*32->64 bit unsigned multiplication and get high part of result
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
  const __m128i multv = _mm_maskz_set1_epi32(0x05, mult1); // zero-extend mult and broadcast
#else
  const __m128i multv = u64x2(u64(mult1)); // zero-extend mult and broadcast
#endif
  __m128i t1 = _mm_mul_epu32(x, multv); // 32x32->64 bit unsigned multiplication of x[0] and x[2]
  if constexpr (round_down) {
    t1 = _mm_add_epi64(
      t1, multv); // compensate for rounding error. (x+1)*m replaced by x*m+m to avoid overflow
  }
  __m128i t2 = _mm_srli_epi64(t1, 32); // high dword of result 0 and 2
  __m128i t3 = _mm_srli_epi64(x, 32); // get x[1] and x[3] into position for multiplication
  __m128i t4 = _mm_mul_epu32(t3, multv); // 32x32->64 bit unsigned multiplication of x[1] and x[3]
  if constexpr (round_down) {
    t4 = _mm_add_epi64(
      t4, multv); // compensate for rounding error. (x+1)*m replaced by x*m+m to avoid overflow
  }
#if STADO_INSTRUCTION_SET >= STADO_SSE4_1
  __m128i t7 = _mm_blend_epi16(t2, t4, 0xCC); // blend two results
#else
  __m128i t5 = _mm_set_epi32(-1, 0, -1, 0); // mask of dword 1 and 3
  __m128i t6 = _mm_and_si128(t4, t5); // high dword of result 1 and 3
  __m128i t7 = _mm_or_si128(t2, t6); // combine all four results into one vector
#endif
  u32x4 q = _mm_srli_epi32(t7, b); // shift right by b
  return q; // no overflow possible
}

// define u32x4 a / const_uint(d)
template<u32 d>
inline u32x4 operator/(const u32x4 a, ConstUint<d> /*b*/) {
  return divide_by_ui<d>(a);
}

// define u32x4 a / const_int(d)
template<i32 d>
inline u32x4 operator/(const u32x4 a, ConstInt<d> /*b*/) {
  static_assert(d < 0x80000000U, "Dividing unsigned integer by negative value is ambiguous");
  return divide_by_ui<d>(a); // unsigned divide
}

// vector operator /= : divide
template<u32 d>
inline u32x4& operator/=(u32x4& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<i32 d>
inline u32x4& operator/=(u32x4& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// Divide i16x8 by compile-time constant
template<int d>
inline i16x8 divide_by_i(const i16x8 x) {
  constexpr i16 d0 = i16(d); // truncate d to 16 bits
  static_assert(d0 != 0, "Integer division by zero"); // Error message if dividing by zero
  if constexpr (d0 == 1) {
    return x; // divide by  1
  }
  if constexpr (d0 == -1) {
    return -x; // divide by -1
  }
  if constexpr (u16(d0) == 0x8000U) {
    return {i16x8(x == i16x8(i16(0x8000))) & i16x8(1)}; // prevent overflow when changing sign
  }
  // if (d > 0x7FFF || d < -0x8000) return 0;            // not relevant when d truncated to 16 bits
  const u32 d1 =
    d > 0 ? u32(d)
          : u32(-d); // compile-time abs(d). (force GCC compiler to treat d as 32 bits, not 64 bits)
  if constexpr ((d1 & (d1 - 1)) == 0) {
    // d is a power of 2. use shift
    constexpr int k = bit_scan_reverse_const(u32(d1));
    __m128i sign;
    if constexpr (k > 1) {
      sign = _mm_srai_epi16(x, k - 1);
    } else {
      sign = x; // k copies of sign bit
    }
    __m128i bias = _mm_srli_epi16(sign, 16 - k); // bias = x >= 0 ? 0 : k-1
    __m128i xpbias = _mm_add_epi16(x, bias); // x + bias
    __m128i q = _mm_srai_epi16(xpbias, k); // (x + bias) >> k
    if (d0 > 0) {
      return q; // d0 > 0: return  q
    }
    return _mm_sub_epi16(_mm_setzero_si128(), q); // d0 < 0: return -q
  }
  // general case
  const int arr = bit_scan_reverse_const(u16(d1 - 1)) + 1; // ceil(log2(d)). (d < 2 handled above)
  const i16 mult = i16(1 + (1U << (15 + arr)) / u32(d1) - 0x10000); // multiplier
  const int shift1 = arr - 1;
  const DivisorI16 div(mult, shift1, d0 > 0 ? 0 : -1);
  return x / div;
}

// define i16x8 a / const_int(d)
template<int d>
inline i16x8 operator/(const i16x8 a, ConstInt<d> /*b*/) {
  return divide_by_i<d>(a);
}

// define i16x8 a / const_uint(d)
template<u32 d>
inline i16x8 operator/(const i16x8 a, ConstUint<d> /*b*/) {
  static_assert(d < 0x8000U, "Dividing signed by overflowing unsigned");
  return divide_by_i<int(d)>(a); // signed divide
}

// vector operator /= : divide
template<i32 d>
inline i16x8& operator/=(i16x8& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<u32 d>
inline i16x8& operator/=(i16x8& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// Divide u16x8 by compile-time constant
template<u32 d>
inline u16x8 divide_by_ui(const u16x8 x) {
  constexpr u16 d0 = u16(d); // truncate d to 16 bits
  static_assert(d0 != 0, "Integer division by zero"); // Error message if dividing by zero
  if constexpr (d0 == 1) {
    return x; // divide by 1
  }
  constexpr int b = bit_scan_reverse_const(d0); // floor(log2(d))
  if constexpr ((d0 & (d0 - 1U)) == 0) {
    // d is a power of 2. use shift
    return _mm_srli_epi16(x, b); // x >> b
  }
  // general case (d > 2)
  constexpr u16 mult = u16((1U << u32(b + 16)) / d0); // multiplier = 2^(32+b) / d
  constexpr u32 rem = (u32(1) << u32(b + 16)) - u32(d0) * mult; // remainder 2^(32+b) % d
  constexpr bool round_down = (2U * rem < d0); // check if fraction is less than 0.5
  u16x8 x1 = x;
  if (round_down) {
    x1 = x1 + u16x8(1); // round down mult and compensate by adding 1 to x
  }
  constexpr u16 mult1 = round_down ? mult : mult + 1;
  const __m128i multv = _mm_set1_epi16((i16)mult1); // broadcast mult
  __m128i xm = _mm_mulhi_epu16(x1, multv); // high part of 16x16->32 bit unsigned multiplication
  u16x8 q = _mm_srli_epi16(xm, (int)b); // shift right by b
  if constexpr (round_down) {
    Mask<16, 8> overfl = (x1 == u16x8(_mm_setzero_si128())); // check for overflow of x+1
    // deal with overflow (rarely needed)
    return select(overfl, u16x8(u16(mult1 >> (u16)b)), q);
  } else {
    return q; // no overflow possible
  }
}

// define u16x8 a / const_uint(d)
template<u32 d>
inline u16x8 operator/(const u16x8 a, ConstUint<d> /*b*/) {
  return divide_by_ui<d>(a);
}

// define u16x8 a / const_int(d)
template<int d>
inline u16x8 operator/(const u16x8 a, ConstInt<d> /*b*/) {
  static_assert(d >= 0, "Dividing unsigned integer by negative is ambiguous");
  return divide_by_ui<d>(a); // unsigned divide
}

// vector operator /= : divide
template<u32 d>
inline u16x8& operator/=(u16x8& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<i32 d>
inline u16x8& operator/=(u16x8& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// define i8x16 a / const_int(d)
template<int d>
inline i8x16 operator/(const i8x16 a, ConstInt<d> /*b*/) {
  // expand into two i16x8
  i16x8 low = extend_low(a) / ConstInt<d>();
  i16x8 high = extend_high(a) / ConstInt<d>();
  return compress(low, high);
}

// define i8x16 a / const_uint(d)
template<u32 d>
inline i8x16 operator/(const i8x16 a, ConstUint<d> /*b*/) {
  static_assert(u8(d) < 0x80U, "Dividing signed integer by overflowing unsigned");
  return a / ConstInt<d>(); // signed divide
}

// vector operator /= : divide
template<i32 d>
inline i8x16& operator/=(i8x16& a, ConstInt<d> b) {
  a = a / b;
  return a;
}
// vector operator /= : divide
template<u32 d>
inline i8x16& operator/=(i8x16& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// define u8x16 a / const_uint(d)
template<u32 d>
inline u8x16 operator/(const u8x16 a, ConstUint<d> /*b*/) {
  // expand into two u16x8c
  u16x8 low = extend_low(a) / ConstUint<d>();
  u16x8 high = extend_high(a) / ConstUint<d>();
  return compress(low, high);
}

// define u8x16 a / const_int(d)
template<int d>
inline u8x16 operator/(const u8x16 a, ConstInt<d> /*b*/) {
  static_assert(i8(d) >= 0, "Dividing unsigned integer by negative is ambiguous");
  return a / ConstUint<d>(); // unsigned divide
}

// vector operator /= : divide
template<u32 d>
inline u8x16& operator/=(u8x16& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<i32 d>
inline u8x16& operator/=(u8x16& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

#if STADO_INSTRUCTION_SET >= STADO_AVX512F // compact boolean vector CompactMask<16>
// to_bits: convert boolean vector to integer bitfield
inline u16 to_bits(const CompactMask<16> x) {
  return __mmask16(x);
}

// to_bits: convert boolean vector to integer bitfield
inline u8 to_bits(const CompactMask<8> x) {
  return u8(CompactMask<8>::Register(x));
}
#endif

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // compact boolean vectors, other sizes
// to_bits: convert boolean vector to integer bitfield
inline u8 to_bits(const CompactMask<4> x) {
  return __mmask8(x) & 0x0F;
}

// to_bits: convert boolean vector to integer bitfield
inline u8 to_bits(const CompactMask<2> x) {
  return __mmask8(x) & 0x03;
}
#endif
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_128I_HPP
