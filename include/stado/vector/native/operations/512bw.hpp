#ifndef INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_512BW_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_512BW_HPP

#include <array>
#include <cstdint>

#include "stado/instruction-set.hpp"
#include "stado/mask/compact.hpp"
#include "stado/vector/native/divisor.hpp"
#include "stado/vector/native/operations/512i.hpp"
#include "stado/vector/native/operations/i08x64.hpp"
#include "stado/vector/native/operations/i16x32.hpp"
#include "stado/vector/native/operations/u16x32.hpp"
#include "stado/vector/native/types.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
namespace stado {
// Permute vector of 32 16-bit integers.
// Index -1 gives 0, index any means don't care.
template<int... i0>
inline i16x32 permute32(const i16x32 a) {
  constexpr std::array<int, 32> indexs{i0...};
  __m512i y = a; // result
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<i16x32>(indexs);

  static_assert(sizeof...(i0) == 32, "permute32 must have 32 indexes");
  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm512_setzero_si512(); // just return zero
  }

  if constexpr (flags.perm) { // permutation needed
    if constexpr (flags.largeblock) { // use larger permutation
      constexpr std::array<int, 16> arr = largeblock_perm<32>(indexs); // permutation pattern
      y = permute16<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
                    arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(i32x16(a));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else if constexpr (flags.same_pattern) { // same pattern in all lanes
      if constexpr (flags.rotate) { // fits palignr. rotate within lanes
        y = _mm512_alignr_epi8(a, a, flags.rot_count);
      } else { // use pshufb
        constexpr std::array<int8_t, 64> bm = pshufb_mask<i16x32>(indexs);
        return _mm512_shuffle_epi8(a, i16x32().load(bm.data()));
      }
    } else { // different patterns in all lanes
      if constexpr (!flags.cross_lane) { // no lane crossing. Use pshufb
        constexpr std::array<int8_t, 64> bm = pshufb_mask<i16x32>(indexs);
        return _mm512_shuffle_epi8(a, i16x32().load(bm.data()));
      } else if constexpr (flags.rotate_big) { // fits full rotate
        constexpr uint8_t rot = flags.rot_count * 2; // rotate count
        constexpr uint8_t r1 = (rot >> 4 << 1) & 7;
        constexpr uint8_t r2 = (r1 + 2) & 7;
        __m512i y1 = a;
        __m512i y2 = a;
        if constexpr (r1 != 0) {
          y1 = _mm512_alignr_epi64(a, a, r1); // rotate 128-bit blocks
        }
        if constexpr (r2 != 0) {
          y2 = _mm512_alignr_epi64(a, a, r2); // rotate 128-bit blocks
        }
        y = _mm512_alignr_epi8(y2, y1, rot & 15);
      } else if constexpr (flags.broadcast && flags.rot_count == 0) {
        y = _mm512_broadcastw_epi16(_mm512_castsi512_si128(y)); // broadcast first element
      } else if constexpr (flags.zext) { // fits zero extension
        y = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(y)); // zero extension
        if constexpr (!flags.addz2) {
          return y;
        }
      }
#if defined(__AVX512VBMI2__)
      else if constexpr (flags.compress) {
        y = _mm512_maskz_compress_epi16(__mmask32(compress_mask(indexs)), y); // compress
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (flags.expand) {
        y = _mm512_maskz_expand_epi16(__mmask32(expand_mask(indexs)), y); // expand
        if constexpr (!flags.addz2) {
          return y;
        }
      }
#endif // AVX512VBMI2
      else { // full permute needed
        constexpr std::array<int16_t, 32> bm = perm_mask_broad<std::int16_t, 32>(indexs);
        y = _mm512_permutexvar_epi16(i16x32().load(bm.data()), y);
      }
    }
  }
  if constexpr (flags.zeroing) { // additional zeroing needed
    y = _mm512_maskz_mov_epi16(zero_mask<32>(indexs), y);
  }
  return y;
}

template<int... i0>
inline u16x32 permute32(const u16x32 a) {
  return u16x32(permute32<i0...>(i16x32(a)));
}

// Permute vector of 64 8-bit integers.
// Index -1 gives 0, index any means don't care.
template<int... i0>
inline i8x64 permute64(const i8x64 a) {
  constexpr std::array<int, 64> indexs{i0...};
  __m512i y = a; // result
  // get flags for possibilities that fit the permutation pattern
  constexpr PermFlags flags = perm_flags<i8x64>(indexs);

  static_assert(sizeof...(i0) == 64, "permute64 must have 64 indexes");
  static_assert(!flags.outofrange, "Index out of range in permute function");

  if constexpr (flags.allzero) {
    return _mm512_setzero_si512(); // just return zero
  }
  if constexpr (flags.perm) { // permutation needed

    if constexpr (flags.largeblock) { // use larger permutation
      constexpr std::array<int, 32> arr = largeblock_perm<64>(indexs); // permutation pattern
      y = permute32<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
                    arr[10], arr[11], arr[12], arr[13], arr[14], arr[15], arr[16], arr[17], arr[18],
                    arr[19], arr[20], arr[21], arr[22], arr[23], arr[24], arr[25], arr[26], arr[27],
                    arr[28], arr[29], arr[30], arr[31]>(i16x32(a));
      if (!flags.addz) {
        return y; // no remaining zeroing
      }
    } else {
      if constexpr (!flags.cross_lane) { // no lane crossing. Use pshufb
        constexpr std::array<int8_t, 64> bm = pshufb_mask<i8x64>(indexs);
        return _mm512_shuffle_epi8(a, i8x64().load(bm.data()));
      } else if constexpr (flags.rotate_big) { // fits full rotate
        constexpr uint8_t rot = flags.rot_count; // rotate count
        constexpr uint8_t r1 = (rot >> 4 << 1) & 7;
        constexpr uint8_t r2 = (r1 + 2) & 7;
        __m512i y1 = a;
        __m512i y2 = a;
        if constexpr (r1 != 0) {
          y1 = _mm512_alignr_epi64(y, y, r1); // rotate 128-bit blocks
        }
        if constexpr (r2 != 0) {
          y2 = _mm512_alignr_epi64(a, a, r2); // rotate 128-bit blocks
        }
        y = _mm512_alignr_epi8(y2, y1, rot & 15);
      } else if constexpr (flags.broadcast && flags.rot_count == 0) {
        y = _mm512_broadcastb_epi8(_mm512_castsi512_si128(y)); // broadcast first element
      } else if constexpr (flags.zext) { // fits zero extension
        y = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y)); // zero extension
        if constexpr (!flags.addz2) {
          return y;
        }
      }
#if defined(__AVX512VBMI2__)
      else if constexpr (flags.compress) {
        y = _mm512_maskz_compress_epi8(__mmask64(compress_mask(indexs)), y); // compress
        if constexpr (!flags.addz2) {
          return y;
        }
      } else if constexpr (flags.expand) {
        y = _mm512_maskz_expand_epi8(__mmask64(expand_mask(indexs)), y); // expand
        if constexpr (!flags.addz2) {
          return y;
        }
      }
#endif // AVX512VBMI2
      else { // full permute needed
#ifdef __AVX512VBMI__ // full permute instruction available
        constexpr std::array<int8_t, 64> bm = perm_mask_broad<std::int8_t, 64>(indexs);
        y = _mm512_permutexvar_epi8(i8x64().load(bm.data()), y);
#else
        // There is no 8-bit full permute. Use 16-bit permute
        // getevenmask: get permutation mask for destination bytes with even position
        auto getevenmask = [](const std::array<int, 64>& indexs) constexpr {
          std::array<uint16_t, 32> u = {0}; // list to return
          for (int i = 0; i < 64; i += 2) { // loop through even indexes
            uint16_t ix = indexs[i] & 63;
            // source bytes with odd position are in opposite 16-bit word becase of 32-bit rotation
            u[i >> 1] = ((ix >> 1) ^ (ix & 1)) | (((ix & 1) ^ 1) << 5);
          }
          return u;
        };
        // getoddmask: get permutation mask for destination bytes with odd position
        auto getoddmask = [](const std::array<int, 64>& indexs) constexpr {
          std::array<uint16_t, 32> u = {0}; // list to return
          for (int i = 1; i < 64; i += 2) { // loop through odd indexes
            uint16_t ix = indexs[i] & 63;
            u[i >> 1] = (ix >> 1) | ((ix & 1) << 5);
          }
          return u;
        };
        std::array<uint16_t, 32> evenmask = getevenmask(indexs);
        std::array<uint16_t, 32> oddmask = getoddmask(indexs);
        // Rotate to get odd bytes into even position, and vice versa.
        // There is no 16-bit rotate, use 32-bit rotate.
        // The wrong position of the odd bytes is compensated for in getevenmask
        __m512i ro = _mm512_rol_epi32(a, 8); // rotate
        __m512i yeven = _mm512_permutex2var_epi16(ro, i16x32().load(evenmask.data()),
                                                  a); // destination bytes with even position
        __m512i yodd = _mm512_permutex2var_epi16(ro, i16x32().load(oddmask.data()),
                                                 a); // destination bytes with odd  position
        __mmask64 maske = 0x5555555555555555; // mask for even position
        y = _mm512_mask_mov_epi8(yodd, maske, yeven); // interleave even and odd position bytes
#endif
      }
    }
  }
  if constexpr (flags.zeroing) { // additional zeroing needed
    y = _mm512_maskz_mov_epi8(zero_mask<64>(indexs), y);
  }
  return y;
}

template<int... i0>
inline u8x64 permute64(const u8x64 a) {
  return u8x64(permute64<i0...>(i8x64(a)));
}

/*****************************************************************************
 *
 *          NativeVector blend functions
 *
 *****************************************************************************/

// permute and blend i16x32
template<int... i0>
inline i16x32 blend32(const i16x32 a, const i16x32 b) {
  constexpr std::array<int, 32> indexs{i0...}; // indexes as array
  static_assert(sizeof...(i0) == 32, "blend32 must have 32 indexes");
  __m512i y = a; // result
  constexpr uint64_t flags =
    blend_flags<i16x32>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm512_setzero_si512(); // just return zero
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute32<i0...>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    // get permutation indexes
    constexpr std::array<int, 64> arr = blend_perm_indexes<32, 2>(indexs);
    return permute32<arr[32], arr[33], arr[34], arr[35], arr[36], arr[37], arr[38], arr[39],
                     arr[40], arr[41], arr[42], arr[43], arr[44], arr[45], arr[46], arr[47],
                     arr[48], arr[49], arr[50], arr[51], arr[52], arr[53], arr[54], arr[55],
                     arr[56], arr[57], arr[58], arr[59], arr[60], arr[61], arr[62], arr[63]>(b);
  }
  if constexpr ((flags & (blend_perma | blend_permb)) == 0) { // no permutation, only blending
    constexpr auto mb = uint32_t(make_bit_mask<32, 0x305>(indexs)); // blend mask
    y = _mm512_mask_mov_epi16(a, mb, b);
  } else if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 32-bit blocks
    constexpr std::array<int, 16> arr = largeblock_perm<32>(indexs); // get 32-bit blend pattern
    y = blend16<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
                arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]>(i32x16(a), i32x16(b));
    if (!(flags & blend_addz)) {
      return y; // no remaining zeroing
    }
  } else { // No special cases
    constexpr std::array<int16_t, 32> bm =
      perm_mask_broad<std::int16_t, 32>(indexs); // full permute
    y = _mm512_permutex2var_epi16(a, i16x32().load(bm.data()), b);
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
    y = _mm512_maskz_mov_epi16(zero_mask<32>(indexs), y);
  }
  return y;
}

template<int... i0>
inline u16x32 blend32(const u16x32 a, const u16x32 b) {
  return u16x32(blend32<i0...>(i16x32(a), i16x32(b)));
}

// permute and blend i8x64
template<int... i0>
inline i8x64 blend64(const i8x64 a, const i8x64 b) {
  constexpr std::array<int, 64> indexs{i0...}; // indexes as array
  static_assert(sizeof...(i0) == 64, "blend64 must have 64 indexes");
  __m512i y = a; // result
  constexpr uint64_t flags =
    blend_flags<i8x64>(indexs); // get flags for possibilities that fit the index pattern

  static_assert((flags & blend_outofrange) == 0, "Index out of range in blend function");

  if constexpr ((flags & blend_allzero) != 0) {
    return _mm512_setzero_si512(); // just return zero
  }

  if constexpr ((flags & blend_b) == 0) { // nothing from b. just permute a
    return permute64<i0...>(a);
  }
  if constexpr ((flags & blend_a) == 0) { // nothing from a. just permute b
    // get permutation indexes
    constexpr std::array<int, 128> arr = blend_perm_indexes<64, 2>(indexs);
    return permute64<
      arr[64], arr[65], arr[66], arr[67], arr[68], arr[69], arr[70], arr[71], arr[72], arr[73],
      arr[74], arr[75], arr[76], arr[77], arr[78], arr[79], arr[80], arr[81], arr[82], arr[83],
      arr[84], arr[85], arr[86], arr[87], arr[88], arr[89], arr[90], arr[91], arr[92], arr[93],
      arr[94], arr[95], arr[96], arr[97], arr[98], arr[99], arr[100], arr[101], arr[102], arr[103],
      arr[104], arr[105], arr[106], arr[107], arr[108], arr[109], arr[110], arr[111], arr[112],
      arr[113], arr[114], arr[115], arr[116], arr[117], arr[118], arr[119], arr[120], arr[121],
      arr[122], arr[123], arr[124], arr[125], arr[126], arr[127]>(b);
  }
  if constexpr ((flags & (blend_perma | blend_permb)) == 0) { // no permutation, only blending
    constexpr uint64_t mb = make_bit_mask<64, 0x306>(indexs); // blend mask
    y = _mm512_mask_mov_epi8(a, mb, b);
  } else if constexpr ((flags & blend_largeblock) != 0) { // blend and permute 16-bit blocks
    constexpr std::array<int, 32> arr = largeblock_perm<64>(indexs); // get 16-bit blend pattern
    y = blend32<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
                arr[10], arr[11], arr[12], arr[13], arr[14], arr[15], arr[16], arr[17], arr[18],
                arr[19], arr[20], arr[21], arr[22], arr[23], arr[24], arr[25], arr[26], arr[27],
                arr[28], arr[29], arr[30], arr[31]>(i16x32(a), i16x32(b));
    if (!(flags & blend_addz)) {
      return y; // no remaining zeroing
    }
  } else { // No special cases
#ifdef __AVX512VBMI__ // AVX512VBMI
    constexpr std::array<int8_t, 64> bm = perm_mask_broad<std::int8_t, 64>(indexs); // full permute
    y = _mm512_permutex2var_epi8(a, i8x64().load(bm.data()), b);
#else // split into two permutes
    constexpr std::array<int, 128> arr = blend_perm_indexes<64, 0>(indexs);
    __m512i ya =
      permute64<arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
                arr[10], arr[11], arr[12], arr[13], arr[14], arr[15], arr[16], arr[17], arr[18],
                arr[19], arr[20], arr[21], arr[22], arr[23], arr[24], arr[25], arr[26], arr[27],
                arr[28], arr[29], arr[30], arr[31], arr[32], arr[33], arr[34], arr[35], arr[36],
                arr[37], arr[38], arr[39], arr[40], arr[41], arr[42], arr[43], arr[44], arr[45],
                arr[46], arr[47], arr[48], arr[49], arr[50], arr[51], arr[52], arr[53], arr[54],
                arr[55], arr[56], arr[57], arr[58], arr[59], arr[60], arr[61], arr[62], arr[63]>(a);
    __m512i yb =
      permute64<arr[64], arr[65], arr[66], arr[67], arr[68], arr[69], arr[70], arr[71], arr[72],
                arr[73], arr[74], arr[75], arr[76], arr[77], arr[78], arr[79], arr[80], arr[81],
                arr[82], arr[83], arr[84], arr[85], arr[86], arr[87], arr[88], arr[89], arr[90],
                arr[91], arr[92], arr[93], arr[94], arr[95], arr[96], arr[97], arr[98], arr[99],
                arr[100], arr[101], arr[102], arr[103], arr[104], arr[105], arr[106], arr[107],
                arr[108], arr[109], arr[110], arr[111], arr[112], arr[113], arr[114], arr[115],
                arr[116], arr[117], arr[118], arr[119], arr[120], arr[121], arr[122], arr[123],
                arr[124], arr[125], arr[126], arr[127]>(b);
    uint64_t bm = make_bit_mask<64, 0x306>(indexs);
    y = _mm512_mask_mov_epi8(ya, bm, yb);
#endif
  }
  if constexpr ((flags & blend_zeroing) != 0) { // additional zeroing needed
    y = _mm512_maskz_mov_epi8(zero_mask<64>(indexs), y);
  }
  return y;
}

template<int... i0>
inline u8x64 blend64(const u8x64 a, const u8x64 b) {
  return u8x64(blend64<i0...>(i8x64(a), i8x64(b)));
}

/*****************************************************************************
 *
 *          NativeVector lookup functions
 *
 ******************************************************************************
 *
 * These functions use vector elements as indexes into a table.
 * The table is given as one or more vectors
 *
 *****************************************************************************/

// lookup in table of 64 int8_t values
inline i8x64 lookup64(const i8x64 index, const i8x64 table) {
#ifdef __AVX512VBMI__ // AVX512VBMI instruction set not supported yet (April 2019)
  return _mm512_permutexvar_epi8(index, table);
#else
  // broadcast each 128-bit lane, because int8_t shuffle is only within 128-bit lanes
  __m512i lane0 = _mm512_broadcast_i32x4(_mm512_castsi512_si128(table));
  __m512i lane1 = _mm512_shuffle_i64x2(table, table, 0x55);
  __m512i lane2 = _mm512_shuffle_i64x2(table, table, 0xAA);
  __m512i lane3 = _mm512_shuffle_i64x2(table, table, 0xFF);
  i8x64 laneselect = index >> 4; // upper part of index selects lane
  // select and permute from each lane
  i8x64 dat0 = _mm512_maskz_shuffle_epi8(laneselect == 0, lane0, index);
  i8x64 dat1 = _mm512_mask_shuffle_epi8(dat0, laneselect == 1, lane1, index);
  i8x64 dat2 = _mm512_maskz_shuffle_epi8(laneselect == 2, lane2, index);
  i8x64 dat3 = _mm512_mask_shuffle_epi8(dat2, laneselect == 3, lane3, index);
  return dat1 | dat3;
#endif
}

// lookup in table of 128 int8_t values
inline i8x64 lookup128(const i8x64 index, const i8x64 table1, const i8x64 table2) {
#ifdef __AVX512VBMI__ // AVX512VBMI instruction set not supported yet (April 2019)
  return _mm512_permutex2var_epi8(table1, index, table2);
#else
  using Vec16 = i16x32;
  // use 16-bits permute, which is included in AVX512BW
  __m512i ieven2 =
    _mm512_srli_epi16(index, 1); // even pos bytes of index / 2 (extra bits will be ignored)
  __m512i e1 =
    _mm512_permutex2var_epi16(table1, ieven2, table2); // 16-bits results for even pos index
  __mmask32 me1 = (Vec16(index) & 1) != Vec16(0); // even pos indexes are odd value
  __m512i e2 = _mm512_mask_srli_epi16(
    e1, me1, e1,
    8); // combined results for even pos index. get upper 8 bits down if index was odd
  __m512i iodd2 = _mm512_srli_epi16(index, 9); // odd  pos bytes of index / 2
  __m512i o1 =
    _mm512_permutex2var_epi16(table1, iodd2, table2); // 16-bits results for odd pos index
  __mmask32 mo1 = (Vec16(index) & 0x100) == Vec16(0); // odd pos indexes have even value
  __m512i o2 = _mm512_mask_slli_epi16(
    o1, mo1, o1, 8); // combined results for odd pos index. get lower 8 bits up if index was even
  __mmask64 maske = 0x5555555555555555; // mask for even position
  return _mm512_mask_mov_epi8(o2, maske, e2); // interleave even and odd position result
#endif
}

// lookup in table of 256 int8_t values.
// The complete table of all possible 256 byte values is contained in four vectors
// The index is treated as unsigned
inline i8x64 lookup256(const i8x64 index, const i8x64 table1, const i8x64 table2,
                       const i8x64 table3, const i8x64 table4) {
#ifdef __AVX512VBMI__ // AVX512VBMI instruction set not supported yet (April 2019)
  i8x64 d12 = _mm512_permutex2var_epi8(table1, index, table2);
  i8x64 d34 = _mm512_permutex2var_epi8(table3, index, table4);
  return select(index < 0, d34, d12); // use sign bit to select
#else
  // the AVX512BW version of lookup128 ignores upper bytes of index
  // (the compiler will optimize away common subexpressions of the two lookup128)
  i8x64 d12 = lookup128(index, table1, table2);
  i8x64 d34 = lookup128(index, table3, table4);
  return select(index < 0, d34, d12);
#endif
}

// lookup in table of 32 values
inline i16x32 lookup32(const i16x32 index, const i16x32 table) {
  return _mm512_permutexvar_epi16(index, table);
}

// lookup in table of 64 values
inline i16x32 lookup64(const i16x32 index, const i16x32 table1, const i16x32 table2) {
  return _mm512_permutex2var_epi16(table1, index, table2);
}

// lookup in table of 128 values
inline i16x32 lookup128(const i16x32 index, const i16x32 table1, const i16x32 table2,
                        const i16x32 table3, const i16x32 table4) {
  i16x32 d12 = _mm512_permutex2var_epi16(table1, index, table2);
  i16x32 d34 = _mm512_permutex2var_epi16(table3, index, table4);
  return select((index >> 6) != i16x32(0), d34, d12);
}

/*****************************************************************************
 *
 *          Byte shifts
 *
 *****************************************************************************/

// Function shift_bytes_up: shift whole vector left by b bytes.
template<unsigned int b>
inline i8x64 shift_bytes_up(const i8x64 a) {
  __m512i ahi;
  __m512i alo;
  if constexpr (b == 0) {
    return a;
  } else if constexpr ((b & 3) == 0) { // b is divisible by 4
    return _mm512_alignr_epi32(a, _mm512_setzero_si512(), (16 - (b >> 2)) & 15);
  } else if constexpr (b < 16) {
    alo = a;
    ahi = _mm512_maskz_shuffle_i64x2(0xFC, a, a, 0x90); // shift a 16 bytes up, zero lower part
  } else if constexpr (b < 32) {
    alo = _mm512_maskz_shuffle_i64x2(0xFC, a, a, 0x90); // shift a 16 bytes up, zero lower part
    ahi = _mm512_maskz_shuffle_i64x2(0xF0, a, a, 0x40); // shift a 32 bytes up, zero lower part
  } else if constexpr (b < 48) {
    alo = _mm512_maskz_shuffle_i64x2(0xF0, a, a, 0x40); // shift a 32 bytes up, zero lower part
    ahi = _mm512_maskz_shuffle_i64x2(0xC0, a, a, 0x00); // shift a 48 bytes up, zero lower part
  } else if constexpr (b < 64) {
    alo = _mm512_maskz_shuffle_i64x2(0xC0, a, a, 0x00); // shift a 48 bytes up, zero lower part
    ahi = _mm512_setzero_si512(); // zero
  } else {
    return _mm512_setzero_si512(); // zero
  }
  return _mm512_alignr_epi8(alo, ahi, 16 - (b & 0xF)); // shift within 16-bytes lane
}

// Function shift_bytes_down: shift whole vector right by b bytes
template<unsigned int b>
inline i8x64 shift_bytes_down(const i8x64 a) {
  if constexpr ((b & 3) == 0) { // b is divisible by 4
    return _mm512_alignr_epi32(_mm512_setzero_si512(), a, ((b >> 2) & 15));
  }
  __m512i ahi;
  __m512i alo;
  if constexpr (b < 16) {
    alo = _mm512_maskz_shuffle_i64x2(0x3F, a, a, 0x39); // shift a 16 bytes down, zero upper part
    ahi = a;
  } else if constexpr (b < 32) {
    alo = _mm512_maskz_shuffle_i64x2(0x0F, a, a, 0x0E); // shift a 32 bytes down, zero upper part
    ahi = _mm512_maskz_shuffle_i64x2(0x3F, a, a, 0x39); // shift a 16 bytes down, zero upper part
  } else if constexpr (b < 48) {
    alo = _mm512_maskz_shuffle_i64x2(0x03, a, a, 0x03); // shift a 48 bytes down, zero upper part
    ahi = _mm512_maskz_shuffle_i64x2(0x0F, a, a, 0x0E); // shift a 32 bytes down, zero upper part
  } else if constexpr (b < 64) {
    alo = _mm512_setzero_si512();
    ahi = _mm512_maskz_shuffle_i64x2(0x03, a, a, 0x03); // shift a 48 bytes down, zero upper part
  } else {
    return _mm512_setzero_si512(); // zero
  }
  return _mm512_alignr_epi8(alo, ahi, b & 0xF); // shift within 16-bytes lane
}

/*****************************************************************************
 *
 *          Functions for conversion between integer sizes
 *
 *****************************************************************************/

// Extend 8-bit integers to 16-bit integers, signed and unsigned

// Function extend_low : extends the low 32 elements to 16 bits with sign extension
inline i16x32 extend_low(const i8x64 a) {
  __m512i a2 = permute8<0, any, 1, any, 2, any, 3, any>(i64x8(a)); // get low 64-bit blocks
  CompactMask<64> sign = _mm512_cmpgt_epi8_mask(_mm512_setzero_si512(), a2); // 0 > a2
  __m512i ss = _mm512_maskz_set1_epi8(sign, -1);
  return _mm512_unpacklo_epi8(a2, ss); // interleave with sign extensions
}

// Function extend_high : extends the high 16 elements to 16 bits with sign extension
inline i16x32 extend_high(const i8x64 a) {
  __m512i a2 = permute8<4, any, 5, any, 6, any, 7, any>(i64x8(a)); // get low 64-bit blocks
  CompactMask<64> sign = _mm512_cmpgt_epi8_mask(_mm512_setzero_si512(), a2); // 0 > a2
  __m512i ss = _mm512_maskz_set1_epi8(sign, -1);
  return _mm512_unpacklo_epi8(a2, ss); // interleave with sign extensions
}

// Function extend_low : extends the low 16 elements to 16 bits with zero extension
inline u16x32 extend_low(const u8x64 a) {
  __m512i a2 = permute8<0, any, 1, any, 2, any, 3, any>(i64x8(a)); // get low 64-bit blocks
  return _mm512_unpacklo_epi8(a2, _mm512_setzero_si512()); // interleave with zero extensions
}

// Function extend_high : extends the high 19 elements to 16 bits with zero extension
inline u16x32 extend_high(const u8x64 a) {
  __m512i a2 = permute8<4, any, 5, any, 6, any, 7, any>(i64x8(a)); // get low 64-bit blocks
  return _mm512_unpacklo_epi8(a2, _mm512_setzero_si512()); // interleave with zero extensions
}

// Extend 16-bit integers to 32-bit integers, signed and unsigned

// Function extend_low : extends the low 8 elements to 32 bits with sign extension
inline i32x16 extend_low(const i16x32 a) {
  __m512i a2 = permute8<0, any, 1, any, 2, any, 3, any>(i64x8(a)); // get low 64-bit blocks
  CompactMask<32> sign = _mm512_cmpgt_epi16_mask(_mm512_setzero_si512(), a2); // 0 > a2
  __m512i ss = _mm512_maskz_set1_epi16(sign, -1);
  return _mm512_unpacklo_epi16(a2, ss); // interleave with sign extensions
}

// Function extend_high : extends the high 8 elements to 32 bits with sign extension
inline i32x16 extend_high(const i16x32 a) {
  __m512i a2 = permute8<4, any, 5, any, 6, any, 7, any>(i64x8(a)); // get low 64-bit blocks
  CompactMask<32> sign = _mm512_cmpgt_epi16_mask(_mm512_setzero_si512(), a2); // 0 > a2
  __m512i ss = _mm512_maskz_set1_epi16(sign, -1);
  return _mm512_unpacklo_epi16(a2, ss); // interleave with sign extensions
}

// Function extend_low : extends the low 8 elements to 32 bits with zero extension
inline u32x16 extend_low(const u16x32 a) {
  __m512i a2 = permute8<0, any, 1, any, 2, any, 3, any>(i64x8(a)); // get low 64-bit blocks
  return _mm512_unpacklo_epi16(a2, _mm512_setzero_si512()); // interleave with zero extensions
}

// Function extend_high : extends the high 8 elements to 32 bits with zero extension
inline u32x16 extend_high(const u16x32 a) {
  __m512i a2 = permute8<4, any, 5, any, 6, any, 7, any>(i64x8(a)); // get low 64-bit blocks
  return _mm512_unpacklo_epi16(a2, _mm512_setzero_si512()); // interleave with zero extensions
}

// Compress 16-bit integers to 8-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Overflow wraps around
inline i8x64 compress(const i16x32 low, const i16x32 high) {
  __mmask64 mask = 0x5555555555555555;
  __m512i lowm = _mm512_maskz_mov_epi8(mask, low); // bytes of low
  __m512i highm = _mm512_maskz_mov_epi8(mask, high); // bytes of high
  __m512i pk = _mm512_packus_epi16(lowm, highm); // unsigned pack
  __m512i in = _mm512_setr_epi32(0, 0, 2, 0, 4, 0, 6, 0, 1, 0, 3, 0, 5, 0, 7, 0);
  return _mm512_permutexvar_epi64(in, pk); // put in right place
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Signed, with saturation
inline i8x64 compress_saturated(const i16x32 low, const i16x32 high) {
  __m512i pk = _mm512_packs_epi16(low, high); // packed with signed saturation
  __m512i in = _mm512_setr_epi32(0, 0, 2, 0, 4, 0, 6, 0, 1, 0, 3, 0, 5, 0, 7, 0);
  return _mm512_permutexvar_epi64(in, pk); // put in right place
}

// Function compress : packs two vectors of 16-bit integers to one vector of 8-bit integers
// Unsigned, overflow wraps around
inline u8x64 compress(const u16x32 low, const u16x32 high) {
  return u8x64(compress((i16x32)low, (i16x32)high));
}

// Function compress : packs two vectors of 16-bit integers into one vector of 8-bit integers
// Unsigned, with saturation
inline u8x64 compress_saturated(const u16x32 low, const u16x32 high) {
  __m512i maxval = _mm512_set1_epi32(0x00FF00FF); // maximum value
  __m512i low1 = _mm512_min_epu16(low, maxval); // upper limit
  __m512i high1 = _mm512_min_epu16(high, maxval); // upper limit
  // this instruction saturates from signed 32 bit to unsigned 16 bit
  __m512i pk = _mm512_packus_epi16(low1, high1);
  __m512i in = _mm512_setr_epi32(0, 0, 2, 0, 4, 0, 6, 0, 1, 0, 3, 0, 5, 0, 7, 0);
  return _mm512_permutexvar_epi64(in, pk); // put in right place
}

// Compress 32-bit integers to 16-bit integers, signed and unsigned, with and without saturation

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
inline i16x32 compress(const i32x16 low, const i32x16 high) {
  __mmask32 mask = 0x55555555;
  __m512i lowm = _mm512_maskz_mov_epi16(mask, low); // words of low
  __m512i highm = _mm512_maskz_mov_epi16(mask, high); // words of high
  __m512i pk = _mm512_packus_epi32(lowm, highm); // unsigned pack
  __m512i in = _mm512_setr_epi32(0, 0, 2, 0, 4, 0, 6, 0, 1, 0, 3, 0, 5, 0, 7, 0);
  return _mm512_permutexvar_epi64(in, pk); // put in right place
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Signed with saturation
inline i16x32 compress_saturated(const i32x16 low, const i32x16 high) {
  __m512i pk = _mm512_packs_epi32(low, high); // pack with signed saturation
  __m512i in = _mm512_setr_epi32(0, 0, 2, 0, 4, 0, 6, 0, 1, 0, 3, 0, 5, 0, 7, 0);
  return _mm512_permutexvar_epi64(in, pk); // put in right place
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Overflow wraps around
inline u16x32 compress(const u32x16 low, const u32x16 high) {
  return u16x32(compress((i32x16)low, (i32x16)high));
}

// Function compress : packs two vectors of 32-bit integers into one vector of 16-bit integers
// Unsigned, with saturation
inline u16x32 compress_saturated(const u32x16 low, const u32x16 high) {
  __m512i maxval = _mm512_set1_epi32(0x0000FFFF); // maximum value
  __m512i low1 = _mm512_min_epu32(low, maxval); // upper limit
  __m512i high1 = _mm512_min_epu32(high, maxval); // upper limit
  // this instruction saturates from signed 32 bit to unsigned 16 bit
  __m512i pk = _mm512_packus_epi32(low1, high1);
  __m512i in = _mm512_setr_epi32(0, 0, 2, 0, 4, 0, 6, 0, 1, 0, 3, 0, 5, 0, 7, 0);
  return _mm512_permutexvar_epi64(in, pk); // put in right place
}

/*****************************************************************************
 *
 *          Integer division operators
 *
 *          Please see the file vectori128.h for explanation.
 *
 *****************************************************************************/

// vector operator / : divide each element by divisor

// vector of 32 16-bit signed integers
inline i16x32 operator/(const i16x32 a, const DivisorI16 d) {
  __m512i m = _mm512_broadcastq_epi64(d.getm()); // broadcast multiplier
  __m512i sgn = _mm512_broadcastq_epi64(d.getsign()); // broadcast sign of d
  __m512i t1 = _mm512_mulhi_epi16(a, m); // multiply high signed words
  __m512i t2 = _mm512_add_epi16(t1, a); // + a
  __m512i t3 = _mm512_sra_epi16(t2, d.gets1()); // shift right artihmetic
  __m512i t4 = _mm512_srai_epi16(a, 15); // sign of a
  __m512i t5 = _mm512_sub_epi16(t4, sgn); // sign of a - sign of d
  __m512i t6 = _mm512_sub_epi16(t3, t5); // + 1 if a < 0, -1 if d < 0
  return _mm512_xor_si512(t6, sgn); // change sign if divisor negative
}

// vector of 16 16-bit unsigned integers
inline u16x32 operator/(const u16x32 a, const DivisorU16 d) {
  __m512i m = _mm512_broadcastq_epi64(d.getm()); // broadcast multiplier
  __m512i t1 = _mm512_mulhi_epu16(a, m); // multiply high signed words
  __m512i t2 = _mm512_sub_epi16(a, t1); // subtract
  __m512i t3 = _mm512_srl_epi16(t2, d.gets1()); // shift right logical
  __m512i t4 = _mm512_add_epi16(t1, t3); // add
  return _mm512_srl_epi16(t4, d.gets2()); // shift right logical
}

// vector of 32 8-bit signed integers
inline i8x64 operator/(const i8x64 a, const DivisorI16 d) {
  // sign-extend even-numbered and odd-numbered elements to 16 bits
  i16x32 even = _mm512_srai_epi16(_mm512_slli_epi16(a, 8), 8);
  i16x32 odd = _mm512_srai_epi16(a, 8);
  i16x32 evend = even / d; // divide even-numbered elements
  i16x32 oddd = odd / d; // divide odd-numbered  elements
  oddd = _mm512_slli_epi16(oddd, 8); // shift left to put back in place
  __m512i res = _mm512_mask_mov_epi8(evend, 0xAAAAAAAAAAAAAAAA, oddd); // interleave even and odd
  return res;
}

// vector of 32 8-bit unsigned integers
inline u8x64 operator/(const u8x64 a, const DivisorU16 d) {
  // zero-extend even-numbered and odd-numbered elements to 16 bits
  u16x32 even = _mm512_maskz_mov_epi8(__mmask64(0x5555555555555555), a);
  u16x32 odd = _mm512_srli_epi16(a, 8);
  u16x32 evend = even / d; // divide even-numbered elements
  u16x32 oddd = odd / d; // divide odd-numbered  elements
  oddd = _mm512_slli_epi16(oddd, 8); // shift left to put back in place
  __m512i res = _mm512_mask_mov_epi8(evend, 0xAAAAAAAAAAAAAAAA, oddd); // interleave even and odd
  return res;
}

// vector operator /= : divide
inline i16x32& operator/=(i16x32& a, const DivisorI16 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline u16x32& operator/=(u16x32& a, const DivisorU16 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline i8x64& operator/=(i8x64& a, const DivisorI16 d) {
  a = a / d;
  return a;
}

// vector operator /= : divide
inline u8x64& operator/=(u8x64& a, const DivisorU16 d) {
  a = a / d;
  return a;
}

/*****************************************************************************
 *
 *          Integer division 2: divisor is a compile-time constant
 *
 *****************************************************************************/

// Divide i16x32 by compile-time constant
template<int d>
inline i16x32 divide_by_i(const i16x32 x) {
  constexpr auto d0 = int16_t(d); // truncate d to 16 bits
  static_assert(d0 != 0, "Integer division by zero");
  if constexpr (d0 == 1) {
    return x; // divide by  1
  }
  if constexpr (d0 == -1) {
    return -x; // divide by -1
  }
  if constexpr (uint16_t(d0) == 0x8000) {
    return _mm512_maskz_set1_epi16(
      x == i16x32(int16_t(0x8000)),
      1); // avoid overflow of abs(d). return (x == 0x80000000) ? 1 : 0;
  }
  constexpr uint16_t d1 = d0 > 0 ? d0 : -d0; // compile-time abs(d0)
  if constexpr ((d1 & (d1 - 1)) == 0) {
    // d is a power of 2. use shift
    constexpr int k = bit_scan_reverse_const(uint32_t(d1));
    __m512i sign;
    if constexpr (k > 1) {
      sign = _mm512_srai_epi16(x, k - 1);
    } else {
      sign = x; // k copies of sign bit
    }
    __m512i bias = _mm512_srli_epi16(sign, 16 - k); // bias = x >= 0 ? 0 : k-1
    __m512i xpbias = _mm512_add_epi16(x, bias); // x + bias
    __m512i q = _mm512_srai_epi16(xpbias, k); // (x + bias) >> k
    if (d0 > 0) {
      return q; // d0 > 0: return  q
    }
    return _mm512_sub_epi16(_mm512_setzero_si512(), q); // d0 < 0: return -q
  }
  // general case
  // ceil(log2(d)). (d < 2 handled above)
  constexpr int arr = bit_scan_reverse_const(uint16_t(d1 - 1)) + 1;
  constexpr auto mult = int16_t(1 + (1U << (15 + arr)) / uint32_t(d1) - 0x10000); // multiplier
  constexpr int shift1 = arr - 1;
  const DivisorI16 div(mult, shift1, d0 > 0 ? 0 : -1);
  return x / div;
}

// define i16x32 a / const_int(d)
template<int d>
inline i16x32 operator/(const i16x32 a, ConstInt<d> /*b*/) {
  return divide_by_i<d>(a);
}

// define i16x32 a / const_uint(d)
template<uint32_t d>
inline i16x32 operator/(const i16x32 a, ConstUint<d> /*b*/) {
  static_assert(d < 0x8000U, "Dividing signed integer by overflowing unsigned");
  return divide_by_i<int(d)>(a); // signed divide
}

// vector operator /= : divide
template<int32_t d>
inline i16x32& operator/=(i16x32& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<uint32_t d>
inline i16x32& operator/=(i16x32& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// Divide u16x32 by compile-time constant
template<uint32_t d>
inline u16x32 divide_by_ui(const u16x32 x) {
  constexpr auto d0 = uint16_t(d); // truncate d to 16 bits
  static_assert(d0 != 0, "Integer division by zero");
  if constexpr (d0 == 1) {
    return x; // divide by 1
  }
  constexpr int b = bit_scan_reverse_const(d0); // floor(log2(d))
  if constexpr ((d0 & (d0 - 1)) == 0) {
    // d is a power of 2. use shift
    return _mm512_srli_epi16(x, b); // x >> b
  }
  // general case (d > 2)
  constexpr auto mult = uint16_t((uint32_t(1) << (b + 16)) / d0); // multiplier = 2^(32+b) / d
  constexpr uint32_t rem =
    (uint32_t(1) << (b + 16)) - uint32_t(d0) * mult; // remainder 2^(32+b) % d
  constexpr bool round_down = (2 * rem < d0); // check if fraction is less than 0.5
  u16x32 x1 = x;
  if constexpr (round_down) {
    x1 = x1 + 1; // round down mult and compensate by adding 1 to x
  }
  constexpr uint16_t mult1 = round_down ? mult : mult + 1;
  const __m512i multv = _mm512_set1_epi16(mult1); // broadcast mult
  __m512i xm = _mm512_mulhi_epu16(x1, multv); // high part of 16x16->32 bit unsigned multiplication
  u16x32 q = _mm512_srli_epi16(xm, b); // shift right by b
  if constexpr (round_down) {
    CompactMask<32> overfl = (x1 == u16x32(_mm512_setzero_si512())); // check for overflow of x+1
    return select(overfl, u16x32(uint32_t(mult1 >> b)),
                  q); // deal with overflow (rarely needed)
  } else {
    return q; // no overflow possible
  }
}

// define u16x32 a / const_uint(d)
template<uint32_t d>
inline u16x32 operator/(const u16x32 a, ConstUint<d> /*b*/) {
  return divide_by_ui<d>(a);
}

// define u16x32 a / const_int(d)
template<int d>
inline u16x32 operator/(const u16x32 a, ConstInt<d> /*b*/) {
  static_assert(d >= 0, "Dividing unsigned integer by negative is ambiguous");
  return divide_by_ui<d>(a); // unsigned divide
}

// vector operator /= : divide
template<uint32_t d>
inline u16x32& operator/=(u16x32& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<int32_t d>
inline u16x32& operator/=(u16x32& a, ConstInt<d> b) {
  a = a / b;
  return a;
}

// define i8x64 a / const_int(d)
template<int d>
inline i8x64 operator/(const i8x64 a, ConstInt<d> /*b*/) {
  // expand into two i16x32
  i16x32 low = extend_low(a) / ConstInt<d>();
  i16x32 high = extend_high(a) / ConstInt<d>();
  return compress(low, high);
}

// define i8x64 a / const_uint(d)
template<uint32_t d>
inline i8x64 operator/(const i8x64 a, ConstUint<d> /*b*/) {
  static_assert(uint8_t(d) < 0x80, "Dividing signed integer by overflowing unsigned");
  return a / ConstInt<d>(); // signed divide
}

// vector operator /= : divide
template<int32_t d>
inline i8x64& operator/=(i8x64& a, ConstInt<d> b) {
  a = a / b;
  return a;
}
// vector operator /= : divide
template<uint32_t d>
inline i8x64& operator/=(i8x64& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// define u8x64 a / const_uint(d)
template<uint32_t d>
inline u8x64 operator/(const u8x64 a, ConstUint<d> /*b*/) {
  // expand into two u16x32
  u16x32 low = extend_low(a) / ConstUint<d>();
  u16x32 high = extend_high(a) / ConstUint<d>();
  return compress(low, high);
}

// define u8x64 a / const_int(d)
template<int d>
inline u8x64 operator/(const u8x64 a, ConstInt<d> /*b*/) {
  static_assert(int8_t(d) >= 0, "Dividing unsigned integer by negative is ambiguous");
  return a / ConstUint<d>(); // unsigned divide
}

// vector operator /= : divide
template<uint32_t d>
inline u8x64& operator/=(u8x64& a, ConstUint<d> b) {
  a = a / b;
  return a;
}

// vector operator /= : divide
template<int32_t d>
inline u8x64& operator/=(u8x64& a, ConstInt<d> b) {
  a = a / b;
  return a;
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_OPERATIONS_512BW_HPP
