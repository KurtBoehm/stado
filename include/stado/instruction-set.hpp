#ifndef INCLUDE_STADO_INSTRUCTION_SET_HPP
#define INCLUDE_STADO_INSTRUCTION_SET_HPP

// Allow the use of floating point permute instructions on integer vectors.
// Some CPU's have an extra latency of 1 or 2 clock cycles for this, but
// it may still be faster than alternative implementations:
#define ALLOW_FP_PERMUTE true

// Macro to indicate 64 bit mode
#if (defined(_M_AMD64) || defined(_M_X64) || defined(__amd64)) && !defined(__x86_64__)
#define __x86_64__ 1 // There are many different macros for this, decide on only one
#endif

// The following values of STADO_INSTRUCTION_SET are currently defined:
// 2:  SSE2
#define STADO_SSE2 2
// 3:  SSE3
#define STADO_SSE3 3
// 4:  SSSE3
#define STADO_SSSE3 4
// 5:  SSE4.1
#define STADO_SSE4_1 5
// 6:  SSE4.2
#define STADO_SSE4_2 6
// 7:  AVX
#define STADO_AVX 7
// 8:  AVX2
#define STADO_AVX2 8
// 9:  AVX-512F
#define STADO_AVX512F 9
// 10: AVX-512BW/DQ/VL, supported on all AVX-512 CPUS other than Xeon Phi
#define STADO_AVX512SKL 10

// Find instruction set from compiler macros if STADO_INSTRUCTION_SET is not defined.
// Note: Some of these macros are not defined in Microsoft compilers
#ifndef STADO_INSTRUCTION_SET
#if defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#define STADO_INSTRUCTION_SET STADO_AVX512SKL
#elif defined(__AVX512F__) || defined(__AVX512__)
#define STADO_INSTRUCTION_SET STADO_AVX512F
#elif defined(__AVX2__)
#define STADO_INSTRUCTION_SET STADO_AVX2
#elif defined(__AVX__)
#define STADO_INSTRUCTION_SET STADO_AVX
#elif defined(__SSE4_2__)
#define STADO_INSTRUCTION_SET STADO_SSE4_2
#elif defined(__SSE4_1__)
#define STADO_INSTRUCTION_SET STADO_SSE4_1
#elif defined(__SSSE3__)
#define STADO_INSTRUCTION_SET STADO_SSSE3
#elif defined(__SSE3__)
#define STADO_INSTRUCTION_SET STADO_SSE3
#elif defined(__SSE2__) || defined(__x86_64__)
#define STADO_INSTRUCTION_SET STADO_SSE2
#elif defined(__SSE__)
#define STADO_INSTRUCTION_SET 1
#elif defined(_M_IX86_FP) // Defined in MS compiler. 1: SSE, 2: SSE2
#define STADO_INSTRUCTION_SET _M_IX86_FP
#else
#define STADO_INSTRUCTION_SET 0
#endif // instruction set defines
#endif // STADO_INSTRUCTION_SET

#if STADO_INSTRUCTION_SET >= STADO_AVX2 && !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
// Prevent error message in g++ and Clang when using FMA intrinsics with avx2:
#if !defined(DISABLE_WARNING_AVX2_WITHOUT_FMA)
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#endif
#elif !defined(__clang__)
#define __FMA__ 1
#endif
#endif

// Header files for non-vector intrinsic functions including _BitScanReverse(int),
// __cpuid(int[4],int), _xgetbv(int)
#ifdef _MSC_VER // Microsoft compiler or compatible Intel compiler
#include <intrin.h>
#else
#include <x86intrin.h> // Gcc or Clang compiler
#endif

#include <array>
#include <cstdlib> // define abs(int)
#include <limits>
#include <type_traits>

#include "stado/defs.hpp"

// GCC version
#if defined(__GNUC__) && !defined(GCC_VERSION) && !defined(__clang__)
#define GCC_VERSION ((__GNUC__) * 10000 + (__GNUC_MINOR__) * 100 + (__GNUC_PATCHLEVEL__))
#endif

// Clang version
#if defined(__clang__)
#define CLANG_VERSION ((__clang_major__) * 10000 + (__clang_minor__) * 100 + (__clang_patchlevel__))
// Problem: The version number is not consistent across platforms
// http://llvm.org/bugs/show_bug.cgi?id=12643
// Apple bug 18746972
#endif

// Fix problem with non-overloadable macros named min and max in WinDef.h
#ifdef _MSC_VER
#if defined(_WINDEF_) && defined(min) && defined(max)
#undef min
#undef max
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

// warning for poor support for AVX512F in MS compiler
#ifndef __INTEL_COMPILER
#if STADO_INSTRUCTION_SET == STADO_AVX512F
#pragma message("Warning: MS compiler cannot generate code for AVX512F without AVX512DQ")
#endif
#if _MSC_VER < 1920 && STADO_INSTRUCTION_SET > STADO_AVX2
#pragma message( \
  "Warning: Your compiler has poor support for AVX512. Code may be erroneous.\nPlease use a newer compiler version or a different compiler!")
#endif
#endif // __INTEL_COMPILER
#endif // _MSC_VER

/* Intel compiler problem:
The Intel compiler currently cannot compile version 2.00 of VCL. It seems to have
a problem with constexpr function returns not being constant enough.
*/
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER < 9999
#error The Intel compiler version 19.00 cannot compile VCL version 2. Use Version 1.xx of VCL instead
#endif

/* Clang problem:
The Clang compiler treats the intrinsic vector types __m128, __m128i, and __m128d as identical.
See the bug report at https://bugs.llvm.org/show_bug.cgi?id=17164
Additional problem: The version number is not consistent across platforms. The Apple build has
different version numbers. We have to rely on __apple_build_version__ on the Mac platform:
http://llvm.org/bugs/show_bug.cgi?id=12643
We have to make switches here when - hopefully - the error some day has been fixed.
We need different version checks with and whithout __apple_build_version__
*/
#if (defined(__clang__) || defined(__apple_build_version__)) && !defined(__INTEL_COMPILER)
#define FIX_CLANG_VECTOR_ALIAS_AMBIGUITY
#endif

#if defined(GCC_VERSION) && GCC_VERSION < 99999 && !defined(__clang__)
// To do: add gcc version that has these zero-extension intrinsics
#define ZEXT_MISSING // Gcc 7.4.0 does not have _mm256_zextsi128_si256 and similar functions
#endif

namespace stado {
// Constant for indicating don't care in permute and blend functions.
// V_DC is -256 in Vector class library version 1.xx
// V_DC can be any value less than -1 in Vector class library version 2.00
inline constexpr int any = -256;

/*****************************************************************************
 *
 *    Helper functions that depend on instruction set, compiler, or platform
 *
 *****************************************************************************/

// Define popcount function. Gives sum of bits
#if STADO_INSTRUCTION_SET >= STADO_SSE4_2
// The popcnt instruction is not officially part of the SSE4.2 instruction set,
// but available in all known processors with SSE4.2
static inline u32 vml_popcnt(u32 a) {
  return u32(_mm_popcnt_u32(a)); // Intel intrinsic. Supported by gcc and clang
}
static inline i64 vml_popcnt(u64 a) {
  return _mm_popcnt_u64(a); // Intel intrinsic.
}
#else // no SSE4.2
static inline u32 vml_popcnt(u32 a) {
  // popcnt instruction not available
  u32 b = a - ((a >> 1U) & 0x55555555U);
  u32 c = (b & 0x33333333U) + ((b >> 2U) & 0x33333333U);
  u32 d = (c + (c >> 4)) & 0x0F0F0F0FU;
  u32 e = d * 0x01010101;
  return e >> 24U;
}
static inline i64 vml_popcnt(u64 a) {
  return vml_popcnt(u32(a >> 32U)) + vml_popcnt(u32(a));
}
#endif

// Define bit-scan-forward function. Gives index to lowest set bit
#if defined(__GNUC__) || defined(__clang__)
// gcc and Clang have no bit_scan_forward intrinsic
#if defined(__clang__) // fix clang bug
// Clang uses a k register as parameter a when inlined from horizontal_find_first
__attribute__((noinline))
#endif
static u32
bit_scan_forward(u32 a) {
  u32 r;
  __asm("bsfl %1, %0" : "=r"(r) : "r"(a) :);
  return r;
}
static inline u32 bit_scan_forward(u64 a) {
  const auto lo = u32(a);
  if (lo != 0U) {
    return bit_scan_forward(lo);
  }
  const auto hi = u32(a >> 32U);
  return bit_scan_forward(hi) + 32;
}
#else // other compilers
static inline u32 bit_scan_forward(u32 a) {
  // defined in intrin.h for MS and Intel compilers
  unsigned long r;
  _BitScanForward(&r, a);
  return r;
}
static inline u32 bit_scan_forward(u64 a) {
  // defined in intrin.h for MS and Intel compilers
  unsigned long r;
  _BitScanForward64(&r, a);
  return (u32)r;
}
#endif

// Define bit-scan-reverse function. Gives index to highest set bit = floor(log2(a))
#if defined(__GNUC__) || defined(__clang__)
static inline u32 bit_scan_reverse(u32 a) __attribute__((pure));
static inline u32 bit_scan_reverse(u32 a) {
  u32 r;
  __asm("bsrl %1, %0" : "=r"(r) : "r"(a) :);
  return r;
}
static inline u32 bit_scan_reverse(u64 a) {
  u64 r;
  __asm("bsrq %1, %0" : "=r"(r) : "r"(a) :);
  return u32(r);
}
#else
static inline u32 bit_scan_reverse(u32 a) {
  // defined in intrin.h for MS and Intel compilers
  unsigned long r;
  _BitScanReverse(&r, a);
  return r;
}
static inline u32 bit_scan_reverse(u64 a) {
  // defined in intrin.h for MS and Intel compilers
  unsigned long r;
  _BitScanReverse64(&r, a);
  return r;
}
#endif

// Same function, for compile-time constants
constexpr int bit_scan_reverse_const(const u64 n) {
  if (n == 0) {
    return -1;
  }
  u64 a = n;
  u64 b = 0;
  u64 j = 64;
  u64 k = 0;
  do {
    j >>= 1U;
    k = u64{1} << j;
    if (a >= k) {
      a >>= j;
      b += j;
    }
  } while (j > 0);
  return int(b);
}

/*****************************************************************************
 *
 *    Common templates
 *
 *****************************************************************************/

// represent compile-time signed integer constant
template<i32 tNum>
class ConstInt {};
// represent compile-time unsigned integer constant
template<u32 tNum>
class ConstUint {};

// template for producing quiet NAN
template<typename TVec>
static inline TVec nan_vec(u32 payload = 0x100) {
  if constexpr (std::is_same_v<typename TVec::Element, f64>) {
    // f64
    union {
      u64 q;
      f64 f;
    } ud;
    // n is left justified to avoid loss of NAN payload when converting to f32
    ud.q = 0x7FF8000000000000U | u64(payload) << 29U;
    return TVec(ud.f);
  } else {
    // f32 will be converted to f64 if necessary
    union {
      u32 i;
      f32 f;
    } uf;
    uf.i = 0x7FC00000U | (payload & 0x003FFFFFU);
    return TVec(uf.f);
  }
}

/*****************************************************************************
 *
 *    Helper functions for permute and blend functions
 *
 *****************************************************************************/

template<typename T>
struct AllBitsSet {
  using Type = T;
  static constexpr T value =
    std::is_signed_v<T> ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
};

template<>
struct AllBitsSet<f32> {
  using Type = u32;
  static constexpr u32 value = std::numeric_limits<u32>::max();
};

template<>
struct AllBitsSet<f64> {
  using Type = u64;
  static constexpr u64 value = std::numeric_limits<u64>::max();
};

// zero_mask: return a compact bit mask mask for zeroing using AVX512 mask.
// Parameter a is a reference to a constexpr int array of permutation indexes
template<std::size_t tSize>
constexpr auto zero_mask(const std::array<int, tSize>& a) {
  u64 mask = 0;
  for (std::size_t i = 0; i < tSize; ++i) {
    if (a[i] >= 0) {
      mask |= u64{1} << i;
    }
  }
  if constexpr (tSize <= 8) {
    return u8(mask);
  } else if constexpr (tSize <= 16) {
    return u16(mask);
  } else if constexpr (tSize <= 32) {
    return u32(mask);
  } else {
    return mask;
  }
}

// zero_mask_broad: return a broad byte mask for zeroing.
// Parameter a is a reference to a constexpr int array of permutation indexes
template<typename T, std::size_t tSize>
constexpr auto zero_mask_broad(const std::array<int, tSize>& indices) {
  using Data = typename AllBitsSet<T>::Type;
  std::array<Data, tSize> u{0};
  for (std::size_t i = 0; i < tSize; ++i) {
    u[i] = indices[i] >= 0 ? AllBitsSet<T>::value : 0;
  }
  return u;
}

// make_bit_mask: return a compact mask of bits from a list of N indexes:
// B contains options indicating how to gather the mask
// bit 0-7 in B indicates which bit in each index to collect
// bit 8 = 0x100:  set 1 in the lower half of the bit mask if the indicated bit is 1.
// bit 8 = 0    :  set 1 in the lower half of the bit mask if the indicated bit is 0.
// bit 9 = 0x200:  set 1 in the upper half of the bit mask if the indicated bit is 1.
// bit 9 = 0    :  set 1 in the upper half of the bit mask if the indicated bit is 0.
// bit 10 = 0x400: set 1 in the bit mask if the corresponding index is -1 or any
// Parameter a is a reference to a constexpr int array of permutation indexes
template<std::size_t tSize, int tBits>
constexpr u64 make_bit_mask(const std::array<int, tSize>& indices) {
  // return value
  u64 r = 0;
  // index to selected bit
  const auto j = u8(tBits & 0xFF);
  for (std::size_t i = 0; i < tSize; ++i) {
    const int ix = indices[i];
    // bit number i in r
    u64 s = 0;
    if (ix < 0) {
      // -1 or any
      s = (tBits >> 10) & 1;
    } else {
      // extract selected bit
      s = ((u32)ix >> j) & 1;
      // 1 if bit not flipped
      u64 f = 0;
      if (i < tSize / 2) {
        // lower half
        f = (tBits >> 8) & 1;
      } else {
        // upper half
        f = (tBits >> 9) & 1;
      }
      // flip bit if needed
      s ^= f ^ 1U;
    }
    // set bit in return value
    r |= s << i;
  }
  return r;
}

// make_broad_mask: Convert a bit mask m to a broad mask
// The return value will be a broad boolean mask with elementsize matching vector class V
template<typename T, std::size_t tSize>
constexpr auto make_broad_mask(const u64 m) {
  using Data = typename AllBitsSet<T>::Type;
  std::array<Data, tSize> u{0};
  for (std::size_t i = 0; i < tSize; ++i) {
    u[i] = ((m >> i) & 1U) != 0 ? AllBitsSet<T>::value : 0;
  }
  return u;
}

// perm_mask_broad: return a mask for permutation by a vector register index.
// Parameter A is a reference to a constexpr int array of permutation indexes
template<typename T, std::size_t tSize>
constexpr auto perm_mask_broad(const std::array<int, tSize>& indices) {
  std::array<T, tSize> u{0};
  for (std::size_t i = 0; i < tSize; ++i) {
    u[i] = T(indices[i]);
  }
  return u;
}

struct PermFlags {
  // Needs zeroing.
  bool zeroing : 1 {};
  // Permutation needed.
  bool perm : 1 {};
  // All is zero or don't care.
  bool allzero : 1 {};
  // Fits permute with a larger block size (e.g permute i64x2 instead of i32x4).
  bool largeblock : 1 {};
  // Additional zeroing needed after permute with larger block size or shift.
  bool addz : 1 {};
  // Additional zeroing needed after perm_zext, perm_compress, or perm_expand.
  bool addz2 : 1 {};
  // Permutation crossing 128-bit lanes.
  bool cross_lane : 1 {};
  // Same permute pattern in all 128-bit lanes.
  bool same_pattern : 1 {};
  // Permutation pattern fits punpckh instruction.
  bool punpckh : 1 {};
  // Permutation pattern fits punpckl instruction.
  bool punpckl : 1 {};
  // Permutation pattern fits 128-bit rotation within lanes. 4 bit byte count returned in rot_count.
  bool rotate : 1 {};
  // Permutation pattern fits swap of adjacent vector elements.
  bool swap : 1 {};
  // Permutation pattern fits shift right within lanes. 4 bit count returned in rot_count.
  bool shright : 1 {};
  // Permutation pattern fits shift left within lanes. Negative count returned in rot_count.
  bool shleft : 1 {};
  // Permutation pattern fits rotation across lanes. 6 bit count returned in rot_count.
  bool rotate_big : 1 {};
  // Permutation pattern fits broadcast of a single element.
  bool broadcast : 1 {};
  // Permutation pattern fits zero extension.
  bool zext : 1 {};
  // Permutation pattern fits vpcompress instruction.
  bool compress : 1 {};
  // Permutation pattern fits vpexpand instruction.
  bool expand : 1 {};
  // Index out of range.
  bool outofrange : 1 {};
  // Rotate or shift count.
  u8 rot_count{};
  // Pattern for pshufd if same_pattern and elementsize >= 4.
  u8 ipattern{};
};

template<typename TVec>
constexpr PermFlags perm_flags(const std::array<int, TVec::size>& a) {
  // number of elements
  constexpr std::size_t size = TVec::size;
  // return value
  PermFlags r{.allzero = true, .largeblock = true, .same_pattern = true};
  // number of 128-bit lanes
  constexpr u32 nlanes = sizeof(TVec) / 16;
  // elements per lane
  constexpr u32 lanesize = size / nlanes;
  // size of each vector element
  constexpr u32 elementsize = sizeof(TVec) / size;
  // rotate left count
  u32 rot = 999;
  // index to broadcasted element
  i32 broadc = 999;
  // remember certain patterns that do not fit
  u32 patfail = 0;
  // remember certain patterns need extra zeroing
  u32 addz2 = 0;
  // last index in perm_compress fit
  i32 compresslasti = -1;
  // last position in perm_compress fit
  i32 compresslastp = -1;
  // last index in perm_expand fit
  i32 expandlasti = -1;
  // last position in perm_expand fit
  i32 expandlastp = -1;

  std::array<int, lanesize> lanepattern{0}; // pattern in each lane

  for (std::size_t i = 0; i < size; ++i) { // loop through indexes
    const int ix = a[i]; // current index
    // meaning of ix: -1 = set to zero, any = don't care, non-negative value = permute.
    if (ix == -1) {
      // zeroing requested
      r.zeroing = true;
    } else if (ix != any && u32(ix) >= size) {
      // index out of range
      r.outofrange = true;
    }
    if (ix >= 0) {
      // not all zero
      r.allzero = false;
      if (ix != (int)i) {
        // needs permutation
        r.perm = true;
      }
      if (broadc == 999) {
        // remember broadcast index
        broadc = ix;
      } else if (broadc != ix) {
        // does not fit broadcast
        broadc = 1000;
      }
    }
    // check if pattern fits a larger block size:
    // even indexes must be even, odd indexes must fit the preceding even index + 1
    if ((i & 1) == 0) {
      // even index
      if (ix >= 0 && (ix & 1)) {
        // not even. does not fit larger block size
        r.largeblock = false;
      }
      // next odd index
      int iy = a[i + 1];
      if (iy >= 0 && (iy & 1) == 0) {
        // not odd. does not fit larger block size
        r.largeblock = false;
      }
      if (ix >= 0 && iy >= 0 && iy != ix + 1) {
        // does not fit preceding index + 1
        r.largeblock = false;
      }
      if (ix == -1 && iy >= 0) {
        // needs additional zeroing at current block size
        r.addz = true;
      }
      if (iy == -1 && ix >= 0) {
        // needs additional zeroing at current block size
        r.addz = true;
      }
    }
    // current lane
    const std::size_t lane = i / lanesize;
    if (lane == 0) {
      // first lane, or no pattern yet
      // save pattern
      lanepattern[i] = ix;
    }
    // check if crossing lanes
    if (ix >= 0) {
      // source lane
      u32 lanei = u32(ix) / lanesize;
      if (lanei != lane) {
        // crossing lane
        r.cross_lane = true;
      }
    }
    // check if same pattern in all lanes
    if (lane != 0 && ix >= 0) {
      // not first lane
      // index into lanepattern
      int j1 = i - int(lane * lanesize);
      // pattern within lane
      int jx = ix - int(lane * lanesize);
      if (jx < 0 || jx >= (int)lanesize) {
        // source is in another lane
        r.same_pattern = false;
      }
      if (lanepattern[j1] < 0) {
        // pattern not known from previous lane
        lanepattern[j1] = jx;
      } else {
        if (lanepattern[j1] != jx) {
          // not same pattern
          r.same_pattern = false;
        }
      }
    }
    if (ix >= 0) {
      // check if pattern fits zero extension (perm_zext)
      if (u32(ix * 2) != i) {
        // does not fit zero extension
        patfail |= 1U;
      }
      // check if pattern fits compress (perm_compress)
      if (ix > compresslasti && ix - compresslasti >= (int)i - compresslastp) {
        if ((int)i - compresslastp > 1) {
          // perm_compress may need additional zeroing
          addz2 |= 2U;
        }
        compresslasti = ix;
        compresslastp = i;
      } else {
        // does not fit perm_compress
        patfail |= 2U;
      }
      // check if pattern fits expand (perm_expand)
      if (ix > expandlasti && ix - expandlasti <= (int)i - expandlastp) {
        if (ix - expandlasti > 1) {
          addz2 |= 4; // perm_expand may need additional zeroing
        }
        expandlasti = ix;
        expandlastp = i;
      } else {
        // does not fit perm_compress
        patfail |= 4U;
      }
    } else if (ix == -1) {
      if ((i & 1) == 0) {
        // zero extension needs additional zeroing
        addz2 |= 1U;
      }
    }
  }
  if (!r.perm) {
    // more checks are superfluous
    return r;
  }

  if (!r.largeblock) {
    // remove irrelevant flag
    r.addz = false;
  }
  if (r.cross_lane) {
    // remove irrelevant flag
    r.same_pattern = false;
  }
  if ((patfail & 1) == 0) {
    // fits zero extension
    r.zext = true;
    if ((addz2 & 1) != 0) {
      r.addz2 = true;
    }
  } else if ((patfail & 2) == 0) {
    // fits compression
    r.compress = true;
    if ((addz2 & 2) != 0) {
      // check if additional zeroing needed
      for (i32 j = 0; j < compresslastp; ++j) {
        if (a[j] == -1) {
          r.addz2 = true;
        }
      }
    }
  } else if ((patfail & 4) == 0) {
    // fits expansion
    r.expand = true;
    if ((addz2 & 4) != 0) {
      // check if additional zeroing needed
      for (i32 j = 0; j < expandlastp; ++j) {
        if (a[j] == -1) {
          r.addz2 = true;
        }
      }
    }
  }

  if (r.same_pattern) {
    // same pattern in all lanes. check if it fits specific patterns
    // fits rotate
    bool fit = true;
    // fits swap
    bool fitswap = true;
    // fit shift or rotate
    for (std::size_t i = 0; i < lanesize; ++i) {
      if (lanepattern[i] >= 0) {
        u32 rot1 = u32(lanepattern[i] + lanesize - i) % lanesize;
        if (rot == 999) {
          rot = rot1;
        } else {
          // check if fit
          if (rot != rot1) {
            fit = false;
          }
        }
        if ((u32)lanepattern[i] != (i ^ 1)) {
          fitswap = false;
        }
      }
    }
    // prevent out of range values
    rot &= lanesize - 1;
    if (fitswap) {
      r.swap = true;
    }
    if (fit) {
      // fits rotate, and possibly shift
      // rotate right count in bytes
      u64 rot2 = (rot * elementsize) & 0xF;
      // put shift/rotate count in output bit 16-19
      r.rot_count = rot2;
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
      r.rotate = true; // allow palignr
#endif
      // fit shift left
      fit = true;
      for (std::size_t i = 0; i < lanesize - rot; ++i) {
        // check if first rot elements are zero or don't care
        if (lanepattern[i] >= 0) {
          fit = false;
        }
      }
      if (fit) {
        r.shleft = true;
        for (std::size_t i = lanesize - rot; i < lanesize; ++i) {
          if (lanepattern[i] == -1) {
            // additional zeroing needed
            r.addz = true;
          }
        }
      }
      // fit shift right
      fit = true;
      for (std::size_t i = lanesize - rot; i < lanesize; ++i) {
        // check if last (lanesize-rot) elements are zero or don't care
        if (lanepattern[i] >= 0) {
          fit = false;
        }
      }
      if (fit) {
        r.shright = true;
        for (std::size_t i = 0; i < lanesize - rot; ++i) {
          if (lanepattern[i] == -1) {
            // additional zeroing needed
            r.addz = true;
          }
        }
      }
    }
    // fit punpckhi
    fit = true;
    u32 j2 = lanesize / 2;
    for (std::size_t i = 0; i < lanesize; ++i) {
      if (lanepattern[i] >= 0 && lanepattern[i] != (int)j2) {
        fit = false;
      }
      if ((i & 1) != 0) {
        j2++;
      }
    }
    if (fit) {
      r.punpckh = true;
    }
    // fit punpcklo
    fit = true;
    j2 = 0;
    for (std::size_t i = 0; i < lanesize; ++i) {
      if (lanepattern[i] >= 0 && lanepattern[i] != (int)j2) {
        fit = false;
      }
      if ((i & 1) != 0) {
        j2++;
      }
    }
    if (fit) {
      r.punpckl = true;
    }
    // fit pshufd
    if constexpr (elementsize >= 4) {
      u32 p = 0;
      for (std::size_t i = 0; i < lanesize; ++i) {
        if constexpr (lanesize == 4) {
          p |= (lanepattern[i] & 3) << 2 * i;
        } else {
          // lanesize = 2
          p |= ((lanepattern[i] & 1) * 10 + 4) << 4 * i;
        }
      }
      r.ipattern = p;
    }
  }
#if STADO_INSTRUCTION_SET >= STADO_AVX512F
  else {
    // not same pattern in all lanes
    if constexpr (nlanes > 1) {
      // Try if it fits big rotate
      for (std::size_t i = 0; i < size; ++i) {
        const int ix = a[i];
        if (ix >= 0) {
          // rotate count
          u32 rot2 = (ix + size - i) % size;
          if (rot == 999) {
            // save rotate count
            rot = rot2;
          } else if (rot != rot2) {
            // does not fit big rotate
            rot = 1000;
            break;
          }
        }
      }
      if (rot < size) {
        // fits big rotate
        r.rotate_big = true;
        r.rot_count = rot;
      }
    }
  }
#endif
  if (broadc < 999 && !r.rotate && !r.shright && !r.shleft && !r.rotate_big) {
    r.broadcast = true;
    r.rot_count = broadc; // fits broadcast
  }
  return r;
}

// compress_mask: returns a bit mask to use for compression instruction.
// It is presupposed that perm_flags indicates perm_compress.
// Additional zeroing is needed if perm_flags indicates perm_addz2
template<std::size_t tSize>
constexpr u64 compress_mask(const std::array<int, tSize>& indices) {
  int lasti = -1;
  int lastp = -1;
  u64 m = 0;
  for (std::size_t i = 0; i < tSize; ++i) {
    const int ix = indices[i]; // permutation index
    if (ix >= 0) {
      m |= u64{1} << ix; // mask for compression source
      for (std::size_t j = 1; j < i - lastp; ++j) {
        m |= u64{1} << (lasti + j); // dummy filling source
      }
      lastp = i;
      lasti = ix;
    }
  }
  return m;
}

// expand_mask: returns a bit mask to use for expansion instruction.
// It is presupposed that perm_flags indicates perm_expand.
// Additional zeroing is needed if perm_flags indicates perm_addz2
template<std::size_t tSize>
constexpr u64 expand_mask(const std::array<int, tSize>& indices) {
  int lasti = -1;
  int lastp = -1;
  u64 m = 0;
  for (std::size_t i = 0; i < tSize; ++i) {
    const int ix = indices[i]; // permutation index
    if (ix >= 0) {
      m |= u64{1} << i; // mask for expansion destination
      for (int j = 1; j < ix - lasti; ++j) {
        m |= u64{1} << (lastp + j); // dummy filling destination
      }
      lastp = i;
      lasti = ix;
    }
  }
  return m;
}

// perm16_flags: returns information about how to permute a vector of 16-bit integers
// Note: It is presupposed that perm_flags reports perm_same_pattern
// The return value is composed of these bits:
// 1:  data from low  64 bits to low  64 bits. pattern in bit 32-39
// 2:  data from high 64 bits to high 64 bits. pattern in bit 40-47
// 4:  data from high 64 bits to low  64 bits. pattern in bit 48-55
// 8:  data from low  64 bits to high 64 bits. pattern in bit 56-63
template<typename TVec>
constexpr u64 perm16_flags(const std::array<int, TVec::size>& indices) {
  // a is a reference to a constexpr array of permutation indexes
  // V is a vector class
  constexpr int size = TVec::size; // number of elements

  u64 retval = 0; // return value
  std::array<u32, 4> pat{0, 0, 0, 0}; // permute patterns
  constexpr std::size_t lanesize = 8; // elements per lane
  std::array<int, lanesize> lanepattern{0}; // pattern in each lane

  for (std::size_t i = 0; i < size; ++i) {
    const int ix = indices[i];
    std::size_t lane = i / lanesize; // current lane
    if (lane == 0) {
      lanepattern[i] = ix; // save pattern
    } else if (ix >= 0) { // not first lane
      u32 j = u32(i) - lane * lanesize; // index into lanepattern
      int jx = ix - lane * lanesize; // pattern within lane
      if (lanepattern[j] < 0) {
        lanepattern[j] = jx; // pattern not known from previous lane
      }
    }
  }
  // four patterns: low2low, high2high, high2low, low2high
  for (std::size_t i = 0; i < 4; ++i) {
    // loop through low pattern
    if (lanepattern[i] >= 0) {
      if (lanepattern[i] < 4) { // low2low
        retval |= 1U;
        pat[0] |= u32(lanepattern[i] & 3) << (2 * i);
      } else { // high2low
        retval |= 4U;
        pat[2] |= u32(lanepattern[i] & 3) << (2 * i);
      }
    }
    // loop through high pattern
    if (lanepattern[i + 4] >= 0) {
      if (lanepattern[i + 4] < 4) { // low2high
        retval |= 8U;
        pat[3] |= u32(lanepattern[i + 4] & 3) << (2 * i);
      } else { // high2high
        retval |= 2U;
        pat[1] |= u32(lanepattern[i + 4] & 3) << (2 * i);
      }
    }
  }
  // join return data
  for (std::size_t i = 0; i < 4; ++i) {
    retval |= (u64)pat[i] << (32 + i * 8);
  }
  return retval;
}

// pshufb_mask: return a broad byte mask for permutation within lanes
// for use with the pshufb instruction (_mm..._shuffle_epi8).
// The pshufb instruction provides fast permutation and zeroing,
// allowing different patterns in each lane but no crossing of lane boundaries
template<typename T, int tOppositeLanes = 0>
constexpr auto pshufb_mask(const std::array<int, T::size>& indices) {
  // Parameter a is a reference to a constexpr array of permutation indexes
  // V is a vector class
  // oppos = 1 for data from the opposite 128-bit lane in 256-bit vectors
  constexpr std::size_t size = T::size; // number of vector elements
  constexpr std::size_t elementsize = sizeof(T) / size; // size of each vector element
  constexpr std::size_t nlanes = sizeof(T) / 16; // number of 128 bit lanes in vector
  constexpr std::size_t elements_per_lane = size / nlanes; // number of vector elements per lane

  std::array<i8, sizeof(T)> u{0}; // list for returning
  std::size_t m = 0;
  std::size_t k = 0;

  for (std::size_t lane = 0; lane < nlanes; ++lane) { // loop through lanes
    for (std::size_t i = 0; i < elements_per_lane; ++i) { // loop through elements in lane
      // permutation index for element within lane
      i8 p = -1;
      int ix = indices[m];
      if (ix >= 0) {
        ix ^= tOppositeLanes * elements_per_lane; // flip bit if opposite lane
      }
      ix -= int(lane * elements_per_lane); // index relative to lane
      if (ix >= 0 && ix < (int)elements_per_lane) { // index points to desired lane
        p = (i8)ix * elementsize;
      }
      for (std::size_t j = 0; j < elementsize; ++j) { // loop through bytes in element
        u[k++] = p < 0 ? i8(-1) : i8(p + j); // store byte permutation index
      }
      ++m;
    }
  }
  return u; // return encapsulated array
}

// largeblock_perm: return indexes for replacing a permute or blend with
// a certain block size by a permute or blend with the double block size.
// Note: it is presupposed that perm_flags() indicates perm_largeblock
// It is required that additional zeroing is added if perm_flags() indicates perm_addz
template<std::size_t tSize>
constexpr std::array<int, tSize / 2> largeblock_perm(const std::array<int, tSize>& a) {
  // Parameter a is a reference to a constexpr array of permutation indexes
  std::array<int, tSize / 2> list{0}; // result indexes
  bool fit_addz = false; // additional zeroing needed at the lower block level

  // check if additional zeroing is needed at current block size
  for (std::size_t i = 0; i < tSize; i += 2) {
    int ix = a[i]; // even index
    int iy = a[i + 1]; // odd index
    if ((ix == -1 && iy >= 0) || (iy == -1 && ix >= 0)) {
      fit_addz = true;
    }
  }

  // loop through indexes
  for (std::size_t i = 0; i < tSize; i += 2) {
    int ix = a[i]; // even index
    int iy = a[i + 1]; // odd index
    int iz = 0; // combined index
    if (ix >= 0) {
      iz = ix / 2; // half index
    } else if (iy >= 0) {
      iz = iy / 2;
    } else {
      iz = ix | iy; // -1 or any. -1 takes precedence
      if (fit_addz) {
        iz = any; // any, because result will be zeroed later
      }
    }
    list[i / 2] = iz; // save to list
  }
  return list;
}

// blend_flags: returns information about how a blend function can be implemented
// The return value is composed of these flag bits:
// needs zeroing
constexpr u64 blend_zeroing = 1;
// all is zero or don't care
constexpr u64 blend_allzero = 2;
// fits blend with a larger block size (e.g permute i64x2 instead of i32x4)
constexpr u64 blend_largeblock = 4;
// additional zeroing needed after blend with larger block size or shift
constexpr u64 blend_addz = 8;
// has data from a
constexpr u64 blend_a = 0x10;
// has data from b
constexpr u64 blend_b = 0x20;
// permutation of a needed
constexpr u64 blend_perma = 0x40;
// permutation of b needed
constexpr u64 blend_permb = 0x80;
// permutation crossing 128-bit lanes
constexpr u64 blend_cross_lane = 0x100;
// same permute/blend pattern in all 128-bit lanes
constexpr u64 blend_same_pattern = 0x200;
// pattern fits punpckh(a,b)
constexpr u64 blend_punpckhab = 0x1000;
// pattern fits punpckh(b,a)
constexpr u64 blend_punpckhba = 0x2000;
// pattern fits punpckl(a,b)
constexpr u64 blend_punpcklab = 0x4000;
// pattern fits punpckl(b,a)
constexpr u64 blend_punpcklba = 0x8000;
// pattern fits palignr(a,b)
constexpr u64 blend_rotateab = 0x10000;
// pattern fits palignr(b,a)
constexpr u64 blend_rotateba = 0x20000;
// pattern fits shufps/shufpd(a,b)
constexpr u64 blend_shufab = 0x40000;
// pattern fits shufps/shufpd(b,a)
constexpr u64 blend_shufba = 0x80000;
// pattern fits rotation across lanes. count returned in bits blend_rotpattern
constexpr u64 blend_rotate_big = 0x100000;
// index out of range
constexpr u64 blend_outofrange = 0x10000000;
// pattern for shufps/shufpd is in bit blend_shufpattern to blend_shufpattern + 7
constexpr u64 blend_shufpattern = 32;
// pattern for palignr is in bit blend_rotpattern to blend_rotpattern + 7
constexpr u64 blend_rotpattern = 40;

template<typename TVec>
constexpr u64 blend_flags(const std::array<int, TVec::size>& a) {
  // number of elements
  constexpr std::size_t size = TVec::size;
  // number of 128-bit lanes
  constexpr std::size_t nlanes = sizeof(TVec) / 16;
  // elements per lane
  constexpr std::size_t lanesize = size / nlanes;
  // return value
  u64 r = blend_largeblock | blend_same_pattern | blend_allzero;
  // rotate left count
  u32 rot = 999;
  // pattern in each lane
  std::array<int, lanesize> lanepattern{0};
  if (lanesize == 2 && size <= 8) {
    // check if it fits shufpd
    r |= blend_shufab | blend_shufba;
  }

  for (std::size_t ii = 0; ii < size; ++ii) {
    // loop through indexes
    // index
    const int ix = a[ii];
    if (ix < 0) {
      if (ix == -1) {
        // set to zero
        r |= blend_zeroing;
      } else if (ix != any) {
        // illegal index
        r = blend_outofrange;
        break;
      }
    } else { // ix >= 0
      r &= ~blend_allzero;
      if (unsigned(ix) < size) {
        // data from a
        r |= blend_a;
        if (unsigned(ix) != ii) {
          // permutation of a
          r |= blend_perma;
        }
      } else if (unsigned(ix) < 2 * size) {
        r |= blend_b; // data from b
        if (unsigned(ix) != ii + size) {
          // permutation of b
          r |= blend_permb;
        }
      } else {
        // illegal index
        r = blend_outofrange;
        break;
      }
    }
    // check if pattern fits a larger block size:
    // even indexes must be even, odd indexes must fit the preceding even index + 1
    if ((ii & 1) == 0) {
      // even index
      if (ix >= 0 && (ix & 1)) {
        // not even. does not fit larger block size
        r &= ~blend_largeblock;
      }
      const int iy = a[ii + 1]; // next odd index
      if (iy >= 0 && (iy & 1) == 0) {
        // not odd. does not fit larger block size
        r &= ~blend_largeblock;
      }
      if (ix >= 0 && iy >= 0 && iy != ix + 1) {
        // does not fit preceding index + 1
        r &= ~blend_largeblock;
      }
      if (ix == -1 && iy >= 0) {
        // needs additional zeroing at current block size
        r |= blend_addz;
      }
      if (iy == -1 && ix >= 0) {
        // needs additional zeroing at current block size
        r |= blend_addz;
      }
    }
    // current lane
    const std::size_t lane = ii / lanesize;
    if (lane == 0) {
      // first lane, or no pattern yet
      lanepattern[ii] = ix; // save pattern
    }
    // check if crossing lanes
    if (ix >= 0) {
      // source lane
      u32 lanei = u32(ix & ~size) / lanesize;
      if (lanei != lane) {
        // crossing lane
        r |= blend_cross_lane;
      }
      if (lanesize == 2) {
        // check if it fits pshufd
        if (lanei != lane) {
          r &= ~(blend_shufab | blend_shufba);
        }
        if ((((ix & size) != 0) ^ ii) & 1) {
          r &= ~blend_shufab;
        } else {
          r &= ~blend_shufba;
        }
      }
    }
    // check if same pattern in all lanes
    if (lane != 0 && ix >= 0) {
      // not first lane
      // index into lanepattern
      int j = ii - int(lane * lanesize);
      // pattern within lane
      int jx = ix - int(lane * lanesize);
      if (jx < 0 || (jx & ~size) >= (int)lanesize) {
        // source is in another lane
        r &= ~blend_same_pattern;
      }
      if (lanepattern[j] < 0) {
        // pattern not known from previous lane
        lanepattern[j] = jx;
      } else {
        if (lanepattern[j] != jx) {
          // not same pattern
          r &= ~blend_same_pattern;
        }
      }
    }
  }
  if (!(r & blend_largeblock)) {
    // remove irrelevant flag
    r &= ~blend_addz;
  }
  if (r & blend_cross_lane) {
    // remove irrelevant flag
    r &= ~blend_same_pattern;
  }
  if (!(r & (blend_perma | blend_permb))) {
    // no permutation. more checks are superfluous
    return r;
  }
  if (r & blend_same_pattern) {
    // same pattern in all lanes. check if it fits unpack patterns
    r |= blend_punpckhab | blend_punpckhba | blend_punpcklab | blend_punpcklba;
    for (std::size_t iu = 0; iu < lanesize; ++iu) {
      // loop through lanepattern
      const int ix = lanepattern[iu];
      if (ix >= 0) {
        if (u32(ix) != iu / 2 + (iu & 1) * size) {
          r &= ~blend_punpcklab;
        }
        if (u32(ix) != iu / 2 + ((iu & 1) ^ 1) * size) {
          r &= ~blend_punpcklba;
        }
        if (u32(ix) != (iu + lanesize) / 2 + (iu & 1) * size) {
          r &= ~blend_punpckhab;
        }
        if (u32(ix) != (iu + lanesize) / 2 + ((iu & 1) ^ 1) * size) {
          r &= ~blend_punpckhba;
        }
      }
    }
#if STADO_INSTRUCTION_SET >= STADO_SSSE3
    // check if it fits palignr
    for (std::size_t iu = 0; iu < lanesize; ++iu) {
      const int ix = lanepattern[iu];
      if (ix >= 0) {
        u32 t = ix & ~size;
        if (ix & size) {
          t += lanesize;
        }
        u32 tb = (t + 2 * lanesize - iu) % (lanesize * 2);
        if (rot == 999) {
          rot = tb;
        } else { // check if fit
          if (rot != tb) {
            rot = 1000;
          }
        }
      }
    }
    if (rot < 999) {
      // firs palignr
      if (rot < lanesize) {
        r |= blend_rotateba;
      } else {
        r |= blend_rotateab;
      }
      const u32 elementsize = sizeof(TVec) / size;
      r |= u64((rot & (lanesize - 1)) * elementsize) << blend_rotpattern;
    }
#endif
    if (lanesize == 4) {
      // check if it fits shufps
      r |= blend_shufab | blend_shufba;
      for (std::size_t ii = 0; ii < 2; ++ii) {
        const int ix = lanepattern[ii];
        if (ix >= 0) {
          if (ix & size) {
            r &= ~blend_shufab;
          } else {
            r &= ~blend_shufba;
          }
        }
      }
      for (std::size_t ii = 2; ii < 4; ++ii) {
        const int ix = lanepattern[ii];
        if (ix >= 0) {
          if (ix & size) {
            r &= ~blend_shufba;
          } else {
            r &= ~blend_shufab;
          }
        }
      }
      if (r & (blend_shufab | blend_shufba)) {
        // fits shufps/shufpd
        u8 shufpattern = 0; // get pattern
        for (std::size_t iu = 0; iu < lanesize; ++iu) {
          shufpattern |= (lanepattern[iu] & 3) << iu * 2;
        }
        // return pattern
        r |= u64(shufpattern) << blend_shufpattern;
      }
    }
  } else if (nlanes > 1) {
    // not same pattern in all lanes
    // check if it fits big rotate
    rot = 999;
    for (std::size_t ii = 0; ii < size; ++ii) {
      const int ix = a[ii];
      if (ix >= 0) {
        // rotate count
        u32 rot2 = (ix + 2 * size - ii) % (2 * size);
        if (rot == 999) {
          // save rotate count
          rot = rot2;
        } else if (rot != rot2) {
          // does not fit big rotate
          rot = 1000;
          break;
        }
      }
    }
    if (rot < 2 * size) {
      // fits big rotate
      r |= blend_rotate_big | u64(rot) << blend_rotpattern;
    }
  }
  if (lanesize == 2 && (r & (blend_shufab | blend_shufba))) {
    // fits shufpd. Get pattern
    for (std::size_t ii = 0; ii < size; ++ii) {
      r |= u64(a[ii] & 1) << (blend_shufpattern + ii);
    }
  }
  return r;
}

// blend_perm_indexes: return an Indexlist for implementing a blend function as
// two permutations. N = vector size.
// dozero = 0: let unused elements be don't care. The two permutation results must be blended
// dozero = 1: zero unused elements in each permuation. The two permutation results can be OR'ed
// dozero = 2: indexes that are -1 or any are preserved
template<std::size_t tSize, int tDoZero>
constexpr std::array<int, 2 * tSize> blend_perm_indexes(const std::array<int, tSize>& indices) {
  std::array<int, 2 * tSize> arr{0};
  // value to use for unused entries
  int u = tDoZero ? -1 : any;

  for (std::size_t j = 0; j < tSize; ++j) {
    // loop through indexes
    // current index
    int ix = indices[j];
    if (ix < 0) {
      // zero or don't care
      if (tDoZero == 2) {
        arr[j] = ix;
        arr[j + tSize] = ix;
      } else {
        arr[j] = u;
        arr[j + tSize] = u;
      }
    } else if (unsigned(ix) < tSize) {
      // value from a
      arr[j] = ix;
      arr[j + tSize] = u;
    } else {
      // value from b
      arr[j] = u;
      arr[j + tSize] = ix - tSize;
    }
  }
  return arr;
}

// largeblock_indexes: return indexes for replacing a permute or blend with a
// certain block size by a permute or blend with the double block size.
// Note: it is presupposed that perm_flags or blend_flags indicates _largeblock
// It is required that additional zeroing is added if perm_flags or blend_flags
// indicates _addz
template<std::size_t tSize>
constexpr std::array<int, tSize / 2> largeblock_indexes(const std::array<int, tSize>& indices) {
  std::array<int, tSize / 2> arr{0};
  // additional zeroing needed at the lower block level
  bool fit_addz = false;

  for (std::size_t i = 0; i < tSize; i += 2) {
    // even index
    const int ix = indices[i];
    // odd index
    const int iy = indices[i + 1];
    // combined index
    int iz = 0;
    if (ix >= 0) {
      // half index
      iz = ix / 2;
    } else if (iy >= 0) {
      // half index
      iz = iy / 2;
    } else {
      // -1 or any. -1 takes precedence
      iz = ix | iy;
    }
    // save to list
    arr[i / 2] = iz;
    // check if additional zeroing is needed at current block size
    if ((ix == -1 && iy >= 0) || (iy == -1 && ix >= 0)) {
      fit_addz = true;
    }
  }
  // replace -1 by any if fit_addz
  if (fit_addz) {
    for (std::size_t i = 0; i < tSize / 2; ++i) {
      if (arr[i] < 0) {
        arr[i] = any;
      }
    }
  }
  return arr;
}
} // namespace stado

#endif // INCLUDE_STADO_INSTRUCTION_SET_HPP
