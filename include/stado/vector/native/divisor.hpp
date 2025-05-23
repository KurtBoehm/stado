#ifndef INCLUDE_STADO_VECTOR_NATIVE_DIVISOR_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_DIVISOR_HPP

#include <cstdlib>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"

namespace stado {
// encapsulate parameters for fast division on vector of 4 32-bit signed integers
class DivisorI32 {
public:
  // Default constructor
  DivisorI32() = default;
  // Constructor with divisor
  DivisorI32(i32 d) {
    set(d);
  }
  // Constructor with precalculated multiplier, shift and sign
  DivisorI32(i32 m, i32 s1, i32 sgn)
      : multiplier_(_mm_set1_epi32(m)), shift1_(_mm_cvtsi32_si128(s1)), sign_(_mm_set1_epi32(sgn)) {
  }
  // Set or change divisor, calculate parameters
  void set(i32 d) {
    const i32 d1 = std::abs(d);
    i32 sh;
    i32 m;
    if (d1 > 1) {
      // shift count = ceil(log2(d1))-1 = (bit_scan_reverse(d1-1)+1)-1
      sh = i32(bit_scan_reverse(u32(d1 - 1)));
      // calculate multiplier
      m = i32((i64(1) << (32 + sh)) / d1 - ((i64(1) << 32) - 1));
    } else {
      // for d1 = 1
      m = 1;
      sh = 0;
      if (d == 0) {
        // provoke error here if d = 0
        m /= d;
      }
      if (u32(d) == 0x80000000U) {
        // fix overflow for this special case
        m = i32(0x80000001);
        sh = 30;
      }
    }
    // broadcast multiplier
    multiplier_ = _mm_set1_epi32(m);
    // shift count
    shift1_ = _mm_cvtsi32_si128(sh);
    // sign of divisor
    if (d < 0) {
      sign_ = _mm_set1_epi32(-1);
    } else {
      sign_ = _mm_set1_epi32(0);
    }
  }
  // get multiplier
  __m128i getm() const {
    return multiplier_;
  }
  // get shift count
  __m128i gets1() const {
    return shift1_;
  }
  // get sign of divisor
  __m128i getsign() const {
    return sign_;
  }

private:
  // multiplier used in fast division
  __m128i multiplier_;
  // shift count used in fast division
  __m128i shift1_;
  // sign of divisor
  __m128i sign_;
};

// encapsulate parameters for fast division on vector of 4 32-bit unsigned integers
class DivisorU32 {
public:
  // Default constructor
  DivisorU32() = default;
  // Constructor with divisor
  DivisorU32(u32 d) {
    set(d);
  }
  // Constructor with precalculated multiplier and shifts
  DivisorU32(u32 m, i32 s1, i32 s2)
      : multiplier_(_mm_set1_epi32(i32(m))), shift1_(_mm_setr_epi32(s1, 0, 0, 0)),
        shift2_(_mm_setr_epi32(s2, 0, 0, 0)) {}
  // Set or change divisor, calculate parameters
  void set(u32 d) {
    u32 sh1;
    u32 sh2;
    u32 m;
    switch (d) {
    // provoke error for d = 0
    case 0: m = sh1 = sh2 = 1 / d; break;
    // parameters for d = 1
    case 1:
      m = 1;
      sh1 = sh2 = 0;
      break;
    // parameters for d = 2
    case 2:
      m = 1;
      sh1 = 1;
      sh2 = 0;
      break;
    // general case for d > 2
    default:
      // ceil(log2(d))
      u32 l = bit_scan_reverse(d - 1) + 1;
      // 2^L, overflow to 0 if L = 32
      u32 l2 = u32(l < 32 ? 1 << l : 0);
      // multiplier
      m = 1 + u32((u64(l2 - d) << 32) / d);
      sh1 = 1;
      sh2 = l - 1; // shift counts
    }
    multiplier_ = _mm_set1_epi32(i32(m));
    shift1_ = _mm_setr_epi32(i32(sh1), 0, 0, 0);
    shift2_ = _mm_setr_epi32(i32(sh2), 0, 0, 0);
  }
  // get multiplier
  __m128i getm() const {
    return multiplier_;
  }
  // get shift count 1
  __m128i gets1() const {
    return shift1_;
  }
  // get shift count 2
  __m128i gets2() const {
    return shift2_;
  }

private:
  // multiplier used in fast division
  __m128i multiplier_;
  // shift count 1 used in fast division
  __m128i shift1_;
  // shift count 2 used in fast division
  __m128i shift2_;
};

// encapsulate parameters for fast division on vector of 8 16-bit signed integers
class DivisorI16 {
public:
  // Default constructor
  DivisorI16() = default;
  // Constructor with divisor
  DivisorI16(u16 d) {
    set(d);
  }
  // Constructor with precalculated multiplier, shift and sign
  DivisorI16(u16 m, i32 s1, i32 sgn)
      : multiplier_(_mm_set1_epi16(i16(m))), shift1_(_mm_setr_epi32(s1, 0, 0, 0)),
        sign_(_mm_set1_epi32(sgn)) {}
  void set(u16 d) { // Set or change divisor, calculate parameters
    const i32 d1 = d;
    i32 sh;
    i32 m;
    if (d1 > 1) {
      // shift count = ceil(log2(d1))-1 = (bit_scan_reverse(d1-1)+1)-1
      sh = i32(bit_scan_reverse(u32(d1 - 1)));
      // calculate multiplier
      m = ((i32(1) << (16 + sh)) / d1 - ((i32(1) << 16) - 1));
    } else {
      // for d1 = 1
      m = 1;
      sh = 0;
      if (d == 0) {
        // provoke error here if d = 0
        m /= d;
      }
      if (d == 0x8000U) {
        // fix overflow for this special case
        m = 0x8001;
        sh = 14;
      }
    }
    // broadcast multiplier
    multiplier_ = _mm_set1_epi16(i16(m));
    // shift count
    shift1_ = _mm_setr_epi32(sh, 0, 0, 0);
    // sign of divisor
    sign_ = _mm_set1_epi32(d < 0 ? -1 : 0);
  }
  // get multiplier
  __m128i getm() const {
    return multiplier_;
  }
  // get shift count
  __m128i gets1() const {
    return shift1_;
  }
  // get sign of divisor
  __m128i getsign() const {
    return sign_;
  }

private:
  // multiplier used in fast division
  __m128i multiplier_;
  // shift count used in fast division
  __m128i shift1_;
  // sign of divisor
  __m128i sign_;
};

// encapsulate parameters for fast division on vector of 8 16-bit unsigned integers
class DivisorU16 {
public:
  // Default constructor
  DivisorU16() = default;
  // Constructor with divisor
  DivisorU16(u16 d) {
    set(d);
  }
  // Constructor with precalculated multiplier and shifts
  DivisorU16(u16 m, i32 s1, i32 s2)
      : multiplier_(_mm_set1_epi16(i16(m))), shift1_(_mm_setr_epi32(s1, 0, 0, 0)),
        shift2_(_mm_setr_epi32(s2, 0, 0, 0)) {}
  // Set or change divisor, calculate parameters
  void set(u16 d) {
    u16 sh1;
    u16 sh2;
    u16 m;
    switch (d) {
    case 0:
      // provoke error for d = 0
      m = sh1 = sh2 = 1U / d;
      break;
    case 1:
      // parameters for d = 1
      m = 1;
      sh1 = sh2 = 0;
      break;
    case 2:
      // parameters for d = 2
      m = 1;
      sh1 = 1;
      sh2 = 0;
      break;
    default:
      // general case for d > 2
      // ceil(log2(d))
      u16 l = u16(bit_scan_reverse(d - 1U)) + 1U;
      // 2^L, overflow to 0 if L = 16
      u16 l2 = u16(1 << l);
      // multiplier
      m = 1U + u16((u32(l2 - d) << 16) / d);
      // shift counts
      sh1 = 1;
      sh2 = l - 1;
    }
    multiplier_ = _mm_set1_epi16(i16(m));
    shift1_ = _mm_setr_epi32(i32(sh1), 0, 0, 0);
    shift2_ = _mm_setr_epi32(i32(sh2), 0, 0, 0);
  }
  // get multiplier
  __m128i getm() const {
    return multiplier_;
  }
  // get shift count 1
  __m128i gets1() const {
    return shift1_;
  }
  // get shift count 2
  __m128i gets2() const {
    return shift2_;
  }

private:
  // multiplier used in fast division
  __m128i multiplier_;
  // shift count 1 used in fast division
  __m128i shift1_;
  // shift count 2 used in fast division
  __m128i shift2_;
};
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_DIVISOR_HPP
