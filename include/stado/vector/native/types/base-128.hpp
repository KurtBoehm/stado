#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_BASE_128_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_BASE_128_HPP

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"

namespace stado {
/*****************************************************************************
 *
 *          selectb function
 *
 *****************************************************************************/
// Select between two sources, byte by byte, using broad boolean vector s.
// Used in various functions and operators
// Corresponds to this pseudocode:
// for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFF (true). No other values are allowed.
// The implementation depends on the instruction set:
// If SSE4.1 is supported then only bit 7 in each byte of s is checked,
// otherwise all bits in s are used.
inline __m128i selectb(const __m128i s, const __m128i a, const __m128i b) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_2
  return _mm_blendv_epi8(b, a, s);
#else
  return _mm_or_si128(_mm_and_si128(s, a), _mm_andnot_si128(s, b));
#endif
}

struct si128 {
  using Register = __m128i;

  // Default constructor:
  si128() = default;
  // Constructor to convert from type __m128i used in intrinsics:
  si128(const __m128i x) : xmm(x) {}
  // Assignment operator to convert from type __m128i used in intrinsics:
  si128& operator=(const __m128i x) {
    xmm = x;
    return *this;
  }
  // Type cast operator to convert to __m128i used in intrinsics
  operator __m128i() const {
    return xmm;
  }
  // Member function to load from array (unaligned)
  si128& load(const void* p) {
    xmm = _mm_loadu_si128((const __m128i*)p);
    return *this;
  }
  // Member function to load from array, aligned by 16
  // "load_a" is faster than "load" on older Intel processors (Pentium 4, Pentium M, Core 1,
  // Merom, Wolfdale, and Atom), but not on other processors from Intel, AMD or VIA.
  // You may use load_a instead of load if you are certain that p points to an address
  // divisible by 16.
  void load_a(const void* p) {
    xmm = _mm_load_si128((const __m128i*)p);
  }
  // Member function to store into array (unaligned)
  void store(void* p) const {
    _mm_storeu_si128((__m128i*)p, xmm);
  }
  // Member function storing into array, aligned by 16
  // "store_a" is faster than "store" on older Intel processors (Pentium 4, Pentium M, Core 1,
  // Merom, Wolfdale, and Atom), but not on other processors from Intel, AMD or VIA.
  // You may use store_a instead of store if you are certain that p points to an address
  // divisible by 16.
  void store_a(void* p) const {
    _mm_store_si128((__m128i*)p, xmm);
  }
  // Member function storing to aligned uncached memory (non-temporal store).
  // This may be more efficient than store_a when storing large blocks of memory if it
  // is unlikely that the data will stay in the cache until it is read again.
  // Note: Will generate runtime error if p is not aligned by 16
  void store_nt(void* p) const {
    _mm_stream_si128((__m128i*)p, xmm);
  }

protected:
  __m128i xmm;
};

// Define operators for this class

// vector operator & : bitwise and
static inline si128 operator&(const si128 a, const si128 b) {
  return _mm_and_si128(a, b);
}
static inline si128 operator&&(const si128 a, const si128 b) {
  return a & b;
}

// vector operator | : bitwise or
static inline si128 operator|(const si128 a, const si128 b) {
  return _mm_or_si128(a, b);
}
static inline si128 operator||(const si128 a, const si128 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
static inline si128 operator^(const si128 a, const si128 b) {
  return _mm_xor_si128(a, b);
}

// vector operator ~ : bitwise not
static inline si128 operator~(const si128 a) {
  return _mm_xor_si128(a, _mm_set1_epi32(-1));
}

// vector operator &= : bitwise and
static inline si128& operator&=(si128& a, const si128 b) {
  a = a & b;
  return a;
}

// vector operator |= : bitwise or
static inline si128& operator|=(si128& a, const si128 b) {
  a = a | b;
  return a;
}

// vector operator ^= : bitwise xor
static inline si128& operator^=(si128& a, const si128 b) {
  a = a ^ b;
  return a;
}

// function andnot: a & ~ b
static inline si128 andnot(const si128 a, const si128 b) {
  return _mm_andnot_si128(b, a);
}

/*****************************************************************************
 *
 *          Horizontal Boolean functions
 *
 *****************************************************************************/

static inline bool horizontal_and(const si128 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_2
  // use ptest
  return _mm_testc_si128(a, _mm_set1_epi32(-1)) != 0;
#else
  __m128i t1 = _mm_unpackhi_epi64(a, a); // get 64 bits down
  __m128i t2 = _mm_and_si128(a, t1); // and 64 bits
  i64 t5 = _mm_cvtsi128_si64(t2); // transfer 64 bits to integer
  return t5 == i64(-1);
#endif
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or(const si128 a) {
#if STADO_INSTRUCTION_SET >= STADO_SSE4_2
  // use ptest
  return _mm_testz_si128(a, a) == 0;
#else
  __m128i t1 = _mm_unpackhi_epi64(a, a); // get 64 bits down
  __m128i t2 = _mm_or_si128(a, t1); // and 64 bits
  i64 t5 = _mm_cvtsi128_si64(t2); // transfer 64 bits to integer
  return t5 != i64(0);
#endif
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_BASE_128_HPP
