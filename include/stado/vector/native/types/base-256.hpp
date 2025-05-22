#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_BASE_256_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_BASE_256_HPP

#include "stado/instruction-set.hpp"
#include "stado/vector/native/types/base-128.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
struct si256 {
  using register_type = __m256i;

  // Default constructor:
  si256() = default;
  // Constructor to build from two si128:
  si256(const si128 a0, const si128 a1) : ymm(_mm256_set_m128i(a1, a0)) {}
  // Constructor to convert from type __m256i used in intrinsics:
  si256(const __m256i x) : ymm(x) {}
  // Assignment operator to convert from type __m256i used in intrinsics:
  si256& operator=(const __m256i x) {
    ymm = x;
    return *this;
  }
  // Type cast operator to convert to __m256i used in intrinsics
  operator __m256i() const {
    return ymm;
  }
  // Member function to load from array (unaligned)
  si256& load(const void* p) {
    ymm = _mm256_loadu_si256((const __m256i*)p);
    return *this;
  }
  // Member function to load from array, aligned by 32
  // You may use load_a instead of load if you are certain that p points to an address
  // divisible by 32, but there is hardly any speed advantage of load_a on modern processors
  si256& load_a(const void* p) {
    ymm = _mm256_load_si256((const __m256i*)p);
    return *this;
  }
  // Member function to store into array (unaligned)
  void store(void* p) const {
    _mm256_storeu_si256((__m256i*)p, ymm);
  }
  // Member function storing into array, aligned by 32
  // You may use store_a instead of store if you are certain that p points to an address
  // divisible by 32, but there is hardly any speed advantage of load_a on modern processors
  void store_a(void* p) const {
    _mm256_store_si256((__m256i*)p, ymm);
  }
  // Member function storing to aligned uncached memory (non-temporal store).
  // This may be more efficient than store_a when storing large blocks of memory if it
  // is unlikely that the data will stay in the cache until it is read again.
  // Note: Will generate runtime error if p is not aligned by 32
  void store_nt(void* p) const {
    _mm256_stream_si256((__m256i*)p, ymm);
  }
  // Member functions to split into two si128:
  [[nodiscard]] si128 get_low() const {
    return _mm256_castsi256_si128(ymm);
  }
  [[nodiscard]] si128 get_high() const {
    return _mm256_extractf128_si256(ymm, 1);
  }

protected:
  __m256i ymm;
};

// Define operators and functions for this class

// vector operator & : bitwise and
static inline si256 operator&(const si256 a, const si256 b) {
  return _mm256_and_si256(a, b);
}
static inline si256 operator&&(const si256 a, const si256 b) {
  return a & b;
}

// vector operator | : bitwise or
static inline si256 operator|(const si256 a, const si256 b) {
  return _mm256_or_si256(a, b);
}
static inline si256 operator||(const si256 a, const si256 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
static inline si256 operator^(const si256 a, const si256 b) {
  return _mm256_xor_si256(a, b);
}

// vector operator ~ : bitwise not
static inline si256 operator~(const si256 a) {
  return _mm256_xor_si256(a, _mm256_set1_epi32(-1));
}

// vector operator &= : bitwise and
static inline si256& operator&=(si256& a, const si256 b) {
  a = a & b;
  return a;
}

// vector operator |= : bitwise or
static inline si256& operator|=(si256& a, const si256 b) {
  a = a | b;
  return a;
}

// vector operator ^= : bitwise xor
static inline si256& operator^=(si256& a, const si256 b) {
  a = a ^ b;
  return a;
}

// function andnot: a & ~ b
static inline si256 andnot(const si256 a, const si256 b) {
  return _mm256_andnot_si256(b, a);
}

// Select between two sources, byte by byte. Used in various functions and operators
// Corresponds to this pseudocode:
// for (int i = 0; i < 32; i++) result[i] = s[i] ? a[i] : b[i];
// Each byte in s must be either 0 (false) or 0xFF (true). No other values are allowed.
// Only bit 7 in each byte of s is checked,
static inline __m256i selectb(const __m256i s, const __m256i a, const __m256i b) {
  return _mm256_blendv_epi8(b, a, s);
}

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and(const si256 a) {
  return _mm256_testc_si256(a, _mm256_set1_epi32(-1)) != 0;
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or(const si256 a) {
  return _mm256_testz_si256(a, a) == 0;
}
} // namespace stado
#endif // AVX2

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_BASE_256_HPP
