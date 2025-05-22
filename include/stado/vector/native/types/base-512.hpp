#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_BASE_512_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_BASE_512_HPP

#include "stado/vector/native/types/base-256.hpp"

namespace stado {
struct si512 {
  using Register = __m512i;

  // Default constructor:
  si512() = default;
  // Constructor to build from two si256:
  si512(const si256 a0, const si256 a1)
      : zmm(_mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1)) {}
  // Constructor to convert from type __m512i used in intrinsics:
  si512(const __m512i x) : zmm(x) {}
  // Assignment operator to convert from type __m512i used in intrinsics:
  si512& operator=(const __m512i x) {
    zmm = x;
    return *this;
  }
  // Type cast operator to convert to __m512i used in intrinsics
  operator __m512i() const {
    return zmm;
  }
  // Member function to load from array (unaligned)
  si512& load(const void* p) {
    zmm = _mm512_loadu_si512(p);
    return *this;
  }
  // Member function to load from array, aligned by 64
  // You may use load_a instead of load if you are certain that p points to an address
  // divisible by 64, but there is hardly any speed advantage of load_a on modern processors
  si512& load_a(const void* p) {
    zmm = _mm512_load_si512(p);
    return *this;
  }
  // Member function to store into array (unaligned)
  void store(void* p) const {
    _mm512_storeu_si512(p, zmm);
  }
  // Member function to store into array, aligned by 64
  // You may use store_a instead of store if you are certain that p points to an address
  // divisible by 64, but there is hardly any speed advantage of store_a on modern processors
  void store_a(void* p) const {
    _mm512_store_si512(p, zmm);
  }
  // Member function storing to aligned uncached memory (non-temporal store).
  // This may be more efficient than store_a when storing large blocks of memory if it
  // is unlikely that the data will stay in the cache until it is read again.
  // Note: Will generate runtime error if p is not aligned by 64
  void store_nt(void* p) const {
    _mm512_stream_si512(reinterpret_cast<__m512i*>(p), zmm);
  }
  // Member functions to split into two si256:
  [[nodiscard]] si256 get_low() const {
    return _mm512_castsi512_si256(zmm);
  }
  [[nodiscard]] si256 get_high() const {
    return _mm512_extracti64x4_epi64(zmm, 1);
  }

protected:
  __m512i zmm;
};

// vector operator & : bitwise and
static inline si512 operator&(const si512 a, const si512 b) {
  return _mm512_and_epi32(a, b);
}
static inline si512 operator&&(const si512 a, const si512 b) {
  return a & b;
}

// vector operator | : bitwise or
static inline si512 operator|(const si512 a, const si512 b) {
  return _mm512_or_epi32(a, b);
}
static inline si512 operator||(const si512 a, const si512 b) {
  return a | b;
}

// vector operator ^ : bitwise xor
static inline si512 operator^(const si512 a, const si512 b) {
  return _mm512_xor_epi32(a, b);
}

// vector operator ~ : bitwise not
static inline si512 operator~(const si512 a) {
  return _mm512_xor_epi32(a, _mm512_set1_epi32(-1));
}

// vector operator &= : bitwise and
static inline si512& operator&=(si512& a, const si512 b) {
  a = a & b;
  return a;
}

// vector operator |= : bitwise or
static inline si512& operator|=(si512& a, const si512 b) {
  a = a | b;
  return a;
}

// vector operator ^= : bitwise xor
static inline si512& operator^=(si512& a, const si512 b) {
  a = a ^ b;
  return a;
}

// function andnot: a & ~ b
static inline si512 andnot(const si512 a, const si512 b) {
  return _mm512_andnot_epi32(b, a);
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_BASE_512_HPP
