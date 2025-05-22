#ifndef INCLUDE_STADO_MASK_BROAD_MASK_64_4_HPP
#define INCLUDE_STADO_MASK_BROAD_MASK_64_4_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad/base.hpp"
#include "stado/mask/broad/mask-64-2.hpp"
#include "stado/vector/native/types/base-256.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX
namespace stado {
template<>
struct BroadMask<64, 4> {
  using Element = bool;
  using Register = __m256d;
  using Half = b64x2;
  static constexpr std::size_t size = 4;

  // Default constructor:
  BroadMask() = default;
// Constructor to build from all elements:
#if STADO_INSTRUCTION_SET >= STADO_AVX2 // AVX2
  BroadMask(bool b0, bool b1, bool b2, bool b3)
      : ymm(_mm256_castsi256_pd(_mm256_setr_epi64x(-i64(b0), -i64(b1), -i64(b2), -i64(b3)))) {}
#else
  BroadMask(bool b0, bool b1, bool b2, bool b3) {
    __m128 blo = _mm_castsi128_ps(_mm_setr_epi32(-(int)b0, -(int)b0, -(int)b1, -(int)b1));
    __m128 bhi = _mm_castsi128_ps(_mm_setr_epi32(-(int)b2, -(int)b2, -(int)b3, -(int)b3));
    ymm = _mm256_castps_pd(_mm256_setr_m128(blo, bhi));
  }
#endif
  // Constructor to build from two Vec2db:
  BroadMask(const Half a0, const Half a1) : ymm(_mm256_setr_m128d(a0, a1)) {}
  // Constructor to convert from type __m256d used in intrinsics:
  BroadMask(const __m256d x) : ymm(x) {}
#if STADO_INSTRUCTION_SET >= STADO_AVX2 // AVX2
  BroadMask(__m256i const x) : ymm(_mm256_castsi256_pd(x)) {}
#endif // AVX2
  // Assignment operator to convert from type __m256d used in intrinsics:
  BroadMask& operator=(const __m256d x) {
    ymm = x;
    return *this;
  }
// Constructor to broadcast the same value into all elements:
#if STADO_INSTRUCTION_SET >= STADO_AVX2 // AVX2
  BroadMask(bool b) : ymm(_mm256_castsi256_pd(_mm256_set1_epi64x(-i64(b)))) {}
#else
  BroadMask(bool b) {
    __m128 b1 = _mm_castsi128_ps(_mm_set1_epi32(-(int)b));
    ymm = _mm256_castps_pd(_mm256_setr_m128(b1, b1));
  }
#endif
  // Assignment operator to broadcast scalar value:
  BroadMask& operator=(bool b) {
    ymm = _mm256_castsi256_pd(_mm256_set1_epi32(-i32(b)));
    return *this;
  }
  // Type cast operator to convert to __m256d used in intrinsics
  operator __m256d() const {
    return ymm;
  }
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  operator __m256i() const {
    return _mm256_castpd_si256(ymm);
  }
  // Member function to change a bitfield to a boolean vector
  BroadMask& load_bits(u8 a) {
    __m256i b1 = _mm256_set1_epi32(i32(a)); // broadcast a
    __m256i m2 = _mm256_setr_epi32(1, 0, 2, 0, 4, 0, 8, 0);
    __m256i d1 = _mm256_and_si256(b1, m2); // isolate one bit in each dword
    __m256i b = _mm256_cmpgt_epi64(
      d1, _mm256_setzero_si256()); // we can use signed compare here because no value is negative
    ymm = _mm256_castsi256_pd(b);
    return *this;
  }
#else
  // Member function to change a bitfield to a boolean vector
  // AVX version. Cannot use float instructions if subnormals are disabled
  BroadMask& load_bits(u8 a) {
    Half a0 = Half().load_bits(a);
    Half a1 = Half().load_bits(u8(a >> 2U));
    *this = BroadMask(a0, a1);
    return *this;
  }
#endif // AVX2
  // Member function to change a single element in vector
  BroadMask& insert(std::size_t index, bool value) {
    static constexpr std::array<i32, 16> maskl{0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0};
    // mask with FFFFFFFFFFFFFFFF at index position
    __m256d mask =
      _mm256_loadu_pd(reinterpret_cast<const f64*>(maskl.data() + 8 - (index & 3U) * 2));
    if (value) {
      ymm = _mm256_or_pd(ymm, mask);
    } else {
      ymm = _mm256_andnot_pd(mask, ymm);
    }
    return *this;
  }
  // Member function extract a single element from vector
  bool extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    __m256i x = _mm256_maskz_compress_epi64(__mmask8(1U << index), _mm256_castpd_si256(ymm));
    return _mm_cvtsi128_si64(_mm256_castsi256_si128(x)) != 0;
#else
    i64 x[4];
    _mm256_storeu_pd(reinterpret_cast<f64*>(x), ymm);
    return x[index & 3] != 0;
#endif
  }
  // Extract a single element. Operator [] can only read an element, not write.
  bool operator[](int index) const {
    return extract(index);
  }
  // Member functions to split into two Vec4fb:
  Half get_low() const {
    return _mm256_castpd256_pd128(ymm);
  }
  Half get_high() const {
    return _mm256_extractf128_pd(ymm, 1);
  }

  // Prevent constructing from int, etc.
  BroadMask(int b) = delete;
  BroadMask& operator=(int x) = delete;

private:
  __m256d ymm;
};
using b64x4 = BroadMask<64, 4>;

// vector operator & : bitwise and
static inline b64x4 operator&(const b64x4 a, const b64x4 b) {
  return _mm256_and_pd(a, b);
}
static inline b64x4 operator&&(const b64x4 a, const b64x4 b) {
  return a & b;
}

// vector operator &= : bitwise and
static inline b64x4& operator&=(b64x4& a, const b64x4 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline b64x4 operator|(const b64x4 a, const b64x4 b) {
  return _mm256_or_pd(a, b);
}
static inline b64x4 operator||(const b64x4 a, const b64x4 b) {
  return a | b;
}

// vector operator |= : bitwise or
static inline b64x4& operator|=(b64x4& a, const b64x4 b) {
  a = a | b;
  return a;
}

// vector operator ~ : bitwise not
static inline b64x4 operator~(const b64x4 a) {
  const __m256i mask =
    _mm256_setr_epi32(i32(0xFFFFFFFFU), i32(0xFFFFFFFFU), i32(0xFFFFFFFFU), i32(0xFFFFFFFFU),
                      i32(0xFFFFFFFFU), i32(0xFFFFFFFFU), i32(0xFFFFFFFFU), i32(0xFFFFFFFFU));
  return _mm256_xor_pd(a, _mm256_castps_pd(mask));
}

// vector operator ^ : bitwise xor
static inline b64x4 operator^(const b64x4 a, const b64x4 b) {
  return _mm256_xor_pd(a, b);
}

// vector operator == : xnor
static inline b64x4 operator==(const b64x4 a, const b64x4 b) {
  return b64x4(a ^ b64x4(~b));
}

// vector operator != : xor
static inline b64x4 operator!=(const b64x4 a, const b64x4 b) {
  return _mm256_xor_pd(a, b);
}

// vector operator ^= : bitwise xor
static inline b64x4& operator^=(b64x4& a, const b64x4 b) {
  a = a ^ b;
  return a;
}

// vector operator ! : logical not
static inline b64x4 operator!(const b64x4 a) {
  return ~a;
}

// Functions for Vec8fb

// andnot: a & ~ b
static inline b64x4 andnot(const b64x4 a, const b64x4 b) {
  return _mm256_andnot_pd(b, a);
}

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and(const b64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available, AVX2
  return horizontal_and(si256(_mm256_castpd_si256(a)));
#else // split into 128 bit vectors
  return horizontal_and(a.get_low() & a.get_high());
#endif
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or(const b64x4 a) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2 // 256 bit integer vectors are available, AVX2
  return horizontal_or(si256(_mm256_castpd_si256(a)));
#else // split into 128 bit vectors
  return horizontal_or(a.get_low() | a.get_high());
#endif
}

// to_bits: convert boolean vector to integer bitfield
static inline u8 to_bits(const b64x4 x) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2 // AVX2
  auto a = u32(_mm256_movemask_epi8(x));
  return ((a & 1) | ((a >> 7) & 2)) | (((a >> 14) & 4) | ((a >> 21) & 8));
#else
  return u8(to_bits(x.get_low()) | to_bits(x.get_high()) << 2);
#endif
}
} // namespace stado

#endif // AVX

#endif // INCLUDE_STADO_MASK_BROAD_MASK_64_4_HPP
