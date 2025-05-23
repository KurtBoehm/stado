#ifndef INCLUDE_STADO_MASK_BROAD_32X08_HPP
#define INCLUDE_STADO_MASK_BROAD_32X08_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/mask/broad/32x04.hpp"
#include "stado/mask/broad/base.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX
namespace stado {
template<>
struct BroadMask<32, 8> {
  using Value = bool;
  using Register = __m256;
  using Half = b32x4;
  static constexpr std::size_t size = 8;
  static constexpr std::size_t element_bits = 32;

  // Default constructor:
  BroadMask() = default;
// Constructor to build from all elements:
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  BroadMask(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7)
      : ymm(_mm256_castsi256_ps(_mm256_setr_epi32(-i32(b0), -i32(b1), -i32(b2), -i32(b3), -i32(b4),
                                                  -i32(b5), -i32(b6), -i32(b7)))) {}
#else
  BroadMask(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7) {
    __m128 blo = _mm_castsi128_ps(_mm_setr_epi32(-i32(b0), -i32(b1), -i32(b2), -i32(b3)));
    __m128 bhi = _mm_castsi128_ps(_mm_setr_epi32(-i32(b4), -i32(b5), -i32(b6), -i32(b7)));
    ymm = _mm256_set_m128(bhi, blo);
  }
#endif
  // Constructor to build from two Vec4fb:
  BroadMask(const Half a0, const Half a1) : ymm(_mm256_set_m128(a1, a0)) {}
  // Constructor to convert from type __m256 used in intrinsics:
  BroadMask(const __m256 x) : ymm(x) {}
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  BroadMask(__m256i const x) : ymm(_mm256_castsi256_ps(x)) {}
#endif
  // Assignment operator to convert from type __m256 used in intrinsics:
  BroadMask& operator=(const __m256 x) {
    ymm = x;
    return *this;
  }
// Constructor to broadcast the same value into all elements:
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  BroadMask(bool b) : ymm(_mm256_castsi256_ps(_mm256_set1_epi32(-i32(b)))) {}
#else
  BroadMask(bool b) {
    __m128 b1 = _mm_castsi128_ps(_mm_set1_epi32(-i32(b)));
    ymm = _mm256_set_m128(b1, b1);
  }
#endif
  // Assignment operator to broadcast scalar value:
  BroadMask& operator=(bool b) {
    *this = BroadMask(b);
    return *this;
  }
  // Type cast operator to convert to __m256 used in intrinsics
  operator __m256() const {
    return ymm;
  }
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  operator __m256i() const {
    return _mm256_castps_si256(ymm);
  }
  // Member function to change a bitfield to a boolean vector
  BroadMask& load_bits(u8 a) {
    __m256i b1 = _mm256_set1_epi32(i32(a)); // broadcast a
    __m256i m2 = _mm256_setr_epi32(1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80);
    __m256i d1 = _mm256_and_si256(b1, m2); // isolate one bit in each dword
    ymm = _mm256_castsi256_ps(_mm256_cmpgt_epi32(d1, _mm256_setzero_si256())); // compare with 0
    return *this;
  }
#else
  // Member function to change a bitfield to an AVX Boolean vector.
  // Cannot use f32 instructions if subnormals are disabled
  BroadMask& load_bits(u8 a) {
    Half y0 = Half().load_bits(a);
    Half y1 = Half().load_bits(u8(a >> 4U));
    *this = BroadMask(y0, y1);
    return *this;
  }
#endif
  // Member function to change a single element in vector
  BroadMask& insert(std::size_t index, bool value) {
    static constexpr std::array<i32, 16> maskl{0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0};
    // mask with FFFFFFFF at index position
    __m256 mask = _mm256_loadu_ps(reinterpret_cast<const f32*>(maskl.data() + 8 - (index & 7U)));
    if (value) {
      ymm = _mm256_or_ps(ymm, mask);
    } else {
      ymm = _mm256_andnot_ps(mask, ymm);
    }
    return *this;
  }
  // Member function extract a single element from vector
  bool extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    __m256i x = _mm256_maskz_compress_epi32(__mmask8(1U << index), _mm256_castps_si256(ymm));
    return _mm_cvtsi128_si32(_mm256_castsi256_si128(x)) != 0;
#else
    i32 x[8];
    _mm256_storeu_ps(reinterpret_cast<f32*>(x), ymm);
    return x[index & 7] != 0;
#endif
  }
  // Extract a single element. Operator [] can only read an element, not write.
  bool operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two Vec4fb:
  Half get_low() const {
    return _mm256_castps256_ps128(ymm);
  }
  Half get_high() const {
    return _mm256_extractf128_ps(ymm, 1);
  }

  // Prevent constructing from int, etc.
  BroadMask(int b) = delete;
  BroadMask& operator=(int x) = delete;

private:
  __m256 ymm;
};
using b32x8 = BroadMask<32, 8>;

// vector operator & : bitwise and
static inline b32x8 operator&(const b32x8 a, const b32x8 b) {
  return _mm256_and_ps(a, b);
}
static inline b32x8 operator&&(const b32x8 a, const b32x8 b) {
  return a & b;
}

// vector operator &= : bitwise and
static inline b32x8& operator&=(b32x8& a, const b32x8 b) {
  a = a & b;
  return a;
}

// vector operator | : bitwise or
static inline b32x8 operator|(const b32x8 a, const b32x8 b) {
  return _mm256_or_ps(a, b);
}
static inline b32x8 operator||(const b32x8 a, const b32x8 b) {
  return a | b;
}

// vector operator |= : bitwise or
static inline b32x8& operator|=(b32x8& a, const b32x8 b) {
  a = a | b;
  return a;
}

// vector operator ~ : bitwise not
static inline b32x8 operator~(const b32x8 a) {
  const auto mask =
    _mm256_setr_epi32(i32(0xFFFFFFFF), i32(0xFFFFFFFF), i32(0xFFFFFFFF), i32(0xFFFFFFFF),
                      i32(0xFFFFFFFF), i32(0xFFFFFFFF), i32(0xFFFFFFFF), i32(0xFFFFFFFF));
  return _mm256_xor_ps(a, _mm256_castsi256_ps(mask));
}

// vector operator ^ : bitwise xor
static inline b32x8 operator^(const b32x8 a, const b32x8 b) {
  return _mm256_xor_ps(a, b);
}

// vector operator == : xnor
static inline b32x8 operator==(const b32x8 a, const b32x8 b) {
  return b32x8(a ^ b32x8(~b));
}

// vector operator != : xor
static inline b32x8 operator!=(const b32x8 a, const b32x8 b) {
  return _mm256_xor_ps(a, b);
}

// vector operator ^= : bitwise xor
static inline b32x8& operator^=(b32x8& a, const b32x8 b) {
  a = a ^ b;
  return a;
}

// vector operator ! : logical not
static inline b32x8 operator!(const b32x8 a) {
  return ~a;
}

// Functions for b32x8

// andnot: a & ~ b
static inline b32x8 andnot(const b32x8 a, const b32x8 b) {
  return _mm256_andnot_ps(b, a);
}

// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and(const b32x8 a) {
  const auto mask =
    _mm256_setr_epi32(i32(0xFFFFFFFF), i32(0xFFFFFFFF), i32(0xFFFFFFFF), i32(0xFFFFFFFF),
                      i32(0xFFFFFFFF), i32(0xFFFFFFFF), i32(0xFFFFFFFF), i32(0xFFFFFFFF));
  return _mm256_testc_ps(a, _mm256_castsi256_ps(mask)) != 0;
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or(const b32x8 a) {
  return _mm256_testz_ps(a, a) == 0;
}

// to_bits: convert boolean vector to integer bitfield
static inline u8 to_bits(const b32x8 x) {
  __m128i a = _mm_packs_epi32(x.get_low(), x.get_high()); // 32-bit dwords to 16-bit words
  __m128i b = _mm_packs_epi16(a, a); // 16-bit words to bytes
  return u8(_mm_movemask_epi8(b));
}
} // namespace stado
#endif

#endif // INCLUDE_STADO_MASK_BROAD_32X08_HPP
