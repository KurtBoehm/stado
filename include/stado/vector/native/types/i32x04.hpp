#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I32X04_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I32X04_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i08x16.hpp"

namespace stado {
template<typename TDerived, typename T>
struct x32x4 : public si128 {
  using Value = T;
  static constexpr std::size_t size = 4;

  // Default constructor:
  x32x4() = default;
  // Constructor to broadcast the same value into all elements:
  x32x4(T i) : si128(_mm_set1_epi32(i32(i))) {}
  // Constructor to build from all elements:
  x32x4(T i0, T i1, T i2, T i3) : si128(_mm_setr_epi32(i32(i0), i32(i1), i32(i2), i32(i3))) {}
  // Constructor to convert from type __m128i used in intrinsics:
  x32x4(const __m128i x) : si128(x) {}
  // Assignment operator to convert from type __m128i used in intrinsics:
  TDerived& operator=(const __m128i x) {
    xmm = x;
    return derived();
  }
  // Type cast operator to convert to __m128i used in intrinsics
  operator __m128i() const {
    return xmm;
  }
  // Member function to load from array (unaligned)
  TDerived& load(const void* p) {
    xmm = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return derived();
  }
  // Member function to load from array (aligned)
  TDerived& load_a(const void* p) {
    xmm = _mm_load_si128(reinterpret_cast<const __m128i*>(p));
    return derived();
  }
  // Partial load. Load n elements and set the rest to 0
  TDerived& load_partial(std::size_t n, const void* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    xmm = _mm_maskz_loadu_epi32(__mmask8((1U << n) - 1), p);
#else
    switch (n) {
    case 0: *this = 0; break;
    case 1: xmm = _mm_cvtsi32_si128(*reinterpret_cast<const i32*>(p)); break;
    case 2:
      // intrinsic for movq is missing!
      xmm = _mm_setr_epi32((reinterpret_cast<const i32*>(p))[0],
                           (reinterpret_cast<const i32*>(p))[1], 0, 0);
      break;
    case 3:
      xmm =
        _mm_setr_epi32((reinterpret_cast<const i32*>(p))[0], (reinterpret_cast<const i32*>(p))[1],
                       (reinterpret_cast<const i32*>(p))[2], 0);
      break;
    case 4: load(p); break;
    default: break;
    }
#endif
    return derived();
  }
  template<std::size_t tN>
  TDerived& load_partial(const void* p) {
    static_assert(tN <= 4);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    xmm = _mm_maskz_loadu_epi32(__mmask8((1U << tN) - 1), p);
#else
    const i32* p32 = reinterpret_cast<const i32*>(p);
    if constexpr (tN == 0) {
      *this = 0;
    } else if constexpr (tN == 1) {
      xmm = _mm_cvtsi32_si128(*p32);
    } else if constexpr (tN == 2) {
      // intrinsic for movq is missing!
      xmm = _mm_setr_epi32(p32[0], p32[1], 0, 0);
    } else if constexpr (tN == 3) {
      xmm = _mm_setr_epi32(p32[0], p32[1], p32[2], 0);
    } else {
      load(p);
    }
#endif
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
    _mm_mask_storeu_epi32(p, __mmask8((1U << n) - 1), xmm);
#else
    union {
      T i[4];
      i64 q[2];
    } u;
    switch (n) {
    case 1: *reinterpret_cast<i32*>(p) = _mm_cvtsi128_si32(xmm); break;
    case 2:
      // intrinsic for movq is missing!
      store(u.i);
      *reinterpret_cast<i64*>(p) = u.q[0];
      break;
    case 3:
      store(u.i);
      *reinterpret_cast<i64*>(p) = u.q[0];
      reinterpret_cast<i32*>(p)[2] = u.i[2];
      break;
    case 4: store(p); break;
    default: break;
    }
#endif
  }

  // cut off vector to n elements. The last 4-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    xmm = _mm_maskz_mov_epi32(__mmask8((1U << n) - 1), xmm);
#else
    *this = i8x16(xmm).cutoff(n * 4);
#endif
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    xmm = _mm_mask_set1_epi32(xmm, __mmask8(1U << index), i32(value));
#else
    // broadcast value into all elements
    __m128i broad = _mm_set1_epi32(value);
    static constexpr std::array<T, 8> maskl{0, 0, 0, 0, -1, 0, 0, 0};
    // mask with FFFFFFFF at index position
    __m128i mask =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(maskl.data() + 4 - (index & 3U)));
    xmm = selectb(mask, broad, xmm);
#endif
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    __m128i x = _mm_maskz_compress_epi32(__mmask8(1U << index), xmm);
    return T(_mm_cvtsi128_si32(x));
#else
    T x[4];
    store(x);
    return x[index & 3];
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  T operator[](std::size_t index) const {
    return extract(index);
  }

private:
  [[nodiscard]] const TDerived& derived() const {
    return static_cast<const TDerived&>(*this);
  }
  [[nodiscard]] TDerived& derived() {
    return static_cast<TDerived&>(*this);
  }
};

template<>
struct NativeVector<i32, 4> : public x32x4<NativeVector<i32, 4>, i32> {
  using x32x4<NativeVector<i32, 4>, i32>::x32x4;
};
using i32x4 = NativeVector<i32, 4>;

template<>
struct NativeVector<u32, 4> : public x32x4<NativeVector<u32, 4>, u32> {
  using x32x4<NativeVector<u32, 4>, u32>::x32x4;
};
using u32x4 = NativeVector<u32, 4>;

template<typename TVec>
concept AnyInt32x4 = std::same_as<TVec, i32x4> || std::same_as<TVec, u32x4>;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I32X04_HPP
