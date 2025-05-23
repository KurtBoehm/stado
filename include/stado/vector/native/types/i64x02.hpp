#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I64X02_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I64X02_HPP

#include <concepts>
#include <cstddef>
#include <limits>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-128.hpp"
#include "stado/vector/native/types/i08x16.hpp"

namespace stado {
template<typename TDerived, typename T>
struct x64x2 : public si128 {
  using Value = T;
  static constexpr std::size_t size = 2;

  // Default constructor:
  x64x2() = default;
  // Constructor to broadcast the same value into all elements:
  x64x2(T i) : si128(_mm_set1_epi64x(i64(i))) {}
  // Constructor to build from all elements:
  x64x2(T i0, T i1) : si128(_mm_set_epi64x(i64(i1), i64(i0))) {}
  // Constructor to convert from type __m128i used in intrinsics:
  x64x2(const __m128i x) : si128(x) {}
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
    xmm = _mm_maskz_loadu_epi64(__mmask8((1U << n) - 1), p);
#else
    switch (n) {
    case 0: *this = 0; break;
    case 1:
      // intrinsic for movq is missing!
      *this = x64x2(*(const T*)p, 0);
      break;
    case 2: load(p); break;
    default: break;
    }
#endif
    return derived();
  }
  template<std::size_t n>
  TDerived& load_partial(const void* p) {
    static_assert(n <= 2);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    xmm = _mm_maskz_loadu_epi64(__mmask8((1U << n) - 1), p);
#else
    if constexpr (n == 0) {
      *this = 0;
    } else if constexpr (n == 1) {
      // intrinsic for movq is missing!
      *this = x64x2(*(const T*)p, 0);
    } else {
      load(p);
    }
#endif
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
    _mm_mask_storeu_epi64(p, __mmask8((1U << n) - 1), xmm);
#else
    switch (n) {
    case 1:
      T q[2];
      store(q);
      *(T*)p = q[0];
      break;
    case 2: store(p); break;
    default: break;
    }
#endif
  }
  // cut off vector to n elements. The last 2-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    xmm = _mm_maskz_mov_epi64(~__mmask8(std::numeric_limits<__mmask8>::max() >> n), xmm);
#else
    *this = i8x16(xmm).cutoff(n * 8);
#endif
    return derived();
  }
  // Member function to change a single element in vector
  // Note: This function is inefficient. Use load function if changing more than one element
  TDerived& insert(std::size_t index, T value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    xmm = _mm_mask_set1_epi64(xmm, __mmask8(1U << index), i64(value));
#elif STADO_INSTRUCTION_SET >= STADO_SSE4_1
    if (index == 0) {
      xmm = _mm_insert_epi64(xmm, value, 0);
    } else {
      xmm = _mm_insert_epi64(xmm, value, 1);
    }
#else // SSE2
    __m128i v = _mm_cvtsi64_si128(value);
    if (index == 0) {
      v = _mm_unpacklo_epi64(v, v);
      xmm = _mm_unpackhi_epi64(v, xmm);
    } else {
      // index = 1
      xmm = _mm_unpacklo_epi64(xmm, v);
    }
#endif
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    __m128i x = _mm_mask_unpackhi_epi64(xmm, __mmask8(index), xmm, xmm);
    return T(_mm_cvtsi128_si64(x));
#else
    T x[2];
    store(x);
    return x[index & 1];
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
struct NativeVector<i64, 2> : public x64x2<NativeVector<i64, 2>, i64> {
  using x64x2<NativeVector<i64, 2>, i64>::x64x2;
};
using i64x2 = NativeVector<i64, 2>;

template<>
struct NativeVector<u64, 2> : public x64x2<NativeVector<u64, 2>, u64> {
  using x64x2<NativeVector<u64, 2>, u64>::x64x2;
};
using u64x2 = NativeVector<u64, 2>;

template<typename TVec>
concept AnyInt64x2 = std::same_as<TVec, i64x2> || std::same_as<TVec, u64x2>;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I64X02_HPP
