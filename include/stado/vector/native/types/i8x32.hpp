#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I8X32_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I8X32_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-256.hpp"
#include "stado/vector/native/types/i8x16.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX2
namespace stado {
template<typename TDerived, typename T>
struct x8x32 : public si256 {
  using Element = T;
  using Half = NativeVector<T, 16>;
  static constexpr std::size_t size = 32;

  // Default constructor:
  x8x32() = default;
  // Constructor to broadcast the same value into all elements:
  x8x32(T i) : si256(_mm256_set1_epi8(i8(i))) {}
  // Constructor to build from all elements:
  x8x32(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13,
        T v14, T v15, T v16, T v17, T v18, T v19, T v20, T v21, T v22, T v23, T v24, T v25, T v26,
        T v27, T v28, T v29, T v30, T v31)
      : si256(_mm256_setr_epi8(i8(v0), i8(v1), i8(v2), i8(v3), i8(v4), i8(v5), i8(v6), i8(v7),
                               i8(v8), i8(v9), i8(v10), i8(v11), i8(v12), i8(v13), i8(v14), i8(v15),
                               i8(v16), i8(v17), i8(v18), i8(v19), i8(v20), i8(v21), i8(v22),
                               i8(v23), i8(v24), i8(v25), i8(v26), i8(v27), i8(v28), i8(v29),
                               i8(v30), i8(v31))) {}
  // Constructor to build from two x8x16:
  x8x32(const Half a0, const Half a1) : si256(_mm256_set_m128i(a1, a0)) {}
  // Constructor to convert from type __m256i used in intrinsics:
  x8x32(const __m256i x) : si256(x) {}
  // Assignment operator to convert from type __m256i used in intrinsics:
  TDerived& operator=(const __m256i x) {
    ymm = x;
    return derived();
  }
  // Constructor to convert from type si256 used in emulation
  x8x32(const si256 x) : si256(x) {}
  // Type cast operator to convert to __m256i used in intrinsics
  operator __m256i() const {
    return ymm;
  }
  // Member function to load from array (unaligned)
  TDerived& load(const void* p) {
    ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return derived();
  }
  // Member function to load from array, aligned by 32
  TDerived& load_a(const void* p) {
    ymm = _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    return derived();
  }
  // Partial load. Load n elements and set the rest to 0
  TDerived& load_partial(std::size_t n, const void* p) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm = _mm256_maskz_loadu_epi8(__mmask32((u64{1} << n) - 1), p);
#else
    if (n <= 0) {
      *this = 0;
    } else if (n <= 16) {
      *this = NativeVector(Half{}.load_partial(n, p), 0);
    } else if (n < 32) {
      *this = NativeVector(Half{}.load(p), Half{}.load_partial(n - 16, (const T*)p + 16));
    } else {
      load(p);
    }
#endif
    return derived();
  }
  template<std::size_t tN>
  TDerived& load_partial(const void* p) {
    static_assert(tN <= 32);
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL
    ymm = _mm256_maskz_loadu_epi8(__mmask32((u64{1} << tN) - 1), p);
#else
    if constexpr (tN == 0) {
      *this = 0;
    } else if constexpr (tN <= 16) {
      *this = NativeVector(Half{}.template load_partial<tN>(p), 0);
    } else if constexpr (tN < 32) {
      *this = NativeVector(Half{}.load(p), Half{}.template load_partial<tN - 16>((const T*)p + 16));
    } else {
      load(p);
    }
#endif
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512VL + AVX512BW
    _mm256_mask_storeu_epi8(p, __mmask32((u64{1} << n) - 1), ymm);
#else
    if (n <= 0) {
      return;
    } else if (n <= 16) {
      get_low().store_partial(n, p);
    } else if (n < 32) {
      get_low().store(p);
      get_high().store_partial(n - 16, (T*)p + 16);
    } else {
      store(p);
    }
#endif
  }
  // cut off vector to n elements. The last 32-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    ymm = _mm256_maskz_mov_epi8(__mmask32((u64{1} << n) - 1), ymm);
#else
    x8x32 index_vec{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    const NativeVector ref_vec(T(n));
    *this &= _mm256_cmpgt_epi8(ref_vec, index_vec);
#endif
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    ymm = _mm256_mask_set1_epi8(ymm, __mmask32(1U << index), i8(value));
#else
    static constexpr std::array<i8, 64> maskl{0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    __m256i broad = _mm256_set1_epi8(value); // broadcast value into all elements
    __m256i mask = _mm256_loadu_si256(
      (const __m256i*)(maskl.data() + 32 - (index & 0x1FU))); // mask with FF at index position
    ymm = selectb(mask, broad, ymm);
#endif
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL && defined(__AVX512VBMI2__)
    const __m256i x = _mm256_maskz_compress_epi8(__mmask32(1U << index), ymm);
    return T(_mm_cvtsi128_si32(_mm256_castsi256_si128(x)));
#else
    T x[32];
    store(x);
    return x[index & 0x1F];
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  T operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two x8x16:
  [[nodiscard]] NativeVector<T, 16> get_low() const {
    return _mm256_castsi256_si128(ymm);
  }
  [[nodiscard]] NativeVector<T, 16> get_high() const {
    return _mm256_extracti128_si256(ymm, 1);
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
struct NativeVector<i8, 32> : public x8x32<NativeVector<i8, 32>, i8> {
  using x8x32<NativeVector<i8, 32>, i8>::x8x32;
};
using i8x32 = NativeVector<i8, 32>;

template<>
struct NativeVector<u8, 32> : public x8x32<NativeVector<u8, 32>, u8> {
  using x8x32<NativeVector<u8, 32>, u8>::x8x32;
};
using u8x32 = NativeVector<u8, 32>;
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I8X32_HPP
