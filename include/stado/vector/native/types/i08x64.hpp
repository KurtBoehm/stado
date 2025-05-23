#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_I08X64_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_I08X64_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/base-512.hpp"
#include "stado/vector/native/types/i08x32.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
namespace stado {
template<typename TDerived, typename T>
struct x8x64 : public si512 {
  using Element = T;
  using Half = NativeVector<T, 32>;
  static constexpr std::size_t size = 64;

  // Default constructor:
  x8x64() = default;
  // Constructor to broadcast the same value into all elements:
  x8x64(T i) : si512(_mm512_set1_epi8(i8(i))) {}
  // Constructor to build from all elements:
  x8x64(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13,
        T v14, T v15, T v16, T v17, T v18, T v19, T v20, T v21, T v22, T v23, T v24, T v25, T v26,
        T v27, T v28, T v29, T v30, T v31, T v32, T v33, T v34, T v35, T v36, T v37, T v38, T v39,
        T v40, T v41, T v42, T v43, T v44, T v45, T v46, T v47, T v48, T v49, T v50, T v51, T v52,
        T v53, T v54, T v55, T v56, T v57, T v58, T v59, T v60, T v61, T v62, T v63)
      : si512(_mm512_set_epi8(
          i8(v63), i8(v62), i8(v61), i8(v60), i8(v59), i8(v58), i8(v57), i8(v56), i8(v55), i8(v54),
          i8(v53), i8(v52), i8(v51), i8(v50), i8(v49), i8(v48), i8(v47), i8(v46), i8(v45), i8(v44),
          i8(v43), i8(v42), i8(v41), i8(v40), i8(v39), i8(v38), i8(v37), i8(v36), i8(v35), i8(v34),
          i8(v33), i8(v32), i8(v31), i8(v30), i8(v29), i8(v28), i8(v27), i8(v26), i8(v25), i8(v24),
          i8(v23), i8(v22), i8(v21), i8(v20), i8(v19), i8(v18), i8(v17), i8(v16), i8(v15), i8(v14),
          i8(v13), i8(v12), i8(v11), i8(v10), i8(v9), i8(v8), i8(v7), i8(v6), i8(v5), i8(v4),
          i8(v3), i8(v2), i8(v1), i8(v0))) {}
  // Constructor to build from two x8x32:
  x8x64(const Half a0, const Half a1)
      : si512(_mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1)) {}
  // Constructor to convert from type __m512i used in intrinsics:
  x8x64(const __m512i x) : si512(x) {}
  // Assignment operator to convert from type __m512i used in intrinsics:
  TDerived& operator=(const __m512i x) {
    zmm = x;
    return derived();
  }
  // Type cast operator to convert to __m512i used in intrinsics
  operator __m512i() const {
    return zmm;
  }
  // Member function to load from array (unaligned)
  TDerived& load(const void* p) {
    zmm = _mm512_loadu_si512(p);
    return derived();
  }
  // Member function to load from array, aligned by 64
  TDerived& load_a(const void* p) {
    zmm = _mm512_load_si512(p);
    return derived();
  }
  // Partial load. Load n elements and set the rest to 0
  TDerived& load_partial(std::size_t n, const void* p) {
    if (n >= 64) {
      zmm = _mm512_loadu_si512(p);
    } else {
      zmm = _mm512_maskz_loadu_epi8(__mmask64((u64{1} << n) - 1), p);
    }
    return derived();
  }
  template<std::size_t tN>
  TDerived& load_partial(const void* p) {
    if (tN >= 64) {
      zmm = _mm512_loadu_si512(p);
    } else {
      zmm = _mm512_maskz_loadu_epi8(__mmask64((u64{1} << tN) - 1), p);
    }
    return derived();
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, void* p) const {
    if (n >= 64) {
      // _mm512_storeu_epi8(p, zmm);
      _mm512_storeu_si512(p, zmm);
    } else {
      _mm512_mask_storeu_epi8(p, __mmask64((u64{1} << n) - 1), zmm);
    }
  }
  // cut off vector to n elements. The last 64-n elements are set to zero
  TDerived& cutoff(std::size_t n) {
    if (n < 64) {
      zmm = _mm512_maskz_mov_epi8(__mmask64((u64{1} << n) - 1), zmm);
    }
    return derived();
  }
  // Member function to change a single element in vector
  TDerived& insert(std::size_t index, T value) {
    zmm = _mm512_mask_set1_epi8(zmm, __mmask64(u64{1} << index), i8(value));
    return derived();
  }
  // Member function extract a single element from vector
  [[nodiscard]] T extract(std::size_t index) const {
#if defined(__AVX512VBMI2__)
    const __m512i x = _mm512_maskz_compress_epi8(__mmask64(u64{1} << index), zmm);
    return T(_mm_cvtsi128_si32(_mm512_castsi512_si128(x)));
#else
    T a[64];
    store(a);
    return a[index & 63];
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  T operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two x8x32:
  [[nodiscard]] Half get_low() const {
    return _mm512_castsi512_si256(zmm);
  }
  [[nodiscard]] Half get_high() const {
    return _mm512_extracti64x4_epi64(zmm, 1);
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
struct NativeVector<i8, 64> : public x8x64<NativeVector<i8, 64>, i8> {
  using x8x64<NativeVector<i8, 64>, i8>::x8x64;
};
using i8x64 = NativeVector<i8, 64>;

template<>
struct NativeVector<u8, 64> : public x8x64<NativeVector<u8, 64>, u8> {
  using x8x64<NativeVector<u8, 64>, u8>::x8x64;
};
using u8x64 = NativeVector<u8, 64>;
} // namespace stado
#endif

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_I08X64_HPP
