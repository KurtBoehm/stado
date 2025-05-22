#ifndef INCLUDE_STADO_VECTOR_NATIVE_TYPES_F64X8_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_TYPES_F64X8_HPP

#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/types/f64x2.hpp"
#include "stado/vector/native/types/f64x4.hpp"

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
namespace stado {
template<>
struct NativeVector<f64, 8> : public NativeVectorBase<f64, 8> {
  using Register = __m512d;

  static NativeVector expand_undef(f64 x) {
    return _mm512_castpd128_pd512(f64x2::expand_undef(x));
  }
  static NativeVector expand_zero(f64 x) {
    return {x, 0, 0, 0, 0, 0, 0, 0};
  }

  // Default constructor:
  NativeVector() = default;
  // Constructor to broadcast the same value into all elements:
  NativeVector(f64 d) : zmm_(_mm512_set1_pd(d)) {}
  // Constructor to build from all elements:
  NativeVector(f64 d0, f64 d1, f64 d2, f64 d3, f64 d4, f64 d5, f64 d6, f64 d7)
      : zmm_(_mm512_setr_pd(d0, d1, d2, d3, d4, d5, d6, d7)) {}
  // Constructor to build from two f64x4:
  NativeVector(const f64x4 a0, const f64x4 a1)
      : zmm_(_mm512_insertf64x4(_mm512_castpd256_pd512(a0), a1, 1)) {}
  // Constructor to convert from type __m512d used in intrinsics:
  NativeVector(const __m512d x) : zmm_(x) {}
  // Assignment operator to convert from type __m512d used in intrinsics:
  NativeVector& operator=(const __m512d x) {
    zmm_ = x;
    return *this;
  }
  // Type cast operator to convert to __m512d used in intrinsics
  operator __m512d() const {
    return zmm_;
  }
  // Member function to load from array (unaligned)
  NativeVector& load(const f64* p) {
    zmm_ = _mm512_loadu_pd(p);
    return *this;
  }
  // Member function to load from array, aligned by 64
  // You may use load_a instead of load if you are certain that p points to an address
  // divisible by 64
  NativeVector& load_a(const f64* p) {
    zmm_ = _mm512_load_pd(p);
    return *this;
  }
  // Member function to store into array (unaligned)
  void store(f64* p) const {
    _mm512_storeu_pd(p, zmm_);
  }
  // Member function storing into array, aligned by 64
  // You may use store_a instead of store if you are certain that p points to an address
  // divisible by 64
  void store_a(f64* p) const {
    _mm512_store_pd(p, zmm_);
  }
  // Member function storing to aligned uncached memory (non-temporal store).
  // This may be more efficient than store_a when storing large blocks of memory if it
  // is unlikely that the data will stay in the cache until it is read again.
  // Note: Will generate runtime error if p is not aligned by 16
  void store_nt(f64* p) const {
    _mm512_stream_pd(p, zmm_);
  }
  // Partial load. Load n elements and set the rest to 0
  NativeVector& load_partial(std::size_t n, const f64* p) {
    zmm_ = _mm512_maskz_loadu_pd(__mmask8((1U << n) - 1), p);
    return *this;
  }
  template<std::size_t n>
  NativeVector& load_partial(const f64* p) {
    zmm_ = _mm512_maskz_loadu_pd(__mmask8((1U << n) - 1), p);
    return *this;
  }
  // Partial store. Store n elements
  void store_partial(std::size_t n, f64* p) const {
    _mm512_mask_storeu_pd(p, __mmask8((1U << n) - 1), zmm_);
  }
  // cut off vector to n elements. The last 8-n elements are set to zero
  NativeVector& cutoff(std::size_t n) {
    zmm_ = _mm512_maskz_mov_pd(__mmask8((1U << n) - 1), zmm_);
    return *this;
  }
  // Member function to change a single element in vector
  NativeVector& insert(std::size_t index, f64 value) {
    zmm_ = _mm512_mask_broadcastsd_pd(zmm_, __mmask8(1U << index), _mm_set_sd(value));
    return *this;
  }
  // Member function extract a single element from vector
  [[nodiscard]] f64 extract(std::size_t index) const {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL
    const __m512d x = _mm512_maskz_compress_pd(__mmask8(1U << index), zmm_);
    return _mm512_cvtsd_f64(x);
#else
    f64 a[8];
    store(a);
    return a[index & 7];
#endif
  }
  // Extract a single element. Use store function if extracting more than one element.
  // Operator [] can only read an element, not write.
  f64 operator[](std::size_t index) const {
    return extract(index);
  }
  // Member functions to split into two f64x4:
  [[nodiscard]] f64x4 get_low() const {
    return _mm512_castpd512_pd256(zmm_);
  }
  [[nodiscard]] f64x4 get_high() const {
    return _mm512_extractf64x4_pd(zmm_, 1);
  }

private:
  __m512d zmm_;
};

using f64x8 = NativeVector<f64, 8>;
} // namespace stado
#endif // AVX512F

#endif // INCLUDE_STADO_VECTOR_NATIVE_TYPES_F64X8_HPP
