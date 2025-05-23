#ifndef INCLUDE_STADO_VECTOR_SINGLE_HPP
#define INCLUDE_STADO_VECTOR_SINGLE_HPP

#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "stado/mask/single.hpp"

namespace stado {
template<typename T>
requires(std::is_trivial_v<T>)
struct SingleVector {
  using Element = T;
  static constexpr std::size_t size = 1;

  static SingleVector expand_zero(T value) {
    return SingleVector{value};
  }

  SingleVector() = default;
  explicit SingleVector(T value) : value_(value) {}

  SingleVector& load(const T* data) {
    value_ = *data;
    return *this;
  }

  SingleVector& load_partial(std::size_t part, const T* data) {
    value_ = (part > 0) ? *data : 0;
    return *this;
  }

  void store(T* ptr) const {
    *ptr = value_;
  }

  void store_partial(std::size_t part, T* ptr) const {
    if (part > 0) {
      *ptr = value_;
    }
  }

  T value() const {
    return value_;
  }
  T& value() {
    return value_;
  }

  void cutoff(std::size_t part) {
    if (part == 0) {
      value_ = 0;
    }
  }

  SingleVector operator-() const {
    return SingleVector{-value_};
  }

  T operator[]([[maybe_unused]] std::size_t index) const {
    assert(index == 0);
    return value_;
  }

  SingleVector& insert([[maybe_unused]] std::size_t index, T value) {
    assert(index == 0);
    value_ = value;
    return *this;
  }

private:
  T value_{};
};

#define COLLECTIVIST_ASSIGN_ARITH_OP(OP) \
  template<typename T> \
  inline SingleVector<T>& operator OP(SingleVector<T>& v1, SingleVector<T> v2) { \
    v1.value() OP v2.value(); \
    return v1; \
  }

COLLECTIVIST_ASSIGN_ARITH_OP(+=)
COLLECTIVIST_ASSIGN_ARITH_OP(-=)
COLLECTIVIST_ASSIGN_ARITH_OP(*=)
COLLECTIVIST_ASSIGN_ARITH_OP(/=)

#undef COLLECTIVIST_ASSIGN_ARITH_OP

#define COLLECTIVIST_ARITH_OP(OP) \
  template<typename T> \
  inline SingleVector<T> operator OP(SingleVector<T> v1, SingleVector<T> v2) { \
    return SingleVector<T>{v1.value() OP v2.value()}; \
  } \
  template<typename T> \
  inline SingleVector<T> operator OP(T v1, SingleVector<T> v2) { \
    return SingleVector<T>{v1 OP v2.value()}; \
  } \
  template<typename T> \
  inline SingleVector<T> operator OP(SingleVector<T> v1, T v2) { \
    return SingleVector<T>{v1.value() OP v2}; \
  }

COLLECTIVIST_ARITH_OP(+)
COLLECTIVIST_ARITH_OP(-)
COLLECTIVIST_ARITH_OP(*)
COLLECTIVIST_ARITH_OP(/)

#undef COLLECTIVIST_ARITH_OP

#define COLLECTIVIST_CMP_OP(OP) \
  template<typename T> \
  inline SingleMask operator OP(SingleVector<T> v1, SingleVector<T> v2) { \
    return SingleMask{v1.value() OP v2.value()}; \
  } \
  template<typename T> \
  inline SingleMask operator OP(T v1, SingleVector<T> v2) { \
    return SingleMask{v1 OP v2.value()}; \
  } \
  template<typename T> \
  inline SingleMask operator OP(SingleVector<T> v1, T v2) { \
    return SingleMask{v1.value() OP v2}; \
  }

COLLECTIVIST_CMP_OP(==)
COLLECTIVIST_CMP_OP(!=)
COLLECTIVIST_CMP_OP(<)
COLLECTIVIST_CMP_OP(>)
COLLECTIVIST_CMP_OP(<=)
COLLECTIVIST_CMP_OP(>=)

#undef COLLECTIVIST_CMP_OP

template<typename T>
inline SingleVector<T> abs(SingleVector<T> v) {
  return SingleVector<T>{std::abs(v.value())};
}

template<typename T>
inline T horizontal_add(SingleVector<T> v) {
  return v.value();
}

template<typename T>
inline SingleVector<T> if_add(SingleMask m, SingleVector<T> v1, SingleVector<T> v2) {
  return m.value() ? (v1 + v2) : v1;
}
template<typename T>
inline SingleVector<T> if_div(SingleMask m, SingleVector<T> v1, SingleVector<T> v2) {
  return m.value() ? (v1 / v2) : v1;
}

template<typename T>
inline SingleVector<T> select(SingleMask m, SingleVector<T> v1, SingleVector<T> v2) {
  return m.value() ? v1 : v2;
}

template<typename T>
inline SingleVector<T> max(SingleVector<T> v1, SingleVector<T> v2) {
  return SingleVector<T>{std::max(v1.value(), v2.value())};
}

template<typename T1, typename T2>
inline SingleVector<T2> lookup(SingleVector<T1> indices, const T2* data) {
  return SingleVector<T2>{data[indices.value()]};
}

template<typename T>
inline SingleVector<T> shuffle_up(SingleVector<T> /*vec*/) {
  return SingleVector<T>(0);
}
template<typename T>
inline SingleVector<T> shuffle_up(SingleVector<T> /*vec*/, T first) {
  return SingleVector<T>(first);
}

template<typename T>
inline SingleVector<T> shuffle_down(SingleVector<T> /*vec*/) {
  return SingleVector<T>(0);
}
template<typename T>
inline SingleVector<T> shuffle_down(SingleVector<T> /*vec*/, T first) {
  return SingleVector<T>(first);
}

namespace detail {
template<typename T>
inline T fma(T x, T y, T z) {
  if constexpr (std::same_as<T, float>) {
    return __builtin_fmaf(x, y, z);
  }
  if constexpr (std::same_as<T, double>) {
    return __builtin_fma(x, y, z);
  }
  if constexpr (std::same_as<T, long double>) {
    return __builtin_fmal(x, y, z);
  }
}
} // namespace detail

template<typename T>
inline SingleVector<T> mul_add(SingleVector<T> x, SingleVector<T> y, SingleVector<T> z) {
  return SingleVector<T>{detail::fma(x.value(), y.value(), z.value())};
}
template<typename T>
inline SingleVector<T> mul_sub(SingleVector<T> x, SingleVector<T> y, SingleVector<T> z) {
  return SingleVector<T>{detail::fma(x.value(), y.value(), -z.value())};
}
template<typename T>
inline SingleVector<T> nmul_add(SingleVector<T> x, SingleVector<T> y, SingleVector<T> z) {
  return SingleVector<T>{detail::fma(-x.value(), y.value(), z.value())};
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_SINGLE_HPP
