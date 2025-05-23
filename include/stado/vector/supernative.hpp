#ifndef INCLUDE_STADO_VECTOR_SUPERNATIVE_HPP
#define INCLUDE_STADO_VECTOR_SUPERNATIVE_HPP

#include <array>
#include <bit>
#include <cstddef>

#include "stado/vector/base.hpp"

namespace stado {
template<typename T, std::size_t tSize>
requires(std::has_single_bit(tSize))
struct SuperNativeVector {
  using Element = T;
  static constexpr std::size_t size = tSize;
  static constexpr std::size_t storage_size = size / 2;
  using Half = Vector<T, storage_size>;

  SuperNativeVector() = default;

  template<typename... TArgs>
  static SuperNativeVector from_values(TArgs&&... args) {
    static_assert(sizeof...(TArgs) == size);
    return SuperNativeVector(std::array<T, storage_size>{std::forward<TArgs>(args)...});
  }

  explicit SuperNativeVector(std::array<T, size> full_data) {
    load(full_data.data());
  }

  SuperNativeVector(Half lower, Half upper) : lower_(lower), upper_(upper) {}
  explicit SuperNativeVector(T value) : lower_(value), upper_(value) {}

  SuperNativeVector& load(const T* data) {
    lower_.load(data);
    upper_.load(data + storage_size);
    return *this;
  }
  SuperNativeVector& load_partial(std::size_t part, const T* data) {
    if (part >= storage_size) {
      lower_.load(data);
      upper_.load_partial(part - storage_size, data + storage_size);
    } else {
      lower_.load_partial(part, data);
      upper_ = Half(0);
    }
    return *this;
  }

  void store(T* data) {
    lower_.store(data);
    upper_.store(data + storage_size);
  }
  void store_partial(std::size_t part, T* data) {
    if (part >= storage_size) {
      lower_.store(data);
      upper_.store_partial(part - storage_size, data + storage_size);
    } else {
      lower_.store_partial(part, data);
    }
  }

  Half& lower() {
    return lower_;
  }
  const Half& lower() const {
    return lower_;
  }

  Half& upper() {
    return upper_;
  }
  const Half& upper() const {
    return upper_;
  }

  SuperNativeVector& cutoff(std::size_t size) {
    if (size >= storage_size) {
      upper_.cutoff(size - storage_size);
    } else {
      lower_.cutoff(size);
      upper_ = Half(0);
    }
    return *this;
  }

  friend SuperNativeVector operator+(const SuperNativeVector& a, const SuperNativeVector& b) {
    return SuperNativeVector(a.lower_ + b.lower_, a.upper_ + b.upper_);
  }

  friend SuperNativeVector operator-(const SuperNativeVector& a, const SuperNativeVector& b) {
    return SuperNativeVector(a.lower_ - b.lower_, a.upper_ - b.upper_);
  }

  friend SuperNativeVector operator*(const SuperNativeVector& a, const SuperNativeVector& b) {
    return SuperNativeVector(a.lower_ * b.lower_, a.upper_ * b.upper_);
  }

  friend SuperNativeVector operator*(T a, const SuperNativeVector& b) {
    return SuperNativeVector(a * b.lower_, a * b.upper_);
  }

  friend SuperNativeVector operator*(const SuperNativeVector& a, T b) {
    return SuperNativeVector(a.lower_ * b, a.upper_ * b);
  }

private:
  Half lower_{};
  Half upper_{};
};

template<typename T, std::size_t tSize>
static T horizontal_add(SuperNativeVector<T, tSize> vec) {
  return horizontal_add(vec.lower()) + horizontal_add(vec.upper());
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_SUPERNATIVE_HPP
