#ifndef INCLUDE_STADO_MASK_SINGLE_HPP
#define INCLUDE_STADO_MASK_SINGLE_HPP

#include <concepts>
#include <cstddef>

#include "stado/mask/base.hpp"

namespace stado {
struct SingleMask {
  explicit SingleMask(bool value) : value_(value) {}

  void insert(std::size_t index, bool value) {
    if (index == 0) {
      value_ = value;
    }
  }

  [[nodiscard]] bool value() const {
    return value_;
  }

private:
  bool value_;
};

inline SingleMask operator&(SingleMask m1, SingleMask m2) {
  return SingleMask{m1.value() && m2.value()};
}

template<std::size_t tValueBits>
struct MaskTrait<tValueBits, 1> {
  using Type = SingleMask;
};

template<typename T>
concept AnySingleMask = std::same_as<T, SingleMask>;
} // namespace stado

#endif // INCLUDE_STADO_MASK_SINGLE_HPP
