#ifndef INCLUDE_STADO_MASK_SUBNATIVE_HPP
#define INCLUDE_STADO_MASK_SUBNATIVE_HPP

#include <bit>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "stado/mask/base.hpp"

namespace stado {
template<std::size_t tElementBits, std::size_t tSize>
requires(std::has_single_bit(tElementBits) && std::has_single_bit(tSize))
struct SubNativeMask {
  static constexpr std::size_t storage_size = 128 / tElementBits;
  static constexpr std::size_t size = tSize;
  static_assert(storage_size >= size, "All elements have to fit!");
  using Native = Mask<tElementBits, storage_size>;
  using Element = bool;

  static SubNativeMask from_native(Native native) {
    return SubNativeMask(native);
  }

  explicit SubNativeMask(bool value) : native_(value) {}

  void insert(std::size_t index, bool value) {
    assert(index < size);
    native_.insert(index, value);
  }

  [[nodiscard]] bool extract(std::size_t index) const {
    assert(index < size);
    return native_.extract(index);
  }

  [[nodiscard]] Native& native() {
    return native_;
  }
  [[nodiscard]] const Native& native() const {
    return native_;
  }

  [[nodiscard]] Native masked_native() const {
    return native_ & native_mask();
  }
  [[nodiscard]] Native one_extended_native() const {
    return native_ | ~native_mask();
  }

private:
  explicit SubNativeMask(Native native) : native_(native) {}

  static Native native_mask() {
    return create_part_mask<Native>(size);
  }

  Native native_;
};

template<std::size_t tElementBits, std::size_t tElementNum>
inline auto operator&(SubNativeMask<tElementBits, tElementNum> m1,
                      SubNativeMask<tElementBits, tElementNum> m2) {
  const auto native = m1.native() & m2.native();
  return SubNativeMask<tElementBits, tElementNum>::from_native(native);
}

template<std::size_t tElementBits, std::size_t tElementNum>
static inline bool horizontal_or(SubNativeMask<tElementBits, tElementNum> m) {
  return horizontal_or(m.masked_native());
}
template<std::size_t tElementBits, std::size_t tElementNum>
static inline bool horizontal_and(SubNativeMask<tElementBits, tElementNum> m) {
  return horizontal_and(m.one_extended_native());
}

template<std::size_t tElementBits, std::size_t tElementNum>
requires(tElementBits* tElementNum < 128)
struct MaskTrait<tElementBits, tElementNum> {
  using Type = SubNativeMask<tElementBits, tElementNum>;
};

template<typename T>
struct IsSubNativeMaskTrait : public std::false_type {};
template<std::size_t tElementBits, std::size_t tSize>
struct IsSubNativeMaskTrait<SubNativeMask<tElementBits, tSize>> : public std::true_type {};
template<typename T>
concept AnySubNativeMask = IsSubNativeMaskTrait<T>::value;
} // namespace stado

#endif // INCLUDE_STADO_MASK_SUBNATIVE_HPP
