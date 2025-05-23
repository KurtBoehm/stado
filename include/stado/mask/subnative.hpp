#ifndef INCLUDE_STADO_MASK_SUBNATIVE_HPP
#define INCLUDE_STADO_MASK_SUBNATIVE_HPP

#include <bit>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "stado/mask/base.hpp"
#include "stado/mask/operations/part-mask.hpp"

namespace stado {
template<std::size_t tValueBits, std::size_t tSize>
requires(std::has_single_bit(tValueBits) && std::has_single_bit(tSize))
struct SubNativeMask {
  static constexpr std::size_t storage_size = 128 / tValueBits;
  static constexpr std::size_t size = tSize;
  static_assert(storage_size >= size, "All elements have to fit!");
  using Native = Mask<tValueBits, storage_size>;
  using Value = bool;

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
    return part_mask<Native>(size);
  }

  Native native_;
};

template<typename T>
struct IsSubNativeMaskTrait : public std::false_type {};
template<std::size_t tValueBits, std::size_t tSize>
struct IsSubNativeMaskTrait<SubNativeMask<tValueBits, tSize>> : public std::true_type {};
template<typename T>
concept AnySubNativeMask = IsSubNativeMaskTrait<T>::value;

template<std::size_t tValueBits, std::size_t tSize>
inline auto operator&(SubNativeMask<tValueBits, tSize> m1, SubNativeMask<tValueBits, tSize> m2) {
  const auto native = m1.native() & m2.native();
  return SubNativeMask<tValueBits, tSize>::from_native(native);
}
template<std::size_t tValueBits, std::size_t tSize>
inline auto operator|(SubNativeMask<tValueBits, tSize> m1, SubNativeMask<tValueBits, tSize> m2) {
  const auto native = m1.native() | m2.native();
  return SubNativeMask<tValueBits, tSize>::from_native(native);
}
template<std::size_t tValueBits, std::size_t tSize>
inline auto operator~(SubNativeMask<tValueBits, tSize> m) {
  const auto native = ~m.native();
  return SubNativeMask<tValueBits, tSize>::from_native(native);
}

template<std::size_t tValueBits, std::size_t tSize>
static inline bool horizontal_or(SubNativeMask<tValueBits, tSize> m) {
  return horizontal_or(m.masked_native());
}
template<std::size_t tValueBits, std::size_t tSize>
static inline bool horizontal_and(SubNativeMask<tValueBits, tSize> m) {
  return horizontal_and(m.one_extended_native());
}

// Since this header requires both base.hpp and part-mask.hpp, the respective traits
// need to be specialized here.
template<std::size_t tValueBits, std::size_t tSize>
requires(tValueBits* tSize < 128)
struct MaskTrait<tValueBits, tSize> {
  using Type = SubNativeMask<tValueBits, tSize>;
};
template<std::size_t tValueBits, std::size_t tSize>
struct PartMaskCreator<SubNativeMask<tValueBits, tSize>> {
  using Mask = SubNativeMask<tValueBits, tSize>;
  static Mask create(const std::size_t part) {
    assert(part <= tSize);
    return Mask::from_native(part_mask<typename Mask::Native>(part));
  }
};
} // namespace stado

#endif // INCLUDE_STADO_MASK_SUBNATIVE_HPP
