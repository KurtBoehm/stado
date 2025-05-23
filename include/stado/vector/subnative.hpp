#ifndef INCLUDE_STADO_VECTOR_SUBNATIVE_HPP
#define INCLUDE_STADO_VECTOR_SUBNATIVE_HPP

#include <array>
#include <cassert>
#include <climits>
#include <cstddef>
#include <type_traits>

#include "stado/mask/subnative.hpp"
#include "stado/vector/native/base.hpp"
#include "stado/vector/native/sizes.hpp"

namespace stado {
template<typename T, std::size_t tSize>
struct SubNativeVector {
  static constexpr std::size_t storage_size = native_sizes<T>.front();
  static constexpr std::size_t size = tSize;
  static_assert(storage_size % size == 0,
                "Only element counts that are powers of two are supported!");
  using Value = T;
  using Native = NativeVector<T, storage_size>;
  using Register = typename Native::Register;

  SubNativeVector() = default;

  explicit SubNativeVector(T value) : native_(value) {}
  // TODO Only exists for compatibility’s sake
  template<typename... TArgs>
  requires(sizeof...(TArgs) == size)
  explicit SubNativeVector(TArgs&&... args)
      : SubNativeVector(std::array<T, storage_size>{std::forward<TArgs>(args)...}) {}

  template<typename... TArgs>
  static SubNativeVector from_values(TArgs&&... args) {
    static_assert(sizeof...(TArgs) == size);
    return SubNativeVector(std::array<T, storage_size>{std::forward<TArgs>(args)...});
  }

  static SubNativeVector from_register(Register reg) {
    return SubNativeVector(reg);
  }

  static SubNativeVector from_native(Native native) {
    return SubNativeVector(native);
  }

  SubNativeVector& load(const T* data) {
    native_.template load_partial<size>(data);
    return *this;
  }

  SubNativeVector& load_partial(std::size_t part, const T* data) {
    native_.load_partial(part, data);
    return *this;
  }

  void store(T* ptr) const {
    native_.store_partial(size, ptr);
  }

  void store_partial(std::size_t part, T* ptr) const {
    assert(part < size);
    native_.store_partial(part, ptr);
  }

  SubNativeVector cutoff(std::size_t part) {
    assert(part < size);
    return from_native(native_.cutoff(part));
  }

  SubNativeVector& insert(std::size_t index, T value) {
    assert(index < size);
    native_.insert(index, value);
    return *this;
  }

  SubNativeVector operator-() const {
    return from_native(-native_);
  }

  Native& native() {
    return native_;
  }
  const Native& native() const {
    return native_;
  }

private:
  explicit SubNativeVector(Register reg) : native_(reg) {}
  explicit SubNativeVector(Native native) : native_(native) {}
  explicit SubNativeVector(const std::array<T, storage_size>& full_data) {
    native_.load(full_data.data());
  }

  Native native_{};
};

#define STADO_ARITH(OP) \
  template<typename T, std::size_t tSize> \
  inline SubNativeVector<T, tSize> operator OP(SubNativeVector<T, tSize> v1, \
                                               SubNativeVector<T, tSize> v2) { \
    return SubNativeVector<T, tSize>::from_native(v1.native() OP v2.native()); \
  } \
  template<typename T, std::size_t tSize> \
  inline SubNativeVector<T, tSize> operator OP(T v1, SubNativeVector<T, tSize> v2) { \
    return SubNativeVector<T, tSize>::from_native(v1 OP v2.native()); \
  } \
  template<typename T, std::size_t tSize> \
  inline SubNativeVector<T, tSize> operator OP(SubNativeVector<T, tSize> v1, T v2) { \
    return SubNativeVector<T, tSize>::from_native(v1 OP v2.native()); \
  } \
  template<typename T, std::size_t tSize> \
  inline SubNativeVector<T, tSize>& operator OP##=(SubNativeVector<T, tSize>& v1, \
                                                   SubNativeVector<T, tSize> v2) { \
    return v1 = v1 OP v2; \
  }

STADO_ARITH(+)
STADO_ARITH(-)
STADO_ARITH(*)

#undef STADO_ARITH

#define STADO_LOGIC(OP) \
  template<typename T, std::size_t tSize> \
  inline auto operator OP(SubNativeVector<T, tSize> v1, SubNativeVector<T, tSize> v2) { \
    const auto native = v1.native() OP v2.native(); \
    return SubNativeMask<CHAR_BIT * sizeof(T), tSize>::from_native(native); \
  } \
  template<typename T, std::size_t tSize> \
  inline auto operator OP(T v1, SubNativeVector<T, tSize> v2) { \
    const auto native = v1 OP v2.native(); \
    return SubNativeMask<CHAR_BIT * sizeof(T), tSize>::from_native(native); \
  } \
  template<typename T, std::size_t tSize> \
  inline auto operator OP(SubNativeVector<T, tSize> v1, T v2) { \
    const auto native = v1.native() OP v2; \
    return SubNativeMask<CHAR_BIT * sizeof(T), tSize>::from_native(native); \
  }

STADO_LOGIC(==)
STADO_LOGIC(!=)
STADO_LOGIC(<)
STADO_LOGIC(>)
STADO_LOGIC(<=)
STADO_LOGIC(>=)

template<std::size_t tValueBits, typename T, std::size_t tSize>
requires(tValueBits == CHAR_BIT * sizeof(T))
inline auto select(SubNativeMask<tValueBits, tSize> mask, SubNativeVector<T, tSize> v1,
                   SubNativeVector<T, tSize> v2) {
  const auto native = select(mask.native(), v1.native(), v2.native());
  return SubNativeVector<T, tSize>::from_native(native);
}

template<std::size_t tValueBits, typename T, std::size_t tSize>
requires(tValueBits == CHAR_BIT * sizeof(T))
inline auto if_add(SubNativeMask<tValueBits, tSize> mask, SubNativeVector<T, tSize> v1,
                   SubNativeVector<T, tSize> v2) {
  const auto native = if_add(mask.native(), v1.native(), v2.native());
  return SubNativeVector<T, tSize>::from_native(native);
}

template<std::size_t tValueBits, typename T, std::size_t tSize>
requires(tValueBits == CHAR_BIT * sizeof(T))
inline auto if_div(SubNativeMask<tValueBits, tSize> mask, SubNativeVector<T, tSize> v1,
                   SubNativeVector<T, tSize> v2) {
  const auto native = if_div(mask.masked_native(), v1.native(), v2.native());
  return SubNativeVector<T, tSize>::from_native(native);
}

#define STADO_IS(op) \
  template<typename T, std::size_t tSize> \
  inline auto is_##op(SubNativeVector<T, tSize> v) { \
    using Mask = SubNativeMask<CHAR_BIT * sizeof(T), tSize>; \
    auto nat = v.native(); \
    nat.cutoff(tSize); \
    return Mask::from_native(is_##op(nat)); \
  }

STADO_IS(finite)
STADO_IS(inf)
STADO_IS(nan)
STADO_IS(subnormal)
STADO_IS(zero_or_subnormal)

#undef STADO_IS

template<typename T, std::size_t tSize>
inline T horizontal_add(const SubNativeVector<T, tSize> v) {
  auto nat = v.native();
  nat.cutoff(tSize);
  return horizontal_add(nat);
}

template<typename T1, std::size_t tSize, typename T2>
inline auto lookup(SubNativeVector<T1, tSize> idxs, const T2* ptr) {
  const auto native = lookup_part(idxs.native(), ptr, tSize);
  return SubNativeVector<T2, tSize>::from_native(native);
}

template<typename T>
struct IsSubNativeVectorTrait : public std::false_type {};
template<typename T, std::size_t tSize>
struct IsSubNativeVectorTrait<SubNativeVector<T, tSize>> : public std::true_type {};
template<typename T>
concept AnySubNativeVector = IsSubNativeVectorTrait<T>::value;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_SUBNATIVE_HPP
