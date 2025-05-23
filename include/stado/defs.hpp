#ifndef INCLUDE_STADO_DEFS_HPP
#define INCLUDE_STADO_DEFS_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace stado {
using f32 = float;
using f64 = double;
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

template<typename T, std::size_t tSize = 1>
struct IsCompleteTrait : public std::false_type {};
template<typename T>
struct IsCompleteTrait<T, sizeof(T) / sizeof(T)> : public std::true_type {};
template<typename T>
concept CompleteType = IsCompleteTrait<T>::value;

template<typename TFrom, typename TTo>
struct ElementConvertTrait;
template<typename T>
struct ElementConvertTrait<T, T> {
  static constexpr bool is_safe = true;
  static constexpr T convert(T v) {
    return v;
  }
};
template<typename TTo, typename TFrom>
requires(CompleteType<ElementConvertTrait<TFrom, TTo>> && ElementConvertTrait<TFrom, TTo>::is_safe)
inline TTo convert_safe(TFrom from) {
  return ElementConvertTrait<TFrom, TTo>::convert(from);
}
template<typename TTo, typename TFrom>
requires(CompleteType<ElementConvertTrait<TFrom, TTo>>)
inline TTo convert_unsafe(TFrom from) {
  return ElementConvertTrait<TFrom, TTo>::convert(from);
}
} // namespace stado

#endif // INCLUDE_STADO_DEFS_HPP
