#ifndef INCLUDE_STADO_DEFS_HPP
#define INCLUDE_STADO_DEFS_HPP

#include <concepts>
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

template<std::size_t tBits>
struct BitIntTrait;
template<>
struct BitIntTrait<8> {
  using Signed = i8;
  using Unsigned = u8;
};
template<>
struct BitIntTrait<16> {
  using Signed = i16;
  using Unsigned = u16;
};
template<>
struct BitIntTrait<32> {
  using Signed = i32;
  using Unsigned = u32;
};
template<>
struct BitIntTrait<64> {
  using Signed = i64;
  using Unsigned = u64;
};
template<std::size_t tBits>
using BitInt = BitIntTrait<tBits>::Signed;
template<std::size_t tBits>
using BitUInt = BitIntTrait<tBits>::Unsigned;

template<typename T>
concept AnyInt8 = std::same_as<T, u8> || std::same_as<T, i8>;
template<typename T>
concept AnyInt16 = std::same_as<T, u16> || std::same_as<T, i16>;
template<typename T>
concept AnyInt32 = std::same_as<T, u32> || std::same_as<T, i32>;
template<typename T>
concept AnyInt64 = std::same_as<T, u64> || std::same_as<T, i64>;
} // namespace stado

#endif // INCLUDE_STADO_DEFS_HPP
