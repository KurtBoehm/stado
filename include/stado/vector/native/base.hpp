#ifndef INCLUDE_STADO_VECTOR_NATIVE_BASE_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_BASE_HPP

#include <cstddef>
#include <type_traits>

namespace stado {
template<typename T, std::size_t tSize>
struct NativeVector;

template<typename T, std::size_t tSize>
struct NativeVectorBase {
  using Value = T;
  static constexpr std::size_t size = tSize;
};

template<typename T>
struct IsNativeVectorTrait : public std::false_type {};
template<typename T, std::size_t tSize>
struct IsNativeVectorTrait<NativeVector<T, tSize>> : public std::true_type {};
template<typename T>
concept AnyNativeVector = IsNativeVectorTrait<T>::value;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_BASE_HPP
