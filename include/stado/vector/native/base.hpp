#ifndef INCLUDE_STADO_VECTOR_NATIVE_BASE_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_BASE_HPP

#include <cstddef>

namespace stado {
template<typename T, std::size_t tSize>
struct NativeVector;

template<typename T, std::size_t tSize>
struct NativeVectorBase {
  using Element = T;
  static constexpr std::size_t size = tSize;
};
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_BASE_HPP
