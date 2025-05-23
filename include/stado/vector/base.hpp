#ifndef INCLUDE_STADO_VECTOR_BASE_HPP
#define INCLUDE_STADO_VECTOR_BASE_HPP

#include <cstddef>

#include "stado/vector/native/base.hpp"
#include "stado/vector/native/sizes.hpp"
#include "stado/vector/single.hpp"
#include "stado/vector/subnative.hpp"

namespace stado {
template<bool tSingle, bool tSubNative, bool tSuperNative, typename T, std::size_t tSize>
struct VectorTraitImpl;
template<typename T, std::size_t tSize>
struct VectorTraitImpl<false, false, false, T, tSize> {
  using Type = NativeVector<T, tSize>;
};
template<typename T, bool tSubNative, bool tSuperNative>
struct VectorTraitImpl<true, tSubNative, tSuperNative, T, 1> {
  using Type = SingleVector<T>;
};
template<typename T, std::size_t tSize>
struct VectorTraitImpl<false, true, false, T, tSize> {
  using Type = SubNativeVector<T, tSize>;
};
#if false
template<typename T, std::size_t tSize>
struct VectorTraitImpl<false, false, true, T, tSize> {
  using Type = SuperNativeVector<T, tSize>;
};
#endif

template<typename T, std::size_t tSize>
struct VectorTrait {
  using Type = typename VectorTraitImpl<(tSize == 1), (tSize < native_sizes<T>.front()),
                                        (tSize > native_sizes<T>.back()), T, tSize>::Type;
};

template<typename T, std::size_t tSize>
using Vector = typename VectorTrait<T, tSize>::Type;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_BASE_HPP
