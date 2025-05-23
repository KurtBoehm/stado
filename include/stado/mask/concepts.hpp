#ifndef INCLUDE_STADO_MASK_CONCEPTS_HPP
#define INCLUDE_STADO_MASK_CONCEPTS_HPP

#include <cstddef>

#include "stado/mask/broad/base.hpp"
#include "stado/mask/compact/base.hpp"
#include "stado/mask/single.hpp"
#include "stado/mask/subnative.hpp"

namespace stado {
template<typename T>
concept AnyMask = AnyBroadMask<T> || AnyCompactMask<T> || AnySingleMask<T> || AnySubNativeMask<T>;
template<typename T, std::size_t tSize>
concept SizeMask = AnyMask<T> && T::size == tSize;
} // namespace stado

#endif // INCLUDE_STADO_MASK_CONCEPTS_HPP
