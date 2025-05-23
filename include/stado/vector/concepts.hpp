#ifndef INCLUDE_STADO_VECTOR_CONCEPTS_HPP
#define INCLUDE_STADO_VECTOR_CONCEPTS_HPP

#include <concepts>
#include <cstddef>

#include "stado/vector/native/base.hpp"
#include "stado/vector/single.hpp"
#include "stado/vector/subnative.hpp"
#include "stado/vector/supernative.hpp"

namespace stado {
template<typename T>
concept AnyVector =
  AnyNativeVector<T> || AnySingleVector<T> || AnySubNativeVector<T> || AnySuperNativeVector<T>;
template<typename T, std::size_t tSize>
concept SizeVector = AnyVector<T> && T::size == tSize;
template<typename T, typename TValue>
concept TypeVector = AnyVector<T> && std::same_as<typename T::Value, TValue>;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_CONCEPTS_HPP
