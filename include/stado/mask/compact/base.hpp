#ifndef INCLUDE_STADO_MASK_COMPACT_BASE_HPP
#define INCLUDE_STADO_MASK_COMPACT_BASE_HPP

#include <cstddef>
#include <type_traits>

namespace stado {
template<std::size_t tSize>
struct CompactMask;

template<typename T>
struct IsCompactMaskTrait : public std::false_type {};
template<std::size_t tSize>
struct IsCompactMaskTrait<CompactMask<tSize>> : public std::true_type {};
template<typename T>
concept AnyCompactMask = IsCompactMaskTrait<T>::value;
} // namespace stado

#endif // INCLUDE_STADO_MASK_COMPACT_BASE_HPP
