#ifndef INCLUDE_STADO_MASK_BROAD_BASE_HPP
#define INCLUDE_STADO_MASK_BROAD_BASE_HPP

#include <cstddef>
#include <type_traits>

namespace stado {
template<std::size_t tValueBits, std::size_t tSize>
struct BroadMask;

template<typename T>
struct IsBroadMaskTrait : public std::false_type {};
template<std::size_t tValueBits, std::size_t tSize>
struct IsBroadMaskTrait<BroadMask<tValueBits, tSize>> : public std::true_type {};
template<typename T>
concept AnyBroadMask = IsBroadMaskTrait<T>::value;
} // namespace stado

#endif // INCLUDE_STADO_MASK_BROAD_BASE_HPP
