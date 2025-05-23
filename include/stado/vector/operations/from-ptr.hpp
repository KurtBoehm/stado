#ifndef INCLUDE_STADO_VECTOR_OPERATIONS_FROM_PTR_HPP
#define INCLUDE_STADO_VECTOR_OPERATIONS_FROM_PTR_HPP

#include <cstddef>
#include <utility>

namespace stado {
template<typename TVec>
inline TVec create_from_ptr(const typename TVec::Element* data) {
  TVec vector{};
  vector.load(data);
  return vector;
}

template<typename TVec>
inline TVec create_partially_from_ptr(std::size_t part, const typename TVec::Element* data) {
  TVec vector{};
  vector.load_partial(part, data);
  return vector;
}

template<typename TVec>
inline TVec create_index_from(const typename TVec::Element base) {
  using Value = TVec::Element;
  static constexpr auto offsets = []<std::size_t... tIdxs>(std::index_sequence<tIdxs...> /*idxs*/) {
    return TVec{Value(tIdxs)...};
  }(std::make_index_sequence<TVec::size>{});
  return create_from_ptr<TVec>(offsets.data()) + base;
}
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_OPERATIONS_FROM_PTR_HPP
