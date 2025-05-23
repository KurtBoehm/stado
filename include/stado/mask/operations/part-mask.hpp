#ifndef INCLUDE_STADO_MASK_OPERATIONS_PART_MASK_HPP
#define INCLUDE_STADO_MASK_OPERATIONS_PART_MASK_HPP

#include <cstddef>
#include <utility>

#include "stado/defs.hpp"
#include "stado/mask/broad.hpp"
#include "stado/mask/compact.hpp"
#include "stado/mask/single.hpp"
#include "stado/vector/native/base.hpp"

namespace stado {
template<typename T>
struct PartMaskCreator;
template<std::size_t tValueBits, std::size_t tSize>
struct PartMaskCreator<BroadMask<tValueBits, tSize>> {
  using Mask = BroadMask<tValueBits, tSize>;
  using Value = BitInt<tValueBits>;
  using Base = NativeVector<Value, tSize>;

  static Mask create(const std::size_t part) {
    const Base index_vec = []<std::size_t... tIdxs>(std::index_sequence<tIdxs...> /*idxs*/) {
      return Base{Value(tIdxs)...};
    }(std::make_index_sequence<tSize>{});
    return Base{Value(part)} > index_vec;
  }
};
template<std::size_t tSize>
struct PartMaskCreator<CompactMask<tSize>> {
  using Mask = CompactMask<tSize>;
  using Register = typename Mask::Register;

  static Mask create(const std::size_t part) {
    return {Register(~(Register(-1) << part))};
  }
};
template<>
struct PartMaskCreator<SingleMask> {
  using Mask = SingleMask;
  static Mask create(const std::size_t part) {
    return SingleMask{part > 0};
  }
};

template<typename T>
static inline T part_mask(const std::size_t part) {
  return PartMaskCreator<T>::create(part);
}
} // namespace stado

#endif // INCLUDE_STADO_MASK_OPERATIONS_PART_MASK_HPP
