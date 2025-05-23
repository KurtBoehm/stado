#ifndef INCLUDE_STADO_MASK_OPERATIONS_HORIZONTAL_HPP
#define INCLUDE_STADO_MASK_OPERATIONS_HORIZONTAL_HPP

#include <concepts>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"

namespace stado {
// horizontal min/max of vector elements
// implemented with universal template, works for all vector types:
template<typename TVec>
auto horizontal_min_nan(const TVec x) {
  if constexpr (std::floating_point<typename TVec::Value>) {
    // T is a float or double vector
    if (horizontal_or(is_nan(x))) {
      // check for NAN because min does not guarantee NAN propagation
      return x[horizontal_find_first(is_nan(x))];
    }
  }
  return horizontal_min(x);
}

// WARNING Does not propagate NaN!
template<typename TVec>
auto horizontal_min(const TVec x) {
  if constexpr (std::same_as<typename TVec::Value, bool>) {
    // boolean vector type
    return horizontal_and(x);
  } else if constexpr (sizeof(TVec) >= 32) {
    // split recursively into smaller vectors
    return horizontal_min(min(x.get_low(), x.get_high()));
  } else if constexpr (TVec::size == 2) {
    TVec a = permute2<1, any>(x); // high half
    TVec b = min(a, x);
    return b[0];
  } else if constexpr (TVec::size == 4) {
    TVec a = permute4<2, 3, any, any>(x); // high half
    TVec b = min(a, x);
    a = permute4<1, any, any, any>(b);
    b = min(a, b);
    return b[0];
  } else if constexpr (TVec::size == 8) {
    TVec a = permute8<4, 5, 6, 7, any, any, any, any>(x); // high half
    TVec b = min(a, x);
    a = permute8<2, 3, any, any, any, any, any, any>(b);
    b = min(a, b);
    a = permute8<1, any, any, any, any, any, any, any>(b);
    b = min(a, b);
    return b[0];
  } else {
    static_assert(TVec::size == 16); // no other size is allowed
    TVec a = permute16<8, 9, 10, 11, 12, 13, 14, 15, any, any, any, any, any, any, any, any>(
      x); // high half
    TVec b = min(a, x);
    a = permute16<4, 5, 6, 7, any, any, any, any, any, any, any, any, any, any, any, any>(b);
    b = min(a, b);
    a = permute16<2, 3, any, any, any, any, any, any, any, any, any, any, any, any, any, any>(b);
    b = min(a, b);
    a = permute16<1, any, any, any, any, any, any, any, any, any, any, any, any, any, any, any>(b);
    b = min(a, b);
    return b[0];
  }
}

template<typename TVec>
auto horizontal_max_nan(const TVec x) {
  if constexpr (std::floating_point<typename TVec::Value>) {
    // T is a float or double vector
    if (horizontal_or(is_nan(x))) {
      // check for NAN because max does not guarantee NAN propagation
      return x[horizontal_find_first(is_nan(x))];
    }
  }
  return horizontal_max(x);
}

// WARNING Does not propagate NaN!
template<typename TVec>
auto horizontal_max(const TVec x) {
  if constexpr (std::same_as<typename TVec::Value, bool>) {
    // boolean vector type
    return horizontal_or(x);
  } else if constexpr (sizeof(TVec) >= 32) {
    // split recursively into smaller vectors
    return horizontal_max(max(x.get_low(), x.get_high()));
  } else if constexpr (TVec::size == 2) {
    TVec a = permute2<1, any>(x); // high half
    TVec b = max(a, x);
    return b[0];
  } else if constexpr (TVec::size == 4) {
    TVec a = permute4<2, 3, any, any>(x); // high half
    TVec b = max(a, x);
    a = permute4<1, any, any, any>(b);
    b = max(a, b);
    return b[0];
  } else if constexpr (TVec::size == 8) {
    TVec a = permute8<4, 5, 6, 7, any, any, any, any>(x); // high half
    TVec b = max(a, x);
    a = permute8<2, 3, any, any, any, any, any, any>(b);
    b = max(a, b);
    a = permute8<1, any, any, any, any, any, any, any>(b);
    b = max(a, b);
    return b[0];
  } else {
    static_assert(TVec::size == 16); // no other size is allowed
    TVec a = permute16<8, 9, 10, 11, 12, 13, 14, 15, any, any, any, any, any, any, any, any>(
      x); // high half
    TVec b = max(a, x);
    a = permute16<4, 5, 6, 7, any, any, any, any, any, any, any, any, any, any, any, any>(b);
    b = max(a, b);
    a = permute16<2, 3, any, any, any, any, any, any, any, any, any, any, any, any, any, any>(b);
    b = max(a, b);
    a = permute16<1, any, any, any, any, any, any, any, any, any, any, any, any, any, any, any>(b);
    b = max(a, b);
    return b[0];
  }
}

// Find first element that is true in a boolean vector
template<typename TVec>
static inline int horizontal_find_first(const TVec x) {
  static_assert(std::same_as<typename TVec::Value, bool>);
  auto bits = to_bits(x); // convert to bits
  if (bits == 0) {
    return -1;
  }
  if constexpr (TVec::size < 32) {
    return bit_scan_forward(u32(bits));
  } else {
    return bit_scan_forward(bits);
  }
}

// Count the number of elements that are true in a boolean vector
template<typename TVec>
static inline int horizontal_count(const TVec x) {
  static_assert(std::same_as<typename TVec::Value, bool>);
  auto bits = to_bits(x); // convert to bits
  if constexpr (TVec::size < 32) {
    return vml_popcnt(u32(bits));
  } else {
    return int(vml_popcnt(bits));
  }
}

// maximum and minimum functions. This version is sure to propagate NANs,
// conforming to the new IEEE-754 2019 standard
template<typename TVec>
static inline TVec maximum(const TVec a, const TVec b) {
  if constexpr (std::floating_point<typename TVec::Value>) {
    // float or double vector
    TVec y = select(is_nan(a), a, max(a, b));
#ifdef SIGNED_ZERO // pedantic about signed zero
    y = select(a == b, a & b, y); // maximum(+0, -0) = +0
#endif
    return y;
  } else {
    // integer type
    return max(a, b);
  }
}

template<typename TVec>
static inline TVec minimum(const TVec a, const TVec b) {
  if constexpr (std::floating_point<typename TVec::Value>) {
    // float or double vector
    TVec y = select(is_nan(a), a, min(a, b));
#ifdef SIGNED_ZERO // pedantic about signed zero
    y = select(a == b, a | b, y); // minimum(+0, -0) = -0
#endif
    return y;
  } else {
    // integer type
    return min(a, b);
  }
}
} // namespace stado

#endif // INCLUDE_STADO_MASK_OPERATIONS_HORIZONTAL_HPP
