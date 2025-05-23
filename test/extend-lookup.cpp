#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "stado/stado.hpp"

constexpr std::size_t vec_size = 4;
using IdxVec = stado::SubNativeVector<std::uint8_t, vec_size>;
using ExtIdxVec = stado::NativeVector<std::uint64_t, vec_size>;
using ValVec = stado::NativeVector<double, vec_size>;

void print_vector(ValVec v) {
  std::cout << "[" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << "]\n";
}

int main() {
  std::array<double, 8> lookup{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<std::uint8_t, 4> indices{0, 3, 2, 1};

  // auto index = IdxVec::from_values(std::uint8_t{0}, std::uint8_t{3}, std::uint8_t{2},
  //                                             std::uint8_t{1});
  IdxVec index{};
  index.load(indices.data());
  auto index_ex = stado::convert_safe<ExtIdxVec>(index);
  auto value = stado::lookup(index_ex, lookup.data());
  print_vector(value);
}
