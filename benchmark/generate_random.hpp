#ifndef TEST_GENERATE_RANDOM_HPP
#define TEST_GENERATE_RANDOM_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "pcg_random.hpp"

template<std::unsigned_integral T, std::size_t tSize, T tLimit = std::numeric_limits<T>::max()>
inline std::array<T, tSize> generate() {
  std::array<T, tSize> indices{};
  pcg64 rng{};
  std::uniform_int_distribution<T> d(0, tLimit);
  std::generate(indices.begin(), indices.end(), [&d, &rng] { return d(rng); });
  return indices;
}

template<std::floating_point T, std::size_t tSize>
inline std::array<T, tSize> generate() {
  std::array<T, tSize> indices{};
  pcg64 rng{};
  std::uniform_real_distribution<T> d{std::numeric_limits<T>::min(), std::numeric_limits<T>::max()};
  std::generate(indices.begin(), indices.end(), [&d, &rng] { return d(rng); });
  return indices;
}

template<std::unsigned_integral T, T tLimit = std::numeric_limits<T>::max()>
inline std::vector<T> generate(std::size_t size) {
  std::vector<T> indices(size);
  pcg64 rng{};
  std::uniform_int_distribution<T> d(0, tLimit);
  std::generate(indices.begin(), indices.end(), [&d, &rng] { return d(rng); });
  return indices;
}

#endif // TEST_GENERATE_RANDOM_HPP
