#include <array>
#include <cstdint>
#include <cstdlib>

#include <benchmark/benchmark.h>

#include "stado/stado.hpp"

#include "generate_random.hpp"

constexpr std::size_t vec_size = 4;
constexpr std::size_t lookup_size = 16;
using IdxVec = stado::SubNativeVector<std::uint8_t, vec_size>;
using ExtIdxVec = stado::NativeVector<std::uint64_t, vec_size>;
using ValVec = stado::NativeVector<double, vec_size>;

[[gnu::noinline]] void lookup_vec(const std::uint8_t* indices, const double* lookup,
                                  const std::size_t vector_count) {
  for (std::size_t i = 0; i < vector_count; ++i) {
    IdxVec index{};
    index.load(indices + i * vec_size);
    auto index_ex = stado::convert_safe<ExtIdxVec>(index);
    auto value = stado::lookup(index_ex, lookup);
    benchmark::DoNotOptimize(value);
  }
}

[[gnu::noinline]] void lookup_seq(const std::uint8_t* indices, const double* lookup,
                                  const std::size_t vector_count) {
  const std::size_t size = vector_count * vec_size;
  for (std::size_t i = 0; i < size; ++i) {
    auto value = lookup[indices[i]];
    benchmark::DoNotOptimize(value);
  }
}

static void bm_lookup_vector(benchmark::State& state) {
  const std::size_t size = state.range(0);
  const auto indices = generate<std::uint8_t, lookup_size>(size * vec_size);
  std::array<double, lookup_size> lookup{};

  for (auto _ : state) {
    lookup_vec(indices.data(), lookup.data(), size);
  }
}

static void bm_lookup_seq(benchmark::State& state) {
  const std::size_t size = state.range(0);
  const auto indices = generate<std::uint8_t, lookup_size>(size * vec_size);
  std::array<double, lookup_size> lookup{};

  for (auto _ : state) {
    lookup_seq(indices.data(), lookup.data(), size);
  }
}

BENCHMARK(bm_lookup_vector)->RangeMultiplier(2)->Range(2, 1U << 30U);
BENCHMARK(bm_lookup_seq)->RangeMultiplier(2)->Range(2, 1U << 30U);
BENCHMARK_MAIN();
