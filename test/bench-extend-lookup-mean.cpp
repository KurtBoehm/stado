#include <array>
#include <cstdlib>

#include "benchmark/benchmark.h"

#include "stado/stado.hpp"

#include "generate_random.hpp"

constexpr std::size_t vec_size = 4;
constexpr std::size_t lookup_size = 16;
using IdxVec = stado::SubNativeVector<stado::u8, vec_size>;
using ExtIdxVec = stado::NativeVector<stado::u64, vec_size>;
using ValVec = stado::NativeVector<double, vec_size>;

inline ValVec harmonic_mean(const ValVec a, const ValVec b) {
  const ValVec s = a + b;
  return stado::if_div(s != ValVec{0}, 2 * a * b, s);
}

[[gnu::noinline]] void compute_mean(const stado::u8* indices, const double* lookup,
                                    const std::size_t vector_count) {
  for (std::size_t i = 1; i < vector_count; ++i) {
    IdxVec index1{};
    index1.load(indices + i * vec_size);
    auto index_ex1 = stado::convert_safe<ExtIdxVec>(index1);
    auto value1 = stado::lookup(index_ex1, lookup);

    IdxVec index2{};
    index2.load(indices + (i - 1) * vec_size);
    auto index_ex2 = stado::convert_safe<ExtIdxVec>(index2);
    auto value2 = stado::lookup(index_ex2, lookup);

    auto value = harmonic_mean(value1, value2);
    benchmark::DoNotOptimize(value);
  }
}

[[gnu::noinline]] void lookup_mean(const stado::u8* indices, const double* lookup,
                                   const std::size_t vector_count) {
  for (std::size_t i = 1; i < vector_count; ++i) {
    IdxVec index1{};
    index1.load(indices + i * vec_size);
    auto index_ex1 = stado::convert_safe<ExtIdxVec>(index1);
    auto value1 = stado::lookup(index_ex1, lookup);
    benchmark::DoNotOptimize(value1);

    IdxVec index2{};
    index2.load(indices + (i - 1) * vec_size);
    auto index_ex2 = stado::convert_safe<ExtIdxVec>(index2);

    ExtIdxVec index_ex = index_ex1 * lookup_size + index_ex2;
    auto value = stado::lookup(index_ex, lookup);
    benchmark::DoNotOptimize(value);
  }
}

static void bm_compute_mean(benchmark::State& state) {
  const std::size_t size = state.range(0);
  const auto indices = generate<stado::u8, lookup_size>(size * vec_size);
  std::array<double, lookup_size> lookup{};

  for (auto _ : state) {
    compute_mean(indices.data(), lookup.data(), size);
  }
}

static void bm_lookup_mean(benchmark::State& state) {
  const std::size_t size = state.range(0);
  const auto indices = generate<stado::u8, lookup_size>(size * vec_size);
  std::array<double, lookup_size * lookup_size> lookup{};

  for (auto _ : state) {
    lookup_mean(indices.data(), lookup.data(), size);
  }
}

BENCHMARK(bm_compute_mean)->RangeMultiplier(2)->Range(2, 1U << 30U);
BENCHMARK(bm_lookup_mean)->RangeMultiplier(2)->Range(2, 1U << 30U);
BENCHMARK_MAIN();
