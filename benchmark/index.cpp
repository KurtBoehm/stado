#include <array>
#include <cstdlib>
#include <random>

#include "benchmark/benchmark.h"
#include "pcg_extras.hpp"
#include "pcg_random.hpp"

#include "stado/stado.hpp"

constexpr std::size_t vec_size = 4;
using IdxVec = stado::NativeVector<stado::u64, vec_size>;

[[gnu::noinline]] IdxVec index_vector1(stado::u64 base) {
  return {base, base + 1, base + 2, base + 3};
}

[[gnu::noinline]] IdxVec index_vector2(stado::u64 base) {
  std::array<stado::u64, 4> data{base, base + 1, base + 2, base + 3};
  return stado::create_from_ptr<IdxVec>(data.data());
}

[[gnu::noinline]] IdxVec index_vector3(stado::u64 base) {
  return IdxVec(0, 1, 2, 3) + base;
}

[[gnu::noinline]] IdxVec index_vector4(stado::u64 base) {
  std::array<stado::u64, 4> data{0, 1, 2, 3};
  return stado::create_from_ptr<IdxVec>(data.data()) + base;
}

static void bm_index_vector1(benchmark::State& state) {
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 rng(seed_source);
  std::uniform_int_distribution<stado::u64> uniform_dist{};

  for (auto _ : state) {
    auto vec = index_vector1(uniform_dist(rng));
    benchmark::DoNotOptimize(vec);
  }
}

static void bm_index_vector2(benchmark::State& state) {
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 rng(seed_source);
  std::uniform_int_distribution<stado::u64> uniform_dist{};

  for (auto _ : state) {
    auto vec = index_vector2(uniform_dist(rng));
    benchmark::DoNotOptimize(vec);
  }
}

static void bm_index_vector3(benchmark::State& state) {
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 rng(seed_source);
  std::uniform_int_distribution<stado::u64> uniform_dist{};

  for (auto _ : state) {
    auto vec = index_vector3(uniform_dist(rng));
    benchmark::DoNotOptimize(vec);
  }
}

static void bm_index_vector4(benchmark::State& state) {
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 rng(seed_source);
  std::uniform_int_distribution<stado::u64> uniform_dist{};

  for (auto _ : state) {
    auto vec = index_vector4(uniform_dist(rng));
    benchmark::DoNotOptimize(vec);
  }
}

BENCHMARK(bm_index_vector1);
BENCHMARK(bm_index_vector2);
BENCHMARK(bm_index_vector3);
BENCHMARK(bm_index_vector4);
BENCHMARK_MAIN();
