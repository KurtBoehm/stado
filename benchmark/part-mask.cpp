#include <array>
#include <cstdlib>
#include <random>

#include "benchmark/benchmark.h"
#include "pcg_extras.hpp"
#include "pcg_random.hpp"

#include "stado/stado.hpp"

using Int = stado::i32;
constexpr std::size_t vec_size = 4;
using Vec = stado::NativeVector<Int, vec_size>;
using Mask = stado::Mask<32, vec_size>;

[[gnu::noinline]] Mask part_mask1(std::size_t part) {
  return {part > 0, part > 1, part > 2, part > 3};
}
[[gnu::noinline]] Mask part_mask2(std::size_t part) {
  return stado::part_mask<Mask>(part);
}
[[gnu::noinline]] Mask part_mask3(std::size_t part) {
  Vec mask_raw(static_cast<Int>(part));
  Vec mask_compare(0, 1, 2, 3);
  return mask_raw > mask_compare;
}
auto part_mask4(std::size_t part) {
  const std::array<stado::i8, 32> mask{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1, 0,  0,  0,  0,  0,  0,
                                       0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
  return stado::Vector<stado::i8, 16>{}.load(mask.data() + 16 - part * 4);
}

static void bm_part_mask_over(benchmark::State& state) {
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 rng(seed_source);
  std::uniform_int_distribution<stado::u64> uniform_dist(0, vec_size - 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(uniform_dist(rng));
  }
}

static void bm_part_mask1(benchmark::State& state) {
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 rng(seed_source);
  std::uniform_int_distribution<stado::u64> uniform_dist(0, vec_size - 1);

  for (auto _ : state) {
    auto vec = part_mask1(uniform_dist(rng));
    benchmark::DoNotOptimize(vec);
  }
}

static void bm_part_mask2(benchmark::State& state) {
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 rng(seed_source);
  std::uniform_int_distribution<stado::u64> uniform_dist(0, vec_size - 1);

  for (auto _ : state) {
    auto vec = part_mask2(uniform_dist(rng));
    benchmark::DoNotOptimize(vec);
  }
}

static void bm_part_mask3(benchmark::State& state) {
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 rng(seed_source);
  std::uniform_int_distribution<stado::u64> uniform_dist(0, vec_size - 1);

  for (auto _ : state) {
    auto vec = part_mask3(uniform_dist(rng));
    benchmark::DoNotOptimize(vec);
  }
}

static void bm_part_mask4(benchmark::State& state) {
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 rng(seed_source);
  std::uniform_int_distribution<stado::u64> uniform_dist(0, vec_size - 1);

  for (auto _ : state) {
    auto vec = part_mask4(uniform_dist(rng));
    benchmark::DoNotOptimize(vec);
  }
}

BENCHMARK(bm_part_mask_over);
BENCHMARK(bm_part_mask1);
BENCHMARK(bm_part_mask2);
BENCHMARK(bm_part_mask3);
BENCHMARK(bm_part_mask4);
BENCHMARK_MAIN();
