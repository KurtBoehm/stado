#include <cstddef>
#include <cstring>
#include <random>

#include "benchmark/benchmark.h"
#include "pcg_extras.hpp"
#include "pcg_random.hpp"

#include "stado/stado.hpp"

using Val = stado::i8;
using Vec = stado::Vector<Val, 16>;

[[gnu::noinline]] auto op1(Vec v, std::size_t i) {
  return v.extract(i);
}
[[gnu::noinline]] auto op2(Vec v, std::size_t i) {
  return _mm_extract_epi8(_mm_shuffle_epi8(v, _mm_set1_epi8(i)), 0);
}

#define DIST_full std::uniform_int_distribution<stado::u64> uniform_dist(0, Vec::size - 1);
#define DIST_1 auto uniform_dist = [](const auto&) { return 1; };
#define DIST_2 auto uniform_dist = [](const auto&) { return 2; };
#define DIST_3 auto uniform_dist = [](const auto&) { return 3; };
#define DIST_4 auto uniform_dist = [](const auto&) { return 4; };
#define DIST_5 auto uniform_dist = [](const auto&) { return 5; };
#define DIST_6 auto uniform_dist = [](const auto&) { return 6; };
#define DIST_7 auto uniform_dist = [](const auto&) { return 7; };
#define DIST_8 auto uniform_dist = [](const auto&) { return 8; };
#define DIST_9 auto uniform_dist = [](const auto&) { return 9; };
#define DIST_10 auto uniform_dist = [](const auto&) { return 10; };
#define DIST_11 auto uniform_dist = [](const auto&) { return 11; };
#define DIST_12 auto uniform_dist = [](const auto&) { return 12; };
#define DIST_13 auto uniform_dist = [](const auto&) { return 13; };
#define DIST_14 auto uniform_dist = [](const auto&) { return 14; };
#define DIST_15 auto uniform_dist = [](const auto&) { return 15; };

#define BM_OP(NUM, DIST) \
  static void bm_op##NUM##_##DIST(benchmark::State& state) { \
    pcg_extras::seed_seq_from<std::random_device> seed_source; \
    pcg32 rng(seed_source); \
    DIST_##DIST; \
    std::uniform_int_distribution<Val> fdist{}; \
    for (auto _ : state) { \
      Vec v{fdist(rng), fdist(rng), fdist(rng), fdist(rng), fdist(rng), fdist(rng), \
            fdist(rng), fdist(rng), fdist(rng), fdist(rng), fdist(rng), fdist(rng), \
            fdist(rng), fdist(rng), fdist(rng), fdist(rng)}; \
      benchmark::DoNotOptimize(op##NUM(v, uniform_dist(rng))); \
    } \
  } \
  BENCHMARK(bm_op##NUM##_##DIST);

#define BM_OPS(DIST) \
  BM_OP(1, DIST) \
  BM_OP(2, DIST)

// NOLINTBEGIN
BM_OPS(full)
BM_OPS(1)
BM_OPS(2)
BM_OPS(3)
BM_OPS(4)
BM_OPS(5)
BM_OPS(6)
BM_OPS(7)
BM_OPS(8)
BM_OPS(9)
BM_OPS(10)
BM_OPS(11)
BM_OPS(12)
BM_OPS(13)
BM_OPS(14)
BM_OPS(15)
BENCHMARK_MAIN();
// NOLINTEND
