#include <array>
#include <cstddef>
#include <cstring>
#include <random>

#include "benchmark/benchmark.h"
#include "pcg_extras.hpp"
#include "pcg_random.hpp"

#include "stado/stado.hpp"

using Val = stado::f64;
using Vec = stado::Vector<Val, 2>;

[[gnu::noinline]] auto op1(Vec v, Val* ptr, std::size_t num) {
  v.store_partial(num, ptr);
  return *ptr;
}
[[gnu::noinline]] auto op2(Vec v, Val* ptr, std::size_t num) {
  _mm_maskmoveu_si128(_mm_castpd_si128(v), stado::part_mask<stado::Mask<64, 2>>(num),
                      reinterpret_cast<char*>(ptr));
  return *ptr;
}
[[gnu::noinline]] auto op3(Vec v, Val* ptr, std::size_t num) {
  _mm_maskstore_epi64(reinterpret_cast<long long*>(ptr), stado::part_mask<stado::Mask<64, 2>>(num),
                      _mm_castpd_si128(v));
  return *ptr;
}
[[gnu::noinline]] auto op4(Vec v, Val* ptr, std::size_t num) {
  if (num >= 1) [[likely]] {
    if (num >= 2) [[unlikely]] {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), _mm_castpd_si128(v));
      return *ptr;
    }
    _mm_storeu_si64(ptr, _mm_castpd_si128(v));
  }
  return *ptr;
}
[[gnu::noinline]] auto op5(Vec v, Val* ptr, std::size_t num) {
  switch (num) {
  [[unlikely]] case 0:
    break;
  [[likely]] case 1:
    _mm_storeu_si64(ptr, _mm_castpd_si128(v));
    break;
  [[unlikely]] default:
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), _mm_castpd_si128(v));
    break;
  }
  return *ptr;
}

#define DIST_full std::uniform_int_distribution<stado::u64> uniform_dist(0, Vec::size);
#define DIST_redu std::uniform_int_distribution<stado::u64> uniform_dist(1, Vec::size - 1);
#define DIST_1 std::uniform_int_distribution<stado::u64> uniform_dist(1, 1);

#define BM_OP(NUM, DIST) \
  static void bm_op##NUM##_##DIST(benchmark::State& state) { \
    pcg_extras::seed_seq_from<std::random_device> seed_source; \
    pcg32 rng(seed_source); \
    DIST_##DIST; \
    std::uniform_real_distribution<Val> fdist{}; \
    std::array<Val, Vec::size> arr{}; \
    for (auto _ : state) { \
      Vec v{fdist(rng), fdist(rng)}; \
      benchmark::DoNotOptimize(op##NUM(v, arr.data(), uniform_dist(rng))); \
    } \
  } \
  BENCHMARK(bm_op##NUM##_##DIST);

#define BM_OPS(DIST) \
  BM_OP(1, DIST) \
  BM_OP(2, DIST) \
  BM_OP(3, DIST) \
  BM_OP(4, DIST) \
  BM_OP(5, DIST)

// NOLINTBEGIN
BM_OPS(full)
BM_OPS(redu)
BM_OPS(1)
BENCHMARK_MAIN();
// NOLINTEND
