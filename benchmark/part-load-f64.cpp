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

[[gnu::noinline]] auto op1(const Val* ptr, std::size_t num) {
  return Vec{}.load_partial(num, ptr);
}
[[gnu::noinline]] auto op2(const Val* ptr, std::size_t num) {
  if (num >= 1) {
    if (num >= 2) [[unlikely]] {
      return _mm_castsi128_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    }
    return _mm_castsi128_pd(_mm_loadu_si64(reinterpret_cast<const __m128i*>(ptr)));
  }
  return _mm_castsi128_pd(_mm_setzero_si128());
}
[[gnu::noinline]] auto op3(const Val* ptr, std::size_t num) {
  std::array<Val, Vec::size> buf{};
  std::memcpy(buf.data(), ptr, 8 * num);
  return Vec{}.load(buf.data());
}
[[gnu::noinline]] auto op4(const Val* ptr, std::size_t num) {
  return Vec{num > 0 ? ptr[0] : 0, num > 1 ? ptr[1] : 0};
}
[[gnu::noinline]] auto op5(const Val* ptr, std::size_t num) {
  return _mm_maskload_pd(ptr, stado::part_mask<stado::Mask<64, 2>>(num));
}

#define DIST_full std::uniform_int_distribution<stado::u64> uniform_dist(0, Vec::size);
#define DIST_redu std::uniform_int_distribution<stado::u64> uniform_dist(1, Vec::size - 1);
#define DIST_1 std::uniform_int_distribution<stado::u64> uniform_dist(1, 1);

#define BM_OP(NUM, DIST) \
  static void bm_op##NUM##_##DIST(benchmark::State& state) { \
    pcg_extras::seed_seq_from<std::random_device> seed_source; \
    pcg32 rng(seed_source); \
    DIST_##DIST; \
    std::array<Val, Vec::size> arr{}; \
    for (auto _ : state) { \
      benchmark::DoNotOptimize(op##NUM(arr.data(), uniform_dist(rng))); \
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
