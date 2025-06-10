#include <array>
#include <cstdint>
#include <cstdlib>

#include <immintrin.h>

#include "benchmark/benchmark.h"

#include "generate_random.hpp"

constexpr std::size_t size = 1024;
using i32 = int32_t;
using u32 = uint32_t;

[[gnu::noinline]] __m128i op1(__m128d v) {
  /* [i32(v[0]), i32(v[1]), -, -] */
  const __m128i vi32 = _mm_cvttpd_epi32(v);
  /* v - [2^31, 2^31, -, -] as f64x2, transforming the range of u32 to that of i32 */
  const __m128d voff = _mm_sub_pd(v, _mm_castsi128_pd(_mm_set1_epi64x(0x41E0000000000000)));
  /* [i32(v[0] - 2^31), i32(v[1] - 2^31), -, -] */
  const __m128i oi32 = _mm_cvttpd_epi32(voff);
  /* sign[i] is true if v[i] < 0 or v[i] in [2^31, 2^32) → we need the offset version */
  const __m128i sign = _mm_srai_epi32(vi32, 31);
  /* mask out the element for which the offset version is required */
  const __m128i mi32 = _mm_and_si128(oi32, sign);
  /* due to modular arithmetic, it suffices to update using bitwise OR */
  return _mm_or_si128(vi32, mi32);
}
[[gnu::noinline]] __m128i op2(__m128d v) {
  const auto v0 = u32(_mm_cvttsd_si64(v));
  const auto v1 = u32(_mm_cvttsd_si64(_mm_shuffle_pd(v, v, 1)));
  return _mm_set_epi32(0, 0, i32(v1), i32(v0));
}

static void bm_op1(benchmark::State& state) {
  const auto vals = generate<double, size>();

  std::size_t i = 0;
  for (auto _ : state) {
    const __m128d v = _mm_loadu_pd(vals.data() + i);
    benchmark::DoNotOptimize(op1(v));
    i = (i + 4) % size;
  }
}

static void bm_op2(benchmark::State& state) {
  const auto vals = generate<double, size>();

  std::size_t i = 0;
  for (auto _ : state) {
    const __m128d v = _mm_loadu_pd(vals.data() + i);
    benchmark::DoNotOptimize(op2(v));
    i = (i + 4) % size;
  }
}

BENCHMARK(bm_op1);
BENCHMARK(bm_op2);
BENCHMARK_MAIN();
