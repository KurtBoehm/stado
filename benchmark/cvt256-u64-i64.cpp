#include <array>
#include <cstdint>
#include <cstdlib>

#include <immintrin.h>

#include "benchmark/benchmark.h"

#include "generate_random.hpp"

constexpr std::size_t size = 1024;
using i64 = int64_t;
using u64 = uint64_t;
using f64 = double;

[[gnu::noinline]] __m256d op1(__m256i v) {
  const f64 v0 = _mm256_extract_epi64(v, 0);
  const f64 v1 = _mm256_extract_epi64(v, 1);
  const f64 v2 = _mm256_extract_epi64(v, 2);
  const f64 v3 = _mm256_extract_epi64(v, 3);
  return _mm256_setr_pd(v0, v1, v2, v3);
}
[[gnu::noinline]] __m256d op2(__m256i v) {
  const __m256i lu32 = _mm256_blend_epi32(v, _mm256_setzero_si256(), 0b10101010);
  const __m256d lf64 =
    _mm256_castsi256_pd(_mm256_or_si256(lu32, _mm256_set1_epi64x(0x4330000000000000)));
  const __m256i hu32 = _mm256_srli_epi64(v, 32);
  const __m256d hf64 =
    _mm256_castsi256_pd(_mm256_or_si256(hu32, _mm256_set1_epi64x(0x4530000000000000)));
  const __m256d hsub =
    _mm256_sub_pd(hf64, _mm256_castsi256_pd(_mm256_set1_epi64x(0x4530000000100000)));
  return _mm256_add_pd(lf64, hsub);
}

static void bm_op1(benchmark::State& state) {
  const auto vals = generate<u64, size>();

  std::size_t i = 0;
  for (auto _ : state) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals.data() + i));
    benchmark::DoNotOptimize(op1(v));
    i = (i + 4) % size;
  }
}

static void bm_op2(benchmark::State& state) {
  const auto vals = generate<u64, size>();

  std::size_t i = 0;
  for (auto _ : state) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals.data() + i));
    benchmark::DoNotOptimize(op2(v));
    i = (i + 4) % size;
  }
}

BENCHMARK(bm_op1);
BENCHMARK(bm_op2);
BENCHMARK_MAIN();
