#include <array>
#include <cstdint>
#include <cstdlib>

#include <immintrin.h>

#include "benchmark/benchmark.h"

#include "generate_random.hpp"

constexpr std::size_t size = 1024;
using i32 = int32_t;
using u64 = uint64_t;

[[gnu::noinline]] __m128i op1(__m256i v) {
  const __m256i idxs8 =
    _mm256_setr_epi8(0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 8, 9, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m256i shuf = _mm256_shuffle_epi8(v, idxs8);
  const __m128i hi128 = _mm256_extracti128_si256(shuf, 1);
  return _mm_unpacklo_epi32(_mm256_castsi256_si128(shuf), hi128);
}
[[gnu::noinline]] __m128i op2(__m256i v) {
  const __m256i blended = _mm256_blend_epi16(v, _mm256_setzero_si256(), 0b11101110);
  const __m128i hu128 = _mm256_extracti128_si256(blended, 1);
  const __m128i vu32 = _mm_packus_epi32(_mm256_castsi256_si128(v), hu128);
  return _mm_packus_epi32(vu32, _mm_setzero_si128());
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
