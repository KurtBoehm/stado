#include <array>
#include <cstdint>
#include <cstdlib>

#include <immintrin.h>

#include "benchmark/benchmark.h"

#include "stado/stado.hpp"

#include "generate_random.hpp"

constexpr std::size_t size = 1024;
using i32 = int32_t;
using u32 = uint32_t;

[[gnu::noinline]] __m128i op1(__m128i v) {
  // return _mm_packus_epi16(_mm_packus_epi16(v, _mm_setzero_si128()), _mm_setzero_si128());
  return _mm_packs_epi16(v, _mm_setzero_si128());
}
[[gnu::noinline]] __m128i op2(__m128i v) {
  return _mm_shuffle_epi8(
    v, _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
}

static void bm_op1(benchmark::State& state) {
  const auto vals = generate<u32, size>();

  std::size_t i = 0;
  for (auto _ : state) {
    const __m128i v = stado::Vector<i32, 4>{
      -i32(vals[i] < u32(0x7FFFFFFF)),
      -i32(vals[i + 1] < u32(0x7FFFFFFF)),
      -i32(vals[i + 2] < u32(0x7FFFFFFF)),
      -i32(vals[i + 3] < u32(0x7FFFFFFF)),
    };
    benchmark::DoNotOptimize(op1(v));
    i = (i + 4) % size;
  }
}

static void bm_op2(benchmark::State& state) {
  const auto vals = generate<u32, size>();

  std::size_t i = 0;
  for (auto _ : state) {
    const __m128i v = stado::Vector<i32, 4>{
      -i32(vals[i] < u32(0x7FFFFFFF)),
      -i32(vals[i + 1] < u32(0x7FFFFFFF)),
      -i32(vals[i + 2] < u32(0x7FFFFFFF)),
      -i32(vals[i + 3] < u32(0x7FFFFFFF)),
    };
    benchmark::DoNotOptimize(op2(v));
    i = (i + 4) % size;
  }
}

BENCHMARK(bm_op1);
BENCHMARK(bm_op2);
BENCHMARK_MAIN();
