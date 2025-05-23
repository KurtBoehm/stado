#include <array>
#include <cstdint>
#include <cstdlib>

#include <immintrin.h>

#include "benchmark/benchmark.h"

#include "generate_random.hpp"

constexpr std::size_t size = 1024;

[[gnu::noinline]] __m128i extend_u8_u64_sse4_2(__m128i v) {
  return _mm_cvtepu8_epi64(v);
}

[[gnu::noinline]] __m128i extend_u8_u64_sse2a(__m128i v) {
  std::array<std::uint8_t, 16> data{};
  _mm_storeu_si128(reinterpret_cast<__m128i*>(data.data()), v);
  return _mm_setr_epi64(_m_from_int64(data[0]), _m_from_int64(data[1]));
}

[[gnu::noinline]] __m128i extend_u8_u64_sse2b(__m128i v) {
  __m128i ex16 = _mm_unpacklo_epi8(v, _mm_setzero_si128());
  __m128i ex32 = _mm_unpacklo_epi16(ex16, _mm_setzero_si128());
  return _mm_unpacklo_epi32(ex32, _mm_setzero_si128());
}

static void bm_extend_u8_u64_sse4_2(benchmark::State& state) {
  const auto indices = generate<std::uint8_t, size>();

  std::size_t i = 0;
  for (auto _ : state) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices.data() + i));
    auto value = extend_u8_u64_sse4_2(v);
    benchmark::DoNotOptimize(value);
    i = (i + 4) % size;
  }
}

static void bm_extend_u8_u64_sse2a(benchmark::State& state) {
  const auto indices = generate<std::uint8_t, size>();

  std::size_t i = 0;
  for (auto _ : state) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices.data() + i));
    auto value = extend_u8_u64_sse2a(v);
    benchmark::DoNotOptimize(value);
    i = (i + 4) % size;
  }
}

static void bm_extend_u8_u64_sse2b(benchmark::State& state) {
  const auto indices = generate<std::uint8_t, size>();

  std::size_t i = 0;
  for (auto _ : state) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices.data() + i));
    auto value = extend_u8_u64_sse2b(v);
    benchmark::DoNotOptimize(value);
    i = (i + 4) % size;
  }
}

BENCHMARK(bm_extend_u8_u64_sse2a);
BENCHMARK(bm_extend_u8_u64_sse2b);
BENCHMARK(bm_extend_u8_u64_sse4_2);
BENCHMARK_MAIN();
