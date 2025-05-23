#include <array>
#include <cstdint>
#include <cstdlib>

#include <immintrin.h>
#include <xmmintrin.h>

#include "benchmark/benchmark.h"

#include "generate_random.hpp"

constexpr std::size_t size = 1024;
constexpr std::size_t lookup_size = 4;

#ifdef __AVX__

static void bm_lookup4_fl32x4_avx(benchmark::State& state) {
  const auto indices = generate<std::uint32_t, size, lookup_size>();
  std::array<float, lookup_size> lookup{};
  __m128 lookup_vec = _mm_loadu_ps(lookup.data());

  std::size_t i = 0;
  for (auto _ : state) {
    auto value = _mm_permutevar_ps(
      lookup_vec, _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(indices.data() + i)));
    benchmark::DoNotOptimize(value);
    i = (i + 4) % size;
  }
}

#endif

static void bm_lookup4_fl32x4_sse_mmx(benchmark::State& state) {
  const auto indices = generate<std::uint32_t, size, lookup_size>();
  std::array<float, lookup_size> lookup{};
  __m128 lookup_vec = _mm_loadu_ps(lookup.data());

  std::size_t i = 0;
  for (auto _ : state) {
    std::array<int32_t, 4> ii{};
    std::array<float, 4> ff{};
    __m128i index = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices.data() + i));
    _mm_storeu_ps(ff.data(), lookup_vec);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ii.data()), index);
    __m128 r01 = _mm_loadh_pi(_mm_load_ss(lookup.data() + ii[0]),
                              reinterpret_cast<const __m64*>(lookup.data() + ii[1]));
    __m128 r23 = _mm_loadh_pi(_mm_load_ss(lookup.data() + ii[2]),
                              reinterpret_cast<const __m64*>(lookup.data() + ii[3]));
    auto value = _mm_shuffle_ps(r01, r23, 0x88);

    benchmark::DoNotOptimize(value);
    i = (i + 4) % size;
  }
}

static void bm_lookup4_fl32x4_sse(benchmark::State& state) {
  const auto indices = generate<std::uint32_t, size, lookup_size>();
  std::array<float, lookup_size> lookup{};
  __m128 lookup_vec = _mm_loadu_ps(lookup.data());

  std::size_t i = 0;
  for (auto _ : state) {
    std::array<int32_t, 4> ii{};
    std::array<float, 4> ff{};
    __m128i index = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(indices.data() + i));
    _mm_storeu_ps(ff.data(), lookup_vec);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ii.data()), index);
    __m128 value = _mm_setr_ps(ff[ii[0]], ff[ii[1]], ff[ii[2]], ff[ii[3]]);

    benchmark::DoNotOptimize(value);
    i = (i + 4) % size;
  }
}

#ifdef __AVX2__

static void bm_lookup_fl32x4_avx2(benchmark::State& state) {
  const auto indices = generate<std::uint32_t, size, lookup_size>();
  std::array<float, lookup_size> lookup{};

  std::size_t i = 0;
  for (auto _ : state) {
    auto value = _mm_i32gather_ps(
      lookup.data(), _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(indices.data() + i)), 4);
    benchmark::DoNotOptimize(value);
    i = (i + 4) % size;
  }
}

#endif

static void bm_lookup_fl32x4_sse2(benchmark::State& state) {
  const auto indices = generate<std::uint32_t, size, lookup_size>();
  std::array<float, lookup_size> lookup{};

  std::size_t i = 0;
  for (auto _ : state) {
    std::array<std::uint32_t, 4> ii{};
    __m128i index = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(indices.data() + i));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ii.data()), index);
    __m128 value = _mm_setr_ps(lookup[ii[0]], lookup[ii[1]], lookup[ii[2]], lookup[ii[3]]);
    benchmark::DoNotOptimize(value);
    i = (i + 4) % size;
  }
}

#ifdef __AVX__
BENCHMARK(bm_lookup4_fl32x4_avx);
#endif
BENCHMARK(bm_lookup4_fl32x4_sse);
BENCHMARK(bm_lookup4_fl32x4_sse_mmx);
#ifdef __AVX2__
BENCHMARK(bm_lookup_fl32x4_avx2);
#endif
BENCHMARK(bm_lookup_fl32x4_sse2);
BENCHMARK_MAIN();
