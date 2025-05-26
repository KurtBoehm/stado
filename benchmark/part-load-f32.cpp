#include <array>
#include <cstddef>
#include <cstring>
#include <random>

#include "benchmark/benchmark.h"
#include "pcg_extras.hpp"
#include "pcg_random.hpp"

#include "stado/stado.hpp"

using Val = stado::f32;
using Vec = stado::Vector<Val, 4>;

[[gnu::noinline]] auto op1(const Val* ptr, std::size_t num) {
  return Vec{}.load_partial(num, ptr);
}
[[gnu::noinline]] auto op2(const Val* ptr, std::size_t num) {
  if (num >= 2) {
    if (num >= 4) [[unlikely]] {
      return _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    }
    __m128i out = _mm_loadu_si64(reinterpret_cast<const __m128i*>(ptr));
    if (num == 3) {
      out = _mm_castps_si128(
        _mm_movelh_ps(_mm_castsi128_ps(out), _mm_loadu_ps(reinterpret_cast<const Val*>(ptr + 2))));
    }
    return _mm_castsi128_ps(out);
  }
  if (num == 1) [[likely]] {
    return _mm_castsi128_ps(_mm_loadu_si32(reinterpret_cast<const __m128i*>(ptr)));
  }
  return _mm_castsi128_ps(_mm_setzero_si128());
}
[[gnu::noinline]] auto op3(const Val* ptr, std::size_t num) {
  std::array<Val, Vec::size> buf{};
  std::memcpy(buf.data(), ptr, num);
  return Vec{}.load(buf.data());
}
[[gnu::noinline]] auto op4(const Val* ptr, std::size_t num) {
  return Vec{
    num > 0 ? ptr[0] : 0,
    num > 1 ? ptr[1] : 0,
    num > 2 ? ptr[2] : 0,
    num > 3 ? ptr[3] : 0,
  };
}
[[gnu::noinline]] Vec op5(const Val* ptr, std::size_t num) {
  if (num >= Vec::size) [[unlikely]] {
    return _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
  }
  Vec out{0};
  if ((num & 2U) != 0) {
    out = _mm_castsi128_ps(_mm_loadu_si64(reinterpret_cast<const __m128i*>(ptr)));
  }
  if ((num & 1U) != 0) {
    __m128i m32 = _mm_loadu_si32(reinterpret_cast<const __m128i*>(ptr + (num - 1)));
    if (num == 1) {
      return _mm_castsi128_ps(m32);
    }
    out = _mm_movelh_ps(out, _mm_castsi128_ps(m32));
  }
  return out;
}
[[gnu::noinline]] auto op6(const Val* ptr, std::size_t num) {
  if (num >= 2) {
    if (num >= 4) [[unlikely]] {
      return _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
    }
    __m128i out = _mm_loadu_si64(reinterpret_cast<const __m128i*>(ptr));
    if (num == 2) {
      return _mm_castsi128_ps(out);
    }
    return _mm_movelh_ps(_mm_castsi128_ps(out),
                         _mm_loadu_ps(reinterpret_cast<const Val*>(ptr + 2)));
  }
  if (num == 1) [[likely]] {
    return _mm_castsi128_ps(_mm_loadu_si32(reinterpret_cast<const __m128i*>(ptr)));
  }
  return _mm_castsi128_ps(_mm_setzero_si128());
}
[[gnu::noinline]] auto op7(const Val* ptr, std::size_t num) {
  return _mm_maskload_ps(ptr, stado::part_mask<stado::Mask<32, 4>>(num));
}

#define DIST_full std::uniform_int_distribution<stado::u64> uniform_dist(0, Vec::size);
#define DIST_redu std::uniform_int_distribution<stado::u64> uniform_dist(1, Vec::size - 1);
#define DIST_1 std::uniform_int_distribution<stado::u64> uniform_dist(1, 1);
#define DIST_2 std::uniform_int_distribution<stado::u64> uniform_dist(2, 2);
#define DIST_3 std::uniform_int_distribution<stado::u64> uniform_dist(3, 3);

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
  BM_OP(5, DIST) \
  BM_OP(6, DIST) \
  BM_OP(7, DIST)

BM_OPS(full)
BM_OPS(redu)
BM_OPS(1)
BM_OPS(2)
BM_OPS(3)

BENCHMARK_MAIN();
