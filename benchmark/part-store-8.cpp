#include <array>
#include <cstddef>
#include <cstring>
#include <random>

#include "benchmark/benchmark.h"
#include "pcg_extras.hpp"
#include "pcg_random.hpp"

#include "stado/stado.hpp"

using Val = stado::i8;
using Vec = stado::Vector<Val, 16>;

auto op32(Vec v, stado::u32* ptr, std::size_t num) {
#if STADO_INSTRUCTION_SET >= STADO_AVX2
  _mm_maskstore_epi32(reinterpret_cast<int*>(ptr), stado::part_mask<stado::Mask<32, 4>>(num), v);
#else
  const auto vf = _mm_castsi128_ps(v);
  switch (num) {
  [[unlikely]] case 0:
    break;
  case 1: _mm_storeu_si32(ptr, v); break;
  case 2: _mm_storeu_si64(ptr, v); break;
  case 3:
    _mm_storeu_si64(ptr, v);
    _mm_storeu_si32(ptr + 2, _mm_castps_si128(_mm_movehl_ps(vf, vf)));
    break;
  [[unlikely]] default:
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v);
    break;
  }
  return *ptr;
#endif
}
auto op16(Vec v, stado::u16* ptr, std::size_t num) {
  const std::size_t num2 = num / 2;
  op32(v, reinterpret_cast<stado::u32*>(ptr), num2);
  if ((num & 1) != 0) {
    const auto idx = num - 1;
    const auto shuf = _mm_set1_epi16(stado::i16(((2 * idx + 1) << 8) + 2 * idx));
    _mm_storeu_si16(ptr + idx, _mm_shuffle_epi8(v, shuf));
  }
  return *ptr;
}

[[gnu::noinline]] auto op1(Vec v, Val* ptr, std::size_t num) {
  v.store_partial(num, ptr);
  return *ptr;
}
[[gnu::noinline]] auto op2(Vec v, Val* ptr, std::size_t num) {
  _mm_maskmoveu_si128(v, stado::part_mask<stado::Mask<8, 16>>(num), reinterpret_cast<char*>(ptr));
  return *ptr;
}
[[gnu::noinline]] auto op3(Vec v, Val* ptr, std::size_t num) {
  const std::size_t num2 = num / 2;
  op16(v, reinterpret_cast<stado::u16*>(ptr), num2);
  if ((num & 1) != 0) {
    const auto idx = num - 1;
    const auto shuf = _mm_set1_epi8(stado::i8(idx));
    ptr[idx] = _mm_extract_epi8(_mm_shuffle_epi8(v, shuf), 0); // SSSE3/SSE4.1
  }
  return *ptr;
}
[[gnu::noinline]] auto op4(Vec v, Val* ptr, std::size_t num) {
  const std::size_t num4 = num / 4;
  op32(v, reinterpret_cast<stado::u32*>(ptr), num4);
  if ((num & 3) != 0) {
    const std::size_t idx = 4 * num4;
    const __m128i shufo = _mm_set1_epi32(stado::i32(idx * 0x01010101 + 0x03020100));
    const __m128i shuf = _mm_shuffle_epi8(v, shufo); // SSSE3
    if ((num & 2) != 0) {
      _mm_storeu_si16(ptr + idx, shuf);
    }
    if ((num & 1) != 0) {
      ptr[idx + 2] = _mm_extract_epi8(shuf, 2); // SSE4.1
    }
  }
  return *ptr;
}
auto op5i(Vec src, Val* dst, std::size_t size) {
  if (size >= 16) {
    src.store(dst);
    return;
  }
  if (size <= 0) {
    return;
  }
  // we are not using _mm_maskmoveu_si128 because it is too slow on many processors
  std::array<Val, Vec::size> buf{};
  src.store(buf.data());
  std::size_t j = 0;
  if ((size & 8) != 0) {
    reinterpret_cast<stado::u64*>(dst)[0] = reinterpret_cast<stado::u64*>(buf.data())[0];
    j += 8;
  }
  if ((size & 4) != 0) {
    reinterpret_cast<stado::u32*>(dst)[j / 4] = reinterpret_cast<stado::u32*>(buf.data())[j / 4];
    j += 4;
  }
  if ((size & 2) != 0) {
    reinterpret_cast<stado::u16*>(dst)[j / 2] = reinterpret_cast<stado::u16*>(buf.data())[j / 2];
    j += 2;
  }
  if ((size & 1) != 0) {
    dst[j] = buf[j];
  }
}
[[gnu::noinline]] auto op5(Vec v, Val* ptr, std::size_t num) {
  op5i(v, ptr, num);
  return *ptr;
}
[[gnu::noinline]] auto op6(Vec v, Val* ptr, std::size_t num) {
  const std::size_t num2 = num / 2;
  op16(v, reinterpret_cast<stado::u16*>(ptr), num2);
  if ((num & 1) != 0) {
    const auto idx = num - 1;
    ptr[idx] = v.extract(idx);
  }
  return *ptr;
}
[[gnu::noinline]] auto op7(Vec v, Val* ptr, std::size_t num) {
  const std::size_t num4 = num / 4;
  op32(v, reinterpret_cast<stado::u32*>(ptr), num4);
  std::size_t j = 4 * num4;
  if ((num & 2) != 0) {
    reinterpret_cast<stado::u16*>(ptr)[j / 2] = stado::Vector<stado::i16, 8>(v).extract(j / 2);
    j += 2;
  }
  if ((num & 1) != 0) {
    ptr[j] = v.extract(j);
  }
  return *ptr;
}

#define DIST_full std::uniform_int_distribution<stado::u64> uniform_dist(0, Vec::size);
#define DIST_redu std::uniform_int_distribution<stado::u64> uniform_dist(1, Vec::size - 1);
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
    std::array<Val, Vec::size> arr{}; \
    for (auto _ : state) { \
      Vec v{fdist(rng), fdist(rng), fdist(rng), fdist(rng), fdist(rng), fdist(rng), \
            fdist(rng), fdist(rng), fdist(rng), fdist(rng), fdist(rng), fdist(rng), \
            fdist(rng), fdist(rng), fdist(rng), fdist(rng)}; \
      benchmark::DoNotOptimize(op##NUM(v, arr.data(), uniform_dist(rng))); \
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

// NOLINTBEGIN
BM_OPS(full)
BM_OPS(redu)
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
