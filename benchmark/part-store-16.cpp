#include <array>
#include <cstddef>
#include <cstring>
#include <random>

#include "benchmark/benchmark.h"
#include "pcg_extras.hpp"
#include "pcg_random.hpp"

#include "stado/stado.hpp"

using Val = stado::i16;
using Vec = stado::Vector<Val, 8>;

auto op32(Vec v, stado::i32* ptr, std::size_t num) {
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

[[gnu::noinline]] auto op1(Vec v, Val* ptr, std::size_t num) {
  v.store_partial(num, ptr);
  return *ptr;
}
[[gnu::noinline]] auto op2(Vec v, Val* ptr, std::size_t num) {
  _mm_maskmoveu_si128(v, stado::part_mask<stado::Mask<16, 8>>(num), reinterpret_cast<char*>(ptr));
  return *ptr;
}
[[gnu::noinline]] auto op3(Vec v, Val* ptr, std::size_t num) {
  const std::size_t num2 = num / 2;
  op32(v, reinterpret_cast<stado::i32*>(ptr), num2);
  if ((num & 1) != 0) {
    switch (num2) {
    case 0: _mm_storeu_si16(ptr, v); break;
    case 1: ptr[2] = _mm_extract_epi16(v, 2); break;
    case 2: ptr[4] = _mm_extract_epi16(v, 4); break;
    case 3: ptr[6] = _mm_extract_epi16(v, 6); break;
    default: break;
    }
  }
  return *ptr;
}
[[gnu::noinline]] auto op4(Vec v, Val* ptr, std::size_t num) {
  const std::size_t num2 = num / 2;
  op32(v, reinterpret_cast<stado::i32*>(ptr), num2);
  if ((num & 1) != 0) {
    if (num2 >= 2) {
      if (num2 >= 3) {
        ptr[6] = _mm_extract_epi16(v, 6);
      } else {
        ptr[4] = _mm_extract_epi16(v, 4);
      }
    } else {
      if (num2 >= 1) {
        ptr[2] = _mm_extract_epi16(v, 2);
      } else {
        _mm_storeu_si16(ptr, v);
      }
    }
  }
  return *ptr;
}
[[gnu::noinline]] auto op5(Vec v, Val* ptr, std::size_t num) {
  std::array<Val, Vec::size> buf{};
  v.store(buf.data());
  std::memcpy(buf.data(), ptr, 2 * num);
  return *ptr;
}
auto op6i(Vec v, Val* ptr, std::size_t num) {
  if (num >= Vec::size) [[unlikely]] {
    v.store(ptr);
    return;
  }
  if (num == 0) [[unlikely]] {
    return;
  }
  std::array<Val, Vec::size> buf{};
  v.store(buf.data());
  std::size_t j = 0;
  if ((num & 4) != 0) {
    *reinterpret_cast<stado::u64*>(ptr) = *reinterpret_cast<stado::u64*>(buf.data());
    j += 8;
  }
  if ((num & 2) != 0) {
    reinterpret_cast<stado::u32*>(ptr)[j / 4] = reinterpret_cast<stado::u32*>(buf.data())[j / 4];
    j += 4;
  }
  if ((num & 1) != 0) {
    ptr[j / 2] = buf[j / 2];
  }
}
[[gnu::noinline]] auto op6(Vec v, Val* ptr, std::size_t num) {
  op6i(v, ptr, num);
  return *ptr;
}
[[gnu::noinline]] auto op7(Vec v, Val* ptr, std::size_t num) {
  const std::size_t num2 = num / 2;
  op32(v, reinterpret_cast<stado::i32*>(ptr), num2);
  if ((num & 1) != 0) {
    const auto idx = num - 1;
    const auto shuf = _mm_set1_epi16(stado::i16(((2 * idx + 1) << 8) + 2 * idx));
    _mm_storeu_si16(ptr + idx, _mm_shuffle_epi8(v, shuf));
  }
  return *ptr;
}
[[gnu::noinline]] auto op8(Vec v, Val* ptr, std::size_t num) {
  const std::size_t num2 = num / 2;
  op32(v, reinterpret_cast<stado::i32*>(ptr), num2);
  if ((num & 1) != 0) {
    std::array<Val, Vec::size> buf{};
    v.store(buf.data());
    const auto idx = num - 1;
    ptr[idx] = buf[idx];
  }
  return *ptr;
}

#define DIST_full std::uniform_int_distribution<stado::u64> uniform_dist(0, Vec::size);
#define DIST_redu std::uniform_int_distribution<stado::u64> uniform_dist(1, Vec::size - 1);
#define DIST_1 std::uniform_int_distribution<stado::u64> uniform_dist(1, 1);
#define DIST_2 std::uniform_int_distribution<stado::u64> uniform_dist(2, 2);
#define DIST_3 std::uniform_int_distribution<stado::u64> uniform_dist(3, 3);
#define DIST_4 std::uniform_int_distribution<stado::u64> uniform_dist(4, 4);
#define DIST_5 std::uniform_int_distribution<stado::u64> uniform_dist(5, 5);
#define DIST_6 std::uniform_int_distribution<stado::u64> uniform_dist(6, 6);
#define DIST_7 std::uniform_int_distribution<stado::u64> uniform_dist(7, 7);

#define BM_OP(NUM, DIST) \
  static void bm_op##NUM##_##DIST(benchmark::State& state) { \
    pcg_extras::seed_seq_from<std::random_device> seed_source; \
    pcg32 rng(seed_source); \
    DIST_##DIST; \
    std::uniform_int_distribution<Val> fdist{}; \
    std::array<Val, Vec::size> arr{}; \
    for (auto _ : state) { \
      Vec v{fdist(rng), fdist(rng), fdist(rng), fdist(rng), \
            fdist(rng), fdist(rng), fdist(rng), fdist(rng)}; \
      benchmark::DoNotOptimize(op##NUM(v, arr.data(), uniform_dist(rng))); \
    } \
  } \
  BENCHMARK(bm_op##NUM##_##DIST);

#define BM_OPS(DIST) \
  BM_OP(1, DIST) \
  BM_OP(2, DIST) \
  BM_OP(3, DIST) \
  /*BM_OP(4, DIST)*/ \
  /*BM_OP(5, DIST)*/ \
  /*BM_OP(6, DIST)*/ \
  BM_OP(7, DIST) \
  BM_OP(8, DIST)

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
BENCHMARK_MAIN();
// NOLINTEND
