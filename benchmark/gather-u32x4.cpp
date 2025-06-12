#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <random>
#include <span>

#include "benchmark/benchmark.h"
#include "pcg_extras.hpp"
#include "pcg_random.hpp"

#include "stado/stado.hpp"

using Val = stado::f32;
using Idx = stado::u32;
inline constexpr std::size_t size = 4;
inline constexpr std::size_t nidx = 8192;
using stado::Vector;

[[gnu::noinline]] Vector<Val, size> op0(std::span<const Val> data, Vector<Idx, size> idxs) {
  return {data[idxs[0]], data[idxs[1]], data[idxs[2]], data[idxs[3]]};
}
[[gnu::noinline]] Vector<Val, size> op1(std::span<const Val> data, Vector<Idx, size> idxs) {
  constexpr Idx limit = Idx{1} << 31;
  if (data.size() >= limit) {
    return _mm_i32gather_ps(data.data() + limit, idxs ^ limit, 4);
  }
  return _mm_i32gather_ps(data.data(), idxs, 4);
}
[[gnu::noinline]] Vector<Val, size> op2(std::span<const Val> data, Vector<Idx, size> idxs) {
  return _mm256_i64gather_ps(data.data(), _mm256_cvtepu32_epi64(idxs), 4);
}

static void bm_base(benchmark::State& state, auto op) {
  constexpr std::size_t data_size = 0xC0000000;

  pcg64 rng{pcg_extras::seed_seq_from<std::random_device>{}};
  const auto data = std::make_unique<Val[]>(data_size);
  std::uniform_real_distribution<Val> vdist{};
#pragma omp parallel for default(shared) private(vdist, rng) schedule(guided)
  for (std::size_t i = 0; i < data_size; ++i) {
    data[i] = vdist(rng);
  }

  std::uniform_int_distribution<Idx> idist{0, data_size - 1};
  std::array<Idx, nidx> idx_arr{};
  std::generate(idx_arr.begin(), idx_arr.end(), [&] { return idist(rng); });

  std::size_t i = 0;
  for (auto _ : state) {
    const auto idxs = Vector<Idx, size>{}.load(idx_arr.data() + i);
    benchmark::DoNotOptimize(op(std::span{data.get(), stado::horizontal_max(idxs) + 1}, idxs));
    i = (i + 4) % nidx;
  }
}
static void bm0(benchmark::State& state) {
  bm_base(state, op0);
}
static void bm1(benchmark::State& state) {
  bm_base(state, op1);
}
static void bm2(benchmark::State& state) {
  bm_base(state, op2);
}

BENCHMARK(bm0);
BENCHMARK(bm1);
BENCHMARK(bm2);
BENCHMARK_MAIN();
