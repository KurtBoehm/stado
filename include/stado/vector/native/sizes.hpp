#ifndef INCLUDE_STADO_VECTOR_NATIVE_SIZES_HPP
#define INCLUDE_STADO_VECTOR_NATIVE_SIZES_HPP

#include <array>
#include <cstddef>

#include "stado/defs.hpp"
#include "stado/instruction-set.hpp"

namespace stado {
template<typename T>
struct NativeSizesTrait;

template<>
struct NativeSizesTrait<f32> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512F
  static constexpr std::array<std::size_t, 3> value{4, 8, 16};
#elif STADO_INSTRUCTION_SET >= STADO_AVX
  static constexpr std::array<std::size_t, 2> value{4, 8};
#else
  static constexpr std::array<std::size_t, 1> value{4};
#endif
};

template<>
struct NativeSizesTrait<f64> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512F
  static constexpr std::array<std::size_t, 3> value{2, 4, 8};
#elif STADO_INSTRUCTION_SET >= STADO_AVX
  static constexpr std::array<std::size_t, 2> value{2, 4};
#else
  static constexpr std::array<std::size_t, 1> value{2};
#endif
};

template<>
struct NativeSizesTrait<u8> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  static constexpr std::array<std::size_t, 3> value{16, 32, 64};
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  static constexpr std::array<std::size_t, 2> value{16, 32};
#else
  static constexpr std::array<std::size_t, 1> value{16};
#endif
};

template<>
struct NativeSizesTrait<i8> : public NativeSizesTrait<u8> {};

template<>
struct NativeSizesTrait<u16> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512SKL // AVX512BW
  static constexpr std::array<std::size_t, 3> value{8, 16, 32};
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  static constexpr std::array<std::size_t, 2> value{8, 16};
#else
  static constexpr std::array<std::size_t, 1> value{8};
#endif
};

template<>
struct NativeSizesTrait<i16> : public NativeSizesTrait<u16> {};

template<>
struct NativeSizesTrait<u32> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512F
  static constexpr std::array<std::size_t, 3> value{4, 8, 16};
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  static constexpr std::array<std::size_t, 2> value{4, 8};
#else
  static constexpr std::array<std::size_t, 1> value{4};
#endif
};

template<>
struct NativeSizesTrait<i32> : public NativeSizesTrait<u32> {};

template<>
struct NativeSizesTrait<u64> {
#if STADO_INSTRUCTION_SET >= STADO_AVX512F
  static constexpr std::array<std::size_t, 3> value{2, 4, 8};
#elif STADO_INSTRUCTION_SET >= STADO_AVX2
  static constexpr std::array<std::size_t, 2> value{2, 4};
#else
  static constexpr std::array<std::size_t, 1> value{2};
#endif
};

template<>
struct NativeSizesTrait<i64> : public NativeSizesTrait<u64> {};

template<typename T>
static constexpr auto native_sizes = NativeSizesTrait<T>::value;
} // namespace stado

#endif // INCLUDE_STADO_VECTOR_NATIVE_SIZES_HPP
