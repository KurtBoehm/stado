#include <iostream>

#include "stado/stado.hpp"
// #include "stado/print.hpp"

void print(stado::f64x2 vec) {
  std::cout << vec.extract(0) << ", " << vec.extract(1) << '\n';
}
void print(stado::f32x4 vec) {
  std::cout << vec.extract(0) << ", " << vec.extract(1) << ", " << vec.extract(2) << ", "
            << vec.extract(3) << '\n';
}
void print(stado::i8x16 vec) {
  std::cout << int(vec.extract(0)) << ", " << int(vec.extract(1)) << ", " << int(vec.extract(2))
            << ", " << int(vec.extract(3)) << ", " << int(vec.extract(4)) << ", "
            << int(vec.extract(5)) << ", " << int(vec.extract(6)) << ", " << int(vec.extract(7))
            << ", " << int(vec.extract(8)) << ", " << int(vec.extract(9)) << ", "
            << int(vec.extract(10)) << ", " << int(vec.extract(11)) << ", " << int(vec.extract(12))
            << ", " << int(vec.extract(13)) << ", " << int(vec.extract(14)) << ", "
            << int(vec.extract(15)) << '\n';
}

void test1() {
  using Vec = stado::f32x4;
  Vec vec(1, 2, 3, 4);
  print(vec);
}

#if STADO_INSTRUCTION_SET >= STADO_AVX
void test2() {
  using Vec = stado::f32x8;

  Vec vec(1, 2, 3, 4, 5, 6, 7, 8);
  std::cout << vec.extract(0) << ", " << vec.extract(1) << ", " << vec.extract(2) << ", "
            << vec.extract(3) << ", " << vec.extract(4) << ", " << vec.extract(5) << ", "
            << vec.extract(6) << ", " << vec.extract(7) << '\n';
}
#else
void test2() {}
#endif

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
void test3() {
  using Vec = stado::f32x16;

  Vec vec(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  std::cout << vec.extract(0) << ", " << vec.extract(1) << ", " << vec.extract(2) << ", "
            << vec.extract(3) << ", " << vec.extract(4) << ", " << vec.extract(5) << ", "
            << vec.extract(6) << ", " << vec.extract(7) << ", " << vec.extract(8) << ", "
            << vec.extract(9) << ", " << vec.extract(10) << ", " << vec.extract(11) << ", "
            << vec.extract(12) << ", " << vec.extract(13) << ", " << vec.extract(14) << ", "
            << vec.extract(15) << '\n';
}
#else
void test3() {}
#endif

void test4() {
  using Vec = stado::f64x2;
  Vec vec(1, 3);
  print(vec);
}

#if STADO_INSTRUCTION_SET >= STADO_AVX
void test5() {
  using vector_type = stado::f64x4;

  vector_type vec(1, 2, 3, 4);
  std::cout << vec.extract(0) << ", " << vec.extract(1) << ", " << vec.extract(2) << ", "
            << vec.extract(3) << '\n';
}
#else
void test5() {}
#endif

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
void test6() {
  using vector_type = stado::f64x8;

  vector_type vec(1, 2, 3, 4, 5, 6, 7, 8);
  std::cout << vec.extract(0) << ", " << vec.extract(1) << ", " << vec.extract(2) << ", "
            << vec.extract(3) << ", " << vec.extract(4) << ", " << vec.extract(5) << ", "
            << vec.extract(6) << ", " << vec.extract(7) << '\n';
}
#else
void test6() {}
#endif

void test7() {
  using Vec = stado::i8x16;
  Vec vec(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 35, 16);
  print(vec);
}

int main() {
  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();
}
