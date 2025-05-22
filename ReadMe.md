# Stado 🐄: A Generic Header-Only SIMD Library Without Dependencies

Stado is a generic header-only C++20 library which provides classes and functions to facilitate using explicit SIMD instructions on x86-64.
It is based heavily on the [Vector Class Library](https://github.com/vectorclass/version2) and inherits its licence, but restructures it into a collection of template classes and adds new operations.

This project is a relatively old attempt at creating a well-structured and high-performance SIMD library, and while its performance is quite good, it retains many structural issues which have proven quite unpleasant to resolve.
As a consequence, I am working on a new SIMD library that is not bogged down by its heritage and will contain support for ARM NEON (and, potentially, additional instruction sets).
Until that is ready enough, this project remains the SIMD back-end of Lineal.
