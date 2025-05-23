# Stado 🐑: A Generic Header-Only SIMD Library Without Dependencies

Stado (Polish “herd, flock”) is a generic header-only C++20 library which provides classes and functions to facilitate using explicit SIMD instructions on x86-64.
It provides template classes for natively supported SIMD vectors and masks, as well as classes representing part of a native SIMD vector or the concatenation of multiple SIMD vectors.
This generic approach simplifies the use within template classes and function to support different values types and vector sizes.
These data structures are complemented by a large collection of operations.

Stado started out as a fork of the [Vector Class Library](https://github.com/vectorclass/version2) (VCL for short), and the implementation of most operations can be characterized as a cleaned-up version of the VCL implementation.
In most other respects, however, Stado deviates very significantly from the VCL, especially with regard to its templated structure and support for extended vector sizes, but also by providing some new and improved operations which are implemented generically:

- Stado uses a more efficient approach for cutting off SIMD vectors at a given size and contains utilities for determining appropriate masks.
- Stado provides generic gather functions, including masked lookups and partial lookups.
- Stado provides generic conversions between value types.

Despite these favourable properties and Stado’s high performance, it retains structural issues which are quite tedious to resolve, make working on this library somewhat unpleasant, and hinder adding support for additional instruction sets significantly.
Therefore, I do not plan on improving Stado significantly beyond its current state, including many generic operations only being implemented for some parameter type combinations and other hurdles to complete generic use.
Instead, I am working on a new SIMD library that is written from the ground up, eschewing the issues Stado has due to its heritage while allowing for a much more consistent code structure and easier extensibility, especially for additional instruction sets such as ARM NEON.

## Licence

Stado, like the VCL, is licenced under the Apache Licence Version 2.0, as provided in [`License`](License).
However, this introduces an unfortunate challenge:
The Apache Licence requires “any modified files to carry prominent notices stating that You changed the files”, which is pointless due to the completely different code structure.
As such, I provide the aforementioned notes as a statement of the modifications made compared to VCL and refrain from adding a modification remark to every file.
