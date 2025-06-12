options_sub = subproject('options')
all_args = options_sub.get_variable('all_args')

benchmark_dep = dependency('benchmark')
omp_dep = dependency('openmp')
pcg_cpp_dep = dependency('pcg-cpp')

foreach name, args : {
  'cvt128-u32-f64': all_args,
  'cvt256-u16-u64': all_args,
  'cvt256-u64-i64': all_args,
  'extend': all_args,
  'extend-lookup': all_args,
  'extend-lookup-mean': all_args,
  'extract-8': ['-march=x86-64-v3'],
  'extract-16': ['-march=x86-64-v3'],
  'gather-u32x4': ['-march=x86-64-v3'],
  'gather-u32x8': ['-march=x86-64-v3'],
  'index': all_args,
  'lookup': all_args,
  'part-load-f32': ['-march=x86-64-v3'],
  'part-load-f64': ['-march=x86-64-v3'],
  'part-store-8': ['-march=x86-64-v3'],
  'part-store-16': ['-march=x86-64-v3'],
  'part-store-32': ['-march=x86-64-v3'],
  'part-store-64': ['-march=x86-64-v3'],
  'part-mask': all_args,
}
  executable(
    'bm-' + name,
    [f'@name@.cpp'],
    cpp_args: args,
    dependencies: [stado_dep, benchmark_dep, omp_dep, pcg_cpp_dep],
  )
endforeach