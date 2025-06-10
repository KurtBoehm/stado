options_sub = subproject('options')
all_args = options_sub.get_variable('all_args')

benchmark_dep = dependency('benchmark')
pcg_cpp_dep = dependency('pcg-cpp')

foreach name, args : {
  'cvt128-u32-f64': all_args,
  'cvt256-u16-u64': all_args,
  'cvt256-u64-i64': all_args,
  'extend': all_args,
  'extend-lookup': all_args,
  'extend-lookup-mean': all_args,
  'index': all_args,
  'lookup': all_args,
  'part-load-f32': ['-march=x86-64-v3'],
  'part-load-f64': ['-march=x86-64-v3'],
  'part-mask': all_args,
}
  executable(
    'bm-' + name,
    [f'@name@.cpp'],
    cpp_args: args,
    dependencies: [stado_dep, benchmark_dep, pcg_cpp_dep],
  )
endforeach
