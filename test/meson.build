options_sub = subproject('options')
args = options_sub.get_variable('all_args')

benchmark_dep = dependency('benchmark')
pcg_cpp_dep = dependency('pcg-cpp')

foreach name, deps : {
  'base': [benchmark_dep, pcg_cpp_dep],
  'bench-extend': [benchmark_dep, pcg_cpp_dep],
  'bench-extend-lookup': [benchmark_dep, pcg_cpp_dep],
  'bench-extend-lookup-mean': [benchmark_dep, pcg_cpp_dep],
  'bench-index': [benchmark_dep, pcg_cpp_dep],
  'bench-lookup': [benchmark_dep, pcg_cpp_dep],
  'bench-part-mask': [benchmark_dep, pcg_cpp_dep],
  'extend-lookup': [],
  'extract-print': [],
}
  test(
    name,
    executable(
      'test-' + name,
      [f'@name@.cpp'],
      cpp_args: args,
      dependencies: [stado_dep] + deps,
    ),
  )
endforeach
