options_sub = subproject('options')
opts = options_sub.get_variable('all_opts')
args = options_sub.get_variable('all_args')

benchmark_dep = dependency('benchmark')
pcg_cpp_dep = dependency('pcg-cpp')

foreach name, deps : {
  'base': [benchmark_dep, pcg_cpp_dep],
}
  test(
    name,
    executable('test-' + name,
      [f'@name@.cpp'],
      override_options: opts,
      cpp_args: args,
      dependencies: [stado_dep] + deps,
    )
  )
endforeach
