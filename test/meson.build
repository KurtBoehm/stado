options_sub = subproject('options')
args = options_sub.get_variable('all_args')

foreach name, deps : {
  'base': [],
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
