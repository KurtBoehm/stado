project('stado', 'cpp', default_options: ['cpp_std=c++20', 'warning_level=3'])

stado_dep = declare_dependency(include_directories: include_directories('include'))

install_subdir('include', install_dir: 'include')

pc = import('pkgconfig')
pc.generate(
  name: 'stado',
  version: meson.project_version(),
  description: 'Generic SIMD classes and functions',
)

if get_option('benchmark')
  subdir('benchmark')
endif
if get_option('test')
  subdir('test')
endif
