setup-opt:
	rm -rf build ; meson setup -Dtest=true --wrap-mode=forcefallback --warnlevel 1 --buildtype=release -Db_ndebug=true build
setup-opt-debug:
	rm -rf build ; meson setup -Dtest=true --wrap-mode=forcefallback --warnlevel 3 -Ddebug=true --optimization=3 build
setup-debug:
	rm -rf build ; meson setup -Dtest=true --wrap-mode=forcefallback --warnlevel 3 --buildtype=debug build
