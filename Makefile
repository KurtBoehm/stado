setup-opt:
	rm -rf build ; meson setup -Dtest=true --wrap-mode=forcefallback --warnlevel 1 --optimization=3 -Db_ndebug=true build
setup-opt-debug:
	rm -rf build ; meson setup -Dtest=true --wrap-mode=forcefallback --warnlevel 1 --optimization=3 -Ddebug=true build
setup-debug:
	rm -rf build ; meson setup -Dtest=true --wrap-mode=forcefallback --warnlevel 3 --buildtype=debug build
