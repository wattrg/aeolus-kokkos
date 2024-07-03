add_library(
	  finite_volume 
	  STATIC 
	  finite_volume/finite_volume.cpp 
	  finite_volume/flux_calc.cpp 
	  finite_volume/convective_flux.cpp
	  finite_volume/viscous_flux.cpp
	  finite_volume/flux_calculators/hanel.cpp
	  finite_volume/flux_calculators/ausmdv.cpp
	  finite_volume/flux_calculators/ldfss2.cpp
	  finite_volume/primative_conserved_conversion.cpp
	  finite_volume/conserved_quantities.cpp
	  finite_volume/boundaries/boundary.cpp
	  finite_volume/gradient.cpp
	  finite_volume/limiter.cpp
)

target_link_libraries(
	  finite_volume 
	  PUBLIC
	  Kokkos::kokkos 
	  doctest 
	  util
	  nlohmann_json::nlohmann_json
	  spdlog::spdlog
	  gas
	  grid
)

target_include_directories(finite_volume PUBLIC .)

if (Ibis_BUILD_TESTS)
    add_executable(
    	  finite_volume_unittest 
    	  test/unittest.cpp 
    	  finite_volume/flux_calc.cpp 
			  finite_volume/convective_flux.cpp
			  finite_volume/viscous_flux.cpp
    	  finite_volume/flux_calculators/hanel.cpp
    	  finite_volume/flux_calculators/ausmdv.cpp
			  finite_volume/flux_calculators/ldfss2.cpp
			  finite_volume/gradient.cpp
			  finite_volume/limiter.cpp
    )
    # target_compile_options(finite_volume_unittest -g)
    target_link_libraries(
    	  finite_volume_unittest 
    	  PRIVATE 
    	  util
    	  doctest
    	  Kokkos::kokkos
    	  nlohmann_json::nlohmann_json
    	  spdlog::spdlog
	      gas
	      grid
    )
    target_include_directories(finite_volume_unittest PRIVATE util)
    
		add_test(NAME finite_volume_unittest COMMAND finite_volume_unittest)
endif(Ibis_BUILD_TESTS)
