add_library(
    util 
    STATIC 
    util/vector3.cpp
    util/ragged_array.cpp
    util/cubic_spline.cpp
)
target_link_libraries(util PUBLIC Kokkos::kokkos doctest)
target_include_directories(util PUBLIC .)


if (Ibis_BUILD_TESTS)
    add_executable(
        util_unittest 
    	  test/unittest.cpp 
    	  util/vector3.cpp 
    	  util/ragged_array.cpp
    	  util/cubic_spline.cpp
    	  util/dual.cpp
    )

    target_link_libraries(
	      util_unittest 	
	      PRIVATE
	      Kokkos::kokkos
	      doctest
    )

    target_include_directories(util_unittest PRIVATE .)

    add_test(NAME util_unittest COMMAND util_unittest)
endif(Ibis_BUILD_TESTS)
