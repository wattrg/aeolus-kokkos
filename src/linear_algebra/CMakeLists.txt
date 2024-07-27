add_library(
    linear_algebra
    STATIC
    linear_algebra/gmres.cpp 
    linear_algebra/vector.cpp
)
target_link_libraries(
    linear_algebra 
    PUBLIC 
    Kokkos::kokkos 
    doctest 
    simulation
    util 
    finite_volume 
)
target_include_directories(linear_algebra PUBLIC .)


if (Ibis_BUILD_TESTS)
    add_executable(
        linear_algebra_unittest 
        test/unittest.cpp
    )

    target_link_libraries(
	      linear_algebra_unittest 	
	      PRIVATE
	      Kokkos::kokkos
	      doctest
    )

    target_include_directories(linear_algebra_unittest PRIVATE .)

    add_test(NAME linear_algebra_unittest COMMAND linear_algebra_unittest)
endif(Ibis_BUILD_TESTS)
