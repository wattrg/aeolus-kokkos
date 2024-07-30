add_library(
    linear_algebra
    STATIC
    linear_algebra/gmres.cpp 
    linear_algebra/dense_linear_algebra.cpp
)
target_link_libraries(
    linear_algebra 
    PUBLIC 
    Kokkos::kokkos 
    doctest 
    simulation
    util 
    finite_volume 
    doctest
)
target_include_directories(linear_algebra PUBLIC .)


if (Ibis_BUILD_TESTS)
    add_executable(
        linear_algebra_unittest 
        test/unittest.cpp
        linear_algebra/dense_linear_algebra.cpp
        linear_algebra/gmres.cpp
    )

    target_link_libraries(
	      linear_algebra_unittest 	
	      PRIVATE
	      Kokkos::kokkos
	      doctest
	      util
	      simulation
    )

    target_include_directories(linear_algebra_unittest PRIVATE .)

    add_test(NAME linear_algebra_unittest COMMAND linear_algebra_unittest)
endif(Ibis_BUILD_TESTS)
