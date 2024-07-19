add_library(
	solver
	STATIC
	solvers/solver.cpp
	solvers/solver.h
	solvers/cfl.cpp
	solvers/steady_state.cpp
)

target_link_libraries(
	solver 
	PUBLIC 
	Kokkos::kokkos 
	nlohmann_json::nlohmann_json 
	linear_algebra
	grid 
	gas
	spdlog::spdlog
	IO
)

add_library(runge_kutta STATIC solvers/runge_kutta.cpp)

target_include_directories(solver PUBLIC .)

target_link_libraries(
	runge_kutta 
	PUBLIC 
	Kokkos::kokkos 
	doctest 
	util 
	grid 
	nlohmann_json::nlohmann_json
	finite_volume
	solver
	spdlog::spdlog
	IO
)

if (Ibis_BUILD_TESTS)
	add_executable(
		solver_unittest
		test/unittest.cpp
		solvers/cfl.cpp
	)
	target_link_libraries(
		solver_unittest 
		PRIVATE
		Kokkos::kokkos 
		nlohmann_json::nlohmann_json 
		grid 
		gas
		spdlog::spdlog
		IO
	)
	add_test(NAME solver_unittest COMMAND solver_unittest)
endif(Ibis_BUILD_TESTS)
