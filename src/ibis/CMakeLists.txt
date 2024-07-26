add_subdirectory(ibis/commands)

add_executable(ibis ibis/main.cpp)

target_link_libraries(
	ibis 
	PUBLIC 
	prep 
	run 
	post
	ibis_clean 
	gas 
	util 
	grid 
	doctest 
	ibis_git_version 
	ibis_version
	runtime_dirs
	spdlog::spdlog
	solver
	IO
	simulation
	CLI11::CLI11
)

set_target_properties(ibis PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if (Ibis_LINK_FS)
	target_link_libraries(ibis PUBLIC stdc++fs)
endif()
install(TARGETS ibis DESTINATION ${CMAKE_INSTALL_BIN_DIR})
