cmake_minimum_required(VERSION 3.20.1)

project(ibis VERSION 0.1.0)

# No in-source builds
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# helpful for linters
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Bunya's AMD nodes need manually linking to the filesystem
# library for some reason. To compile on bunya, add
# -DIbis_LINK_FS=ON to the cmake configuration command
if (NOT Ibis_LINK_FS)
    set(Ibis_LINK_FS OFF)
endif()

# Optionally use mold (a fast, parallel linker)
if (Ibis_USE_MOLD)
    set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=mold")
endif()

# default install location is in inst folder in the
# root of the repository
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/inst")

# Use c++ 17
set(CMAKE_CXX_STANDARD 17)

# to build the python bindings, we need to make position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# get information about the git commit being built
INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ibis_git_info.cmake)
check_git_setup()

# get information about the install directory
include(GNUInstallDirs)
INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/runtime_dirs.cmake)
setup_runtime_dirs()

# embed the version in the executable
INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ibis_version.cmake)
setup_ibis_version()

# add external dependancies
add_subdirectory(extern/doctest EXCLUDE_FROM_ALL)
add_subdirectory(extern/kokkos EXCLUDE_FROM_ALL)
add_subdirectory(extern/json EXCLUDE_FROM_ALL)
add_subdirectory(extern/spdlog EXCLUDE_FROM_ALL)
add_subdirectory(extern/CLI11 EXCLUDE_FROM_ALL)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(extern/pybind11 EXCLUDE_FROM_ALL)

# tests configuration
option(Ibis_BUILD_TESTS "Build Ibis CI tests" ON)
if (Ibis_BUILD_TESTS)
    enable_testing()
else()
    add_compile_options( -DDOCTEST_CONFIG_DISABLE )
endif(Ibis_BUILD_TESTS)

# be picky about warnings
if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options( -Wall -Wextra -Wpedantic )
endif()

# add all the source code
add_subdirectory(src/gas)
add_subdirectory(src/util)
add_subdirectory(src/linear_algebra)
add_subdirectory(src/grid)
add_subdirectory(src/finite_volume)
add_subdirectory(src/ibis/)
add_subdirectory(src/solvers)
add_subdirectory(src/io)
add_subdirectory(src/python)
add_subdirectory(src/simulation)
add_subdirectory(share)

# allow packaging to distribute a pre-compiled 
# version of the code
include(CPack)
