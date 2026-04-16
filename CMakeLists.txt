cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(LegionFuzzer)

find_package(Legion REQUIRED)

include(GNUInstallDirs)

set(FUZZER_TEST_LAUNCHER "" CACHE STRING "launcher for test commands")
set(FUZZER_TEST_ARGS "" CACHE STRING "arguments for test commands")

# Legion requires applications not be built with -DNDEBUG in debug mode:
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "-O2 -g")

enable_testing()

add_subdirectory(src)
