cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(LegionFuzzer)

find_package(Legion REQUIRED)

include(GNUInstallDirs)

set(FUZZER_TEST_LAUNCHER "" CACHE STRING "launcher for test commands")

# Legion requires applications not be built with -DNDEBUG in debug mode:
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

enable_testing()

add_subdirectory(src)
