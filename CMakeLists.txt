cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(LegionFuzzer)

find_package(Legion REQUIRED)
find_package(Threads REQUIRED)

include(GNUInstallDirs)

set(FUZZER_TEST_LAUNCHER "" CACHE STRING "launcher for test commands")
set(FUZZER_TEST_ARGS "" CACHE STRING "arguments for test commands")

# Legion requires applications not be built with -DNDEBUG in debug mode:
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "-O2 -g")

enable_testing()

add_subdirectory(src)

# The shims are x86-64 Linux only (signal handling, ptrace registers, futex).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  add_subdirectory(sidecar)
endif()
