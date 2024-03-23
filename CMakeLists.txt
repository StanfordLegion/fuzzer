cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(LegionFuzzer)

find_package(Legion REQUIRED)

include(GNUInstallDirs)

set(FUZZER_TEST_LAUNCHER "" CACHE STRING "launcher for test commands")

enable_testing()

add_subdirectory(src)
