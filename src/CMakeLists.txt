set(CPU_SOURCES
  deterministic_random.h deterministic_random.cc deterministic_random.inl
  fuzzer.cc
  hasher.h hasher.cc hasher.inl
  mapper.h mapper.cc
  siphash.h siphash.c
  tasks.h tasks.cc)

if(Legion_USE_CUDA)
  enable_language(CUDA)
  set(GPU_SOURCES tasks_gpu.h tasks_gpu.cu)
  add_executable(fuzzer ${CPU_SOURCES} ${GPU_SOURCES})
  set_target_properties(fuzzer PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
elseif(Legion_USE_HIP)
  if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    # nvidia HIP_PLATFORM requies 3.28+ for enable_language(HIP) to work properly
    cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
  endif()
  enable_language(HIP)
  set(GPU_SOURCES tasks_gpu.h tasks_gpu.cu)
  add_executable(fuzzer ${CPU_SOURCES} ${GPU_SOURCES})
  set_target_properties(fuzzer PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
else()
  add_executable(fuzzer ${CPU_SOURCES})
endif()

target_link_libraries(fuzzer Legion::Legion)
set_target_properties(fuzzer PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

add_test(
  NAME fuzzer
  COMMAND $<TARGET_FILE:fuzzer> ${PROJECT_SOURCE_DIR}/tests/runner.py -- ${FUZZER_TEST_LAUNCHER} $<TARGET_FILE:fuzzer> ${FUZZER_TEST_ARGS}
)
