add_executable(fuzzer
  deterministic_random.h deterministic_random.cc deterministic_random.inl
  fuzzer.cc
  hasher.h hasher.cc hasher.inl
  mapper.h mapper.cc
  siphash.h siphash.c)
target_link_libraries(fuzzer Legion::Legion)
set_target_properties(fuzzer PROPERTIES CXX_STANDARD 17)

add_test(
  NAME fuzzer
  COMMAND $<TARGET_FILE:fuzzer> ${PROJECT_SOURCE_DIR}/tests/runner.py -- ${FUZZER_TEST_LAUNCHER} $<TARGET_FILE:fuzzer>
)
