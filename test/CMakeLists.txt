enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*test.cpp)

add_executable(
  UnitTestExecutor
  ${TEST_SRCS}
)
target_link_libraries(
  UnitTestExecutor
  gtest_main
)

# add_compile_options(-Werror -Wno-error=pass-failed -Wno-tautological-constant-compare)
include(GoogleTest)
gtest_discover_tests(
  UnitTestExecutor
)


