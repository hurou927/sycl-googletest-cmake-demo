enable_testing()
include_directories(${PROJECT_SOURCE_DIR}/include)
cmake_minimum_required(VERSION 3.4.3)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)



project(sycl_gtest)

set(CMAKE_CXX_COMPILER "dpcpp")
set(CMAKE_C_COMPILE "clang")

set(CMAKE_CXX_STANDARD 20)


# add_compile_options(-Wall -Wextra -pedantic -Werror)
# add_compile_options(-O3)
# add_compile_options(-Wno)
# add_compile_options(-Werror -Wno-tautological-constant-compare)
# add_compile_options(-Werror -Wno-error=pass-failed -Wno-tautological-constant-compare)


## Intel
# https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html
# ❯ sudo lshw -class processor
#   *-cpu
#        description: CPU
#        product: 11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz
# set(CMAKE_CXX_FLAGS "-fsycl-targets=spir64_gen -Xs \"-device tgllp\"")

## CUDA
# set(CMAKE_CXX_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
add_subdirectory(test)
