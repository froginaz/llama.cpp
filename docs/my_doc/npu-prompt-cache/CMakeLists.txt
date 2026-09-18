# npu-prompt-cache - builds like llama.cpp on every platform:
#   cmake -B build
#   cmake --build build --config Release
#   ctest --test-dir build --build-config Release --output-on-failure
#
# On Windows use the same three commands from a VS developer prompt (MSVC),
# exactly as for llama.cpp. The GNU Makefile remains for quick POSIX use.

cmake_minimum_required(VERSION 3.14)
project(npu_prompt_cache CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The tests are assert-based. MSVC/GCC Release flags define NDEBUG, which
# compiles the asserts out and turns every test into a false pass - scrub it.
foreach(flags_var
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_MINSIZEREL)
    if(DEFINED ${flags_var})
        string(REPLACE "/DNDEBUG" "" ${flags_var} "${${flags_var}}")
        string(REPLACE "-DNDEBUG" "" ${flags_var} "${${flags_var}}")
    endif()
endforeach()

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

find_package(Threads REQUIRED)

add_library(npu_pc STATIC
    src/kv_prefix_cache.cpp
    src/kv_prefix_cache.hpp
    src/backend_stream_tracker.hpp)
target_include_directories(npu_pc PUBLIC src)
target_link_libraries(npu_pc PUBLIC Threads::Threads)

enable_testing()

foreach(t test_kv_prefix_cache test_lcp_ubatch_split test_backend_stream_tracker)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE npu_pc)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
