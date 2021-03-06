cmake_minimum_required(VERSION 3.17)
project(gpu_compressstore2_avx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(INCLUDES
    src
)

set(SOURCES
    src/main.cpp
    src/fast_prng.cpp
)

add_executable(gpu_compressstore2_avx ${SOURCES})

option(AVXPOWER "use avx512" OFF)
if(AVXPOWER)
    target_compile_definitions(gpu_compressstore2_avx PUBLIC AVXPOWER)
    set(CMAKE_CXX_FLAGS "-march=native")
endif()

target_include_directories(gpu_compressstore2_avx PRIVATE ${INCLUDES})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set_target_properties(gpu_compressstore2_avx PROPERTIES EXPORT_COMPILE_COMMANDS true)
