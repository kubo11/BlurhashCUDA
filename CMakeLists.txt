cmake_minimum_required(VERSION 3.18)

project(blurhash
    VERSION 1.0
    DESCRIPTION "Blurhash encoder/decoder implemented on cpu and gpu"
    LANGUAGES C CUDA
)

set(BLURHASH_LIBRARY_NAME blurhash)
set(BLURHASH_STANDALONE_NAME blurhashed)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

option(BLURHASH_BUILD_STANDALONE "Build standalone executable encoder/decoder" OFF)

if(MSVC)
    add_compile_definitions(_SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS)
endif()

set(BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(external)
add_subdirectory(src)