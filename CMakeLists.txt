cmake_minimum_required(VERSION 2.8)
project(resaxs_suite)

# enable local modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR} "tclap-1.2.1/include")

add_subdirectory(resaxslib)
add_subdirectory(test_common)
add_subdirectory(test_resaxs)
add_subdirectory(test_recalc)
add_subdirectory(pdblib)
add_subdirectory(calc_saxs)
