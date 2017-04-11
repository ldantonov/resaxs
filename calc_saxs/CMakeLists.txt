cmake_minimum_required(VERSION 2.8)
project(calc_saxs)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++11 HAS_C11_FLAG)
check_cxx_compiler_flag(-std=c++0x HAS_C0X_FLAG)
if (HAS_C11_FLAG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (HAS_C0X_FLAG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../resaxslib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../test_common/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../pdblib/include)

file(GLOB _all_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  *.cpp)

add_executable(calc_saxs ${_all_sources})

if (WIN32)
  set(EXTRA_LIBS "")
else()
  set(EXTRA_LIBS -ldl)
endif()

target_link_libraries(calc_saxs resaxslib test_common pdblib ${OPENCL_LIBRARY} ${EXTRA_LIBS})

add_custom_command(TARGET calc_saxs POST_BUILD
    COMMAND    ${CMAKE_COMMAND} ARGS -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../saxs_parameters $<TARGET_FILE_DIR:calc_saxs>
    COMMENT "Copying support files...")
