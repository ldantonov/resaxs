cmake_minimum_required(VERSION 2.8)
project(test_recalc)

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

file(GLOB _all_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  *.cpp)

add_executable(test_recalc ${_all_sources})

target_link_libraries(test_recalc resaxslib test_common ${OPENCL_LIBRARY})

add_custom_command(TARGET test_recalc POST_BUILD
    COMMAND    ${CMAKE_COMMAND} ARGS -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../saxs_parameters $<TARGET_FILE_DIR:test_recalc>
    COMMENT "Copying support files...")
