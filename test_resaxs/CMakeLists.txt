cmake_minimum_required(VERSION 2.8)
project(test_resaxs)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++11 HAS_C11_FLAG)
check_cxx_compiler_flag(-std=c++0x HAS_C0X_FLAG)
if (HAS_C11_FLAG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (HAS_C0X_FLAG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../resaxslib/include)

file(GLOB _all_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  *.cpp)

add_executable(test_resaxs ${_all_sources})

target_link_libraries(test_resaxs resaxslib ${OPENCL_LIBRARY})

file(GLOB _test_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    test*.txt)

foreach(test_file ${_test_files})
    add_custom_command(TARGET test_resaxs POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${test_file} $<TARGET_FILE_DIR:test_resaxs>
        COMMENT "Copying support files...")

endforeach(test_file)
