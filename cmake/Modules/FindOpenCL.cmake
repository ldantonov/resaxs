set(opencl_search_path
    ${opencl_search_path}
    $ENV{OPENCL_DIR}
    $ENV{AMDAPPSDKROOT}
    $ENV{NVSDKCOMPUTE_ROOT}     # NVIDIA on Windows
    $ENV{CUDA_PATH_V6_5}
    $ENV{CUDA_PATH}
    $ENV{INTELOCLSDKROOT}
    $ENV{ATISTREAMSDKROOT}      # Legacy Stream SDK
    "/usr/local/cuda"           # Default path NVIDIA on Linux
    "/usr/local/streamsdk"
    "/usr")    

find_path(OPENCL_INCLUDE_DIR
    NAMES
            CL/cl.h OpenCL/cl.h
    PATHS
        ${opencl_search_path}
    PATH_SUFFIXES include OpenCL/common/inc
)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(lib_suffixes "/lib/x86" "/lib/Win32" "/OpenCL/common/lib/Win32")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(lib_suffixes "/lib/x86_64" "/lib/x64" "/OpenCL/common/lib/x64")
endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

find_library(OPENCL_LIBRARY
    NAMES OpenCL
    PATHS
        ${OPENCL_LIB_SEARCH_PATH}
        ${opencl_search_path}
    PATH_SUFFIXES ${lib_suffixes} lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCL DEFAULT_MSG OPENCL_LIBRARY OPENCL_INCLUDE_DIR)

if(OPENCL_FOUND)
  set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
  mark_as_advanced(CLEAR OPENCL_INCLUDE_DIR)
  mark_as_advanced(CLEAR OPENCL_LIBRARY)
else(OPENCL_FOUND)
  set(OPENCL_LIBRARIES)
  mark_as_advanced(OPENCL_INCLUDE_DIR)
  mark_as_advanced(OPENCL_LIBRARY)
endif(OPENCL_FOUND)
