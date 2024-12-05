cmake_minimum_required(VERSION 3.5)
include(${CMAKE_SOURCE_DIR}/cmake/FindModernGPU.cmake)

# gswitch
project(gswitch)

#############################################################################
# Options
#############################################################################

#############################################################################
# Configure
#############################################################################

set(DEFAULT_BUILD_TYPE "Release")

#find mgpu
if(mgpu_INCLUDE_DIRS)
  include_directories(${mgpu_INCLUDE_DIRS})
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -I ${mgpu_INCLUDE_DIRS})
else()
  message(SEND_ERROR "ModernGPU not found")
endif()

# find openMP
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# find cuda
find_package(CUDA QUIET REQUIRED)
if (CUDA_VERSION VERSION_LESS "9.0")
  message(WARNING "Disable gpu support because cuda >= 9.0 not found")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  if(${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
    cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
    message(STATUS "CUDA_ARCH = ${ARCH_FLAGS}")
    string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
    string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
  else()
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    message(STATUS "CUDA_ARCH = ${CUDA_ARCH_LIST}")
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
    list(GET CUDA_ARCH_LIST 0 CUDA_ARCH_CODE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CUDA_ARCH_CODE}")
  endif()

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-lineinfo;-Xcompiler;-ggdb;-std=c++11;--expt-extended-lambda)
  else ()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11;-maxrregcount=64;--expt-extended-lambda)
  endif ()

  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_SEPARABLE_COMPILATION OFF)
  
  message(STATUS "Host Compiler: ${CUDA_HOST_COMPILER}")
endif()

# if(GDB)
#   message("-- enable GDB.")
#   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -g -G -O0
#     --compiler-options -Wall -Xptxas -v -Xcompiler -fopenmp -Xcompiler
#     -rdynamic -lineinfo -ftz=true)
# else() 
#   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -O3 -maxrregcount=64
#     --compiler-options -Wall --compiler-options -fmax-errors=5 --expt-extended-lambda)
# #-Xcompiler -fopenmp -Xptxas -v
# #-maxrregcount=64
# endif()

message("-- CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")


#############################################################################
# subdirs
#############################################################################
include_directories(src)
add_subdirectory(src)
add_subdirectory(application)
