cmake_minimum_required(VERSION 2.8)
include(${CMAKE_SOURCE_DIR}/cmake/FindModernGPU.cmake)

# gswitch
project(gswitch)

#############################################################################
# Options
#############################################################################
option(GDB "enable GDB" OFF)


#############################################################################
# Configure
#############################################################################


# find CUDA
if(${CMAKE_MINOR_VERSION} GREATER 0)
  cmake_policy(SET CMP0054 OLD)
endif()

#find mgpu
if(mgpu_INCLUDE_DIRS)
  include_directories(${mgpu_INCLUDE_DIRS})
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -I ${mgpu_INCLUDE_DIRS})
else()
  message(SEND_ERROR "ModernGPU not found")
endif()

#find openMP
find_package(OpenMP)
find_package(CUDA QUIET REQUIRED)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# confiuge CUDA flags
if(GDB)
  message("-- enable GDB.")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -arch=sm_35 -g -G -O0
    --compiler-options -Wall -Xptxas -v -Xcompiler -fopenmp -Xcompiler
    -rdynamic -lineinfo -ftz=true --expt-extended-lambda)
else() 
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -arch=sm_35 -O3 -maxrregcount=64
    --compiler-options -Wall --compiler-options -fmax-errors=5 --expt-extended-lambda)
#-Xcompiler -fopenmp -Xptxas -v
#-maxrregcount=64
endif()


message("-- CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")


#############################################################################
# subdirs
#############################################################################
include_directories(src)
add_subdirectory(src)
add_subdirectory(application)


#############################################################################
# Optimization
#############################################################################


