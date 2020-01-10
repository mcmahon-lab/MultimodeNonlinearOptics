cmake_minimum_required(VERSION 2.8.12)

project(nonlinearmedium)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Cmake module path for finding nonstandard libraries
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")

# Eigen (might be located in various places)
include_directories(/usr/include/eigen3 /usr/local/include/eigen3)

# FFTW (optional)
find_package(FFTW QUIET)

# Pybind11
find_package(pybind11 REQUIRED)

# Generate Python module
pybind11_add_module(nonlinearmedium NonlinearMedium.cpp nlmModulePy.cpp)

# Compile options for optimization
set(COMPILE_FLAGS -Wall)
if(CMAKE_BUILD_TYPE MATCHES Release) # Release mode default
  set(COMPILE_FLAGS ${COMPILE_FLAGS} -O3 -DNDEBUG -march=native -fvisibility=hidden)
#  string(REGEX REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}") # O2 instead of O3 hack
endif()

# Use FFTW if found
if (FFTW_FOUND)
  set(COMPILE_FLAGS ${COMPILE_FLAGS} -DEIGEN_FFTW_DEFAULT -lfftw3)
  target_link_libraries(nonlinearmedium PRIVATE fftw3)
  message("-- FFTW installation found!")
endif()

target_compile_options(nonlinearmedium PUBLIC ${COMPILE_FLAGS})

# Create the output in the source directory
set_target_properties(nonlinearmedium PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
