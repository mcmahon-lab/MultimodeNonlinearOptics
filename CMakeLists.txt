cmake_minimum_required(VERSION 3.0.0)

project(nonlinearmedium)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Cmake module path for finding nonstandard libraries
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")

# Eigen (might be located in various places)
include_directories(/usr/include/eigen3 /usr/local/include/eigen3)

# Hack: recent GCC has issues with our usage of EigenFFT
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

# FFTW (optional)
find_package(FFTW QUIET)

# Pybind11
find_package(pybind11 REQUIRED)

# Generate Python module
include_directories(src solvers)
file(GLOB sourceFiles "src/*.cpp")
pybind11_add_module(nonlinearmedium ${sourceFiles})

# Compile options for optimization
set(COMPILE_FLAGS -Wall)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -fvisibility=hidden -flto -fno-math-errno")

# Check what is/is not inlined
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winline")

# Use FFTW if found
if (FFTW_FOUND)
  set(COMPILE_FLAGS ${COMPILE_FLAGS} -DEIGEN_FFTW_DEFAULT -lfftw3)
  target_link_libraries(nonlinearmedium PRIVATE fftw3)
  message("-- FFTW installation found!")
endif()

target_compile_options(nonlinearmedium PUBLIC ${COMPILE_FLAGS})

# Create the output in the source directory
set_target_properties(nonlinearmedium PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
