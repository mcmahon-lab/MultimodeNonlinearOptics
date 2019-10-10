cmake_minimum_required(VERSION 2.8.12)

project(nonlinearmedium)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Eigen
include_directories(/usr/include/eigen3)

# Generate Python module
# Pybind11
find_package(pybind11 REQUIRED)
pybind11_add_module(nonlinearmedium NonlinearMedium.cpp nlmModulePy.cpp)

# Compile options for optimization
set(COMPILE_FLAGS -Wall)
if(CMAKE_BUILD_TYPE MATCHES Release) # Release mode default
  set(COMPILE_FLAGS ${COMPILE_FLAGS} -O3 -DNDEBUG -march=native -fvisibility=hidden)
#  string(REGEX REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}") # O2 instead of O3 hack
endif()
target_compile_options(nonlinearmedium PUBLIC ${COMPILE_FLAGS})

# Create the output in the source directory
set_target_properties(nonlinearmedium PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
