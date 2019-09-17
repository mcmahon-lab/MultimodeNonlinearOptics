cmake_minimum_required(VERSION 2.8.12)

project(nonlinearmedium)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Eigen
include_directories(/usr/include/eigen3)

# Generate Python module
# Pybind11
find_package(pybind11 REQUIRED)
pybind11_add_module(nonlinearmedium NonlinearMedium.cpp)

# Compile options for optimization
add_compile_options(-Wall)
if(NOT CMAKE_BUILD_TYPE MATCHES Debug) # Release mode default
  add_compile_options(-O3)
  add_compile_options(-fvisibility=hidden)
endif()

# Create the output in the source directory
set_target_properties(nonlinearmedium PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
