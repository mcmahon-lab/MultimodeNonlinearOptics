cmake_minimum_required(VERSION 3.5.0)

project(nonlinearmedium)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Cmake module path for finding nonstandard libraries
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")

# Eigen (might be located in various places)
include_directories(/usr/include/eigen3 /usr/local/include/eigen3)

# FFTW (optional)
find_package(FFTW QUIET)

# Pybind11
find_package(pybind11 REQUIRED)

# LibTorch
find_package(Torch QUIET)

# Generate Python module
include_directories(src solvers)
file(GLOB sourceFiles "src/*.cpp")
pybind11_add_module(nonlinearmedium ${sourceFiles})
if (Torch_FOUND)
  message("-- Torch installation found!")
  pybind11_add_module(nonlinearmediumGPU ${sourceFiles})
  target_compile_definitions(nonlinearmediumGPU PUBLIC USE_GPU)
else()
  message("-- Torch not found")
endif()

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
else()
  message("-- FFTW not found")
endif()

target_compile_options(nonlinearmedium PUBLIC ${COMPILE_FLAGS})

# Create the output in the source directory
set_target_properties(nonlinearmedium PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# Use LibTorch if found
if (Torch_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  include_directories("${TORCH_INCLUDE_DIRS}")
  target_link_libraries(nonlinearmediumGPU PRIVATE "${TORCH_LIBRARIES}")
  target_link_libraries(nonlinearmediumGPU PRIVATE fftw3)

  target_compile_options(nonlinearmediumGPU PUBLIC ${COMPILE_FLAGS})
  set_target_properties(nonlinearmediumGPU PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
