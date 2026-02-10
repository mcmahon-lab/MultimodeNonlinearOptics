cmake_minimum_required(VERSION 3.8.0)

project(nonlinearmedium)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Cmake module path for finding nonstandard libraries
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")

# Eigen (might be located in various places)
include_directories(/usr/include/eigen3 /usr/local/include/eigen3)

#set(PYTHON_EXECUTABLE /directory_to/bin/python)

# FFTW (optional)
find_package(FFTW QUIET)

# Pybind11
#set(pybind11_DIR "directory_to/share/cmake/pybind11/")
find_package(pybind11 REQUIRED)

# LibTorch
#set(CUDA_TOOLKIT_ROOT_DIR "/directory_to/cuda-X")
#set(CMAKE_CUDA_COMPILER "/directory_to/nvcc")
#set(Torch_DIR /directory_to/libtorch/share/cmake/Torch)
find_package(Torch QUIET)

# Generate Python module
include_directories(src solvers)
file(GLOB sourceFiles "src/*.cpp")
pybind11_add_module(nonlinearmedium ${sourceFiles})
if (Torch_FOUND)
  message("-- Torch installation found!")
  enable_language(CUDA)
  file(GLOB sourceFilesGPU "src/*.cpp" "src/*.cu")
  pybind11_add_module(nonlinearmediumGPU ${sourceFilesGPU})
  target_compile_definitions(nonlinearmediumGPU PUBLIC USE_GPU)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dc -O3 -DNDEBUG -march=native -fvisibility=hidden -flto -fno-math-errno") # -dlto
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
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
