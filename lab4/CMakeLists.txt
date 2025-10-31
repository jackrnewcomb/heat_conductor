cmake_minimum_required(VERSION 3.18)
project(Lab4 LANGUAGES CXX) #CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Gather source files
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
include_directories(include)

# Create the executable
add_executable(main ${SRC_FILES})

# Ensure separable compilation for CUDA (lets you split kernels across files)
#set_target_properties(heat_conduction PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON
#)

# Optional: enable extra warnings for debugging
