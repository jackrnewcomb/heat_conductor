cmake_minimum_required(VERSION 3.18)
project(Lab4 LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include headers
include_directories(include)

# Gather source files
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)

# Create the executable
add_executable(heat-dispersion ${SRC_FILES})
