# CMake project for remote_sim_gl
cmake_minimum_required(VERSION 3.10)
project(RemoteSimGL LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 89) # 89 for Ada Lovelace architecture GPU

include(FetchContent)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.3.8
)

FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	0af55ccecd98d4e5a8d1fad7de25ba429d60e863 #refs/tags/1.0.1
)

FetchContent_MakeAvailable(glm)

find_package(CUDAToolkit REQUIRED)

# Define source files
file(GLOB SRC_FILES src/main.cpp src/gl/*.cpp)
file(GLOB CU_FILES src/cuda/*.cu)
file(GLOB HEADER_FILES include/**/*.h include/**/*.hpp include/**/*.cuh)
set(GLAD_SRC deps/glad/src/gl.c)

# Create the executable with source files
add_executable(sim ${SRC_FILES} ${CU_FILES} ${HEADER_FILES} ${GLAD_SRC})

# Include directories
target_include_directories(sim PRIVATE include 
  include
  deps/glad/include)

# Link libraries
target_link_libraries(sim PRIVATE
  CUDA::cuda_driver
  CUDA::cudart
  glfw
  opengl32
  gdi32
  user32
  glm::glm
)

# Enable separate compilation for CUDA files
set_target_properties(sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
