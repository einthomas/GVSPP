cmake_minimum_required(VERSION 3.11)

project(Vulkan_RTX_Master_Qt_CMake LANGUAGES CXX CUDA)
#project(Vulkan_RTX_Master_Qt_CMake LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(GLSLC_PATH E:/VulkanSDK/1.2.135.0/Bin/glslc)
set(SHADERS_PATH C:/Users/thoma/Documents/dev/Master/Vulkan_RTX_Master_Qt_CMake/shaders/)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; " -gencode arch=compute_75,code=sm_75")

#find_package(Qt5Core REQUIRED)
#find_package(Qt5Gui REQUIRED)

include_directories(E:/VulkanSDK/1.2.135.0/Include)
include_directories(C:/Users/thoma/Documents/dev/cpplib/glm)
include_directories(C:/Users/thoma/Documents/dev/cpplib/stb)
include_directories(C:/Users/thoma/Documents/dev/cpplib/tinyobjloader)
include_directories(C:/Users/thoma/Documents/dev/cpplib/glfw-3.3.2.bin.WIN64/include)
#include_directories(C:/VulkanSDK/1.2.131.2/Include)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include")
#include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include")


file(GLOB SRC_FILES
    "*.h"
    "*.hpp"
    "*.cpp"
    "*.cu"
)
add_executable(Vulkan_RTX_Master_Qt_CMake ${SRC_FILES})

set_target_properties(Vulkan_RTX_Master_Qt_CMake PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
target_compile_options(Vulkan_RTX_Master_Qt_CMake PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)

#target_link_libraries(Vulkan_RTX_Master_Qt_CMake Qt5::Core)
#target_link_libraries(Vulkan_RTX_Master_Qt_CMake Qt5::Gui)
target_link_libraries(Vulkan_RTX_Master_Qt_CMake "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/x64/cuda.lib")
#target_link_libraries(Vulkan_RTX_Master_Qt_CMake C:/Users/thoma/Documents/dev/cpplib/glfw-3.3.2.bin.WIN64/lib-mingw-w64/libglfw3.a)
target_link_libraries(Vulkan_RTX_Master_Qt_CMake C:/Users/thoma/Documents/dev/cpplib/glfw-3.3.2.bin.WIN64/lib-vc2019/glfw3.lib)
target_link_libraries(Vulkan_RTX_Master_Qt_CMake E:/VulkanSDK/1.2.135.0/Lib/vulkan-1.lib)


if (UNIX)
    add_custom_command(
        TARGET Vulkan_RTX_Master_Qt_CMake
        POST_BUILD
        COMMAND /bin/sh ${SHADERS_PATH}compileShaders.sh ${GLSLC_PATH} ${SHADERS_PATH}
    )
endif (UNIX)

if (WIN32)
    add_custom_command(
        TARGET Vulkan_RTX_Master_Qt_CMake
        POST_BUILD
        COMMAND cmd /c ${SHADERS_PATH}compileShaders.bat
    )
endif (WIN32)

add_definitions(-DUSE_CUDA)
