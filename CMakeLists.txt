cmake_minimum_required(VERSION 3.11)

project(Vulkan_RTX_Master_Qt_CMake LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -D_USE_MATH_DEFINES")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_XCB_KHR -DUSEVULKANSDK -DVK_ENABLE_BETA_EXTENSIONS -DVK_USE_PLATFORM_XLIB_KHR -DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 -DGLFW_INCLUDE_VULKAN")

include_directories($ENV{VULKAN_SDK}/include)
include_directories(${CMAKE_SOURCE_DIR}/external/glm)
include_directories(${CMAKE_SOURCE_DIR}/external/stb)
include_directories(${CMAKE_SOURCE_DIR}/external/tinyobjloader)
include_directories(${CMAKE_SOURCE_DIR}/external/glfw)

file(GLOB SRC_FILES
    "*.h"
    "*.hpp"
    "*.cpp"
    "*.cu"
    "gpuHashTable/*"
)
add_executable(Vulkan_RTX_Master_Qt_CMake ${SRC_FILES})

target_link_libraries(Vulkan_RTX_Master_Qt_CMake -lGL -lGLU ${CMAKE_SOURCE_DIR}/external/glfw/libglfw3.a -lX11 -lXxf86vm -lXrandr -lpthread -lXi -ldl $ENV{VULKAN_SDK}/lib/libvulkan.so)
