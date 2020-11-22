cmake_minimum_required(VERSION 3.11)

project(Vulkan_RTX_Master_Qt_CMake LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(GLSLC_PATH ${CMAKE_SOURCE_DIR}/external/vulkan/1.2.154.0/Bin/glslc)
set(SHADERS_PATH shaders)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -D_USE_MATH_DEFINES")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_XCB_KHR -DUSEVULKANSDK -DVK_ENABLE_BETA_EXTENSIONS -DVK_USE_PLATFORM_XLIB_KHR -DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 -DGLFW_INCLUDE_VULKAN")

#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; " -gencode arch=compute_75,code=sm_75")


#include_directories(external/vulkan/1.2.154.0/Include)
#include_directories(/home/thomas/Documents/MasterNew/external/vulkan/1.2.154.0/x86_64/include)
include_directories($ENV{VULKAN_SDK}/include)
include_directories(${CMAKE_SOURCE_DIR}/external/glm)
include_directories(${CMAKE_SOURCE_DIR}/external/stb)
include_directories(${CMAKE_SOURCE_DIR}/external/tinyobjloader)
include_directories(${CMAKE_SOURCE_DIR}/external/glfw)

###include_directories(E:/VulkanSDK/1.2.135.0/Include)
###include_directories(C:/Users/thoma/Documents/dev/cpplib/glm)
###include_directories(C:/Users/thoma/Documents/dev/cpplib/stb)
###include_directories(C:/Users/thoma/Documents/dev/cpplib/tinyobjloader)
###include_directories(C:/Users/thoma/Documents/dev/cpplib/glfw-3.3.2.bin.WIN64/include)
#include_directories(C:/VulkanSDK/1.2.131.2/Include)
#include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include")
#include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include")

file(GLOB SRC_FILES
    "*.h"
    "*.hpp"
    "*.cpp"
    "*.cu"
    "gpuHashTable/*"
)
add_executable(Vulkan_RTX_Master_Qt_CMake ${SRC_FILES})

#set_target_properties(Vulkan_RTX_Master_Qt_CMake PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
#target_compile_options(Vulkan_RTX_Master_Qt_CMake PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)

#target_link_libraries(Vulkan_RTX_Master_Qt_CMake Qt5::Core)
#target_link_libraries(Vulkan_RTX_Master_Qt_CMake Qt5::Gui)
#target_link_libraries(Vulkan_RTX_Master_Qt_CMake "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/x64/cuda.lib")
#target_link_libraries(Vulkan_RTX_Master_Qt_CMake C:/Users/thoma/Documents/dev/cpplib/glfw-3.3.2.bin.WIN64/lib-mingw-w64/libglfw3.a)
###target_link_libraries(Vulkan_RTX_Master_Qt_CMake C:/Users/thoma/Documents/dev/cpplib/glfw-3.3.2.bin.WIN64/lib-vc2019/glfw3.lib)
###target_link_libraries(Vulkan_RTX_Master_Qt_CMake E:/VulkanSDK/1.2.135.0/Lib/vulkan-1.lib)
target_link_libraries(Vulkan_RTX_Master_Qt_CMake -lGL -lGLU ${CMAKE_SOURCE_DIR}/external/glfw/libglfw3.a -lX11 -lXxf86vm -lXrandr -lpthread -lXi -ldl $ENV{VULKAN_SDK}/lib/libvulkan.so)



#if (UNIX)
#    add_custom_command(
#        TARGET Vulkan_RTX_Master_Qt_CMake
#        POST_BUILD
#        COMMAND /bin/sh ${SHADERS_PATH}compileShaders.sh ${GLSLC_PATH} ${SHADERS_PATH}
#    )
#endif (UNIX)
#
#if (WIN32)
#    add_custom_command(
#        TARGET Vulkan_RTX_Master_Qt_CMake
#        POST_BUILD
#        COMMAND cmd /c ${SHADERS_PATH}compileShaders.bat
#    )
#endif (WIN32)

#add_definitions(-DUSE_CUDA)
