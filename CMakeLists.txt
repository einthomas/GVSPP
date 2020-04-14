cmake_minimum_required(VERSION 3.5)

project(Vulkan_RTX_Master_Qt_CMake LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GLSLC_PATH E:/VulkanSDK/1.2.135.0/Bin/glslc)
set(SHADERS_PATH C:/Users/thoma/Documents/dev/Master/Vulkan_RTX_Master_Qt_CMake/shaders/)

find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)

include_directories(E:/VulkanSDK/1.2.135.0/Include)
include_directories(C:/Users/thoma/Documents/dev/cpplib/glm)
include_directories(C:/Users/thoma/Documents/dev/cpplib/stb)
include_directories(C:/Users/thoma/Documents/dev/cpplib/tinyobjloader)

file(GLOB SRC_FILES
    "*.h"
    "*.hpp"
    "*.cpp"
)
add_executable(Vulkan_RTX_Master_Qt_CMake ${SRC_FILES})

target_link_libraries(Vulkan_RTX_Master_Qt_CMake Qt5::Core)
target_link_libraries(Vulkan_RTX_Master_Qt_CMake Qt5::Gui)
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
