#version 460

#extension GL_EXT_multiview : enable

#include "rt/defines.glsl"

layout(binding = 0, set = 0) uniform UniformBufferObject {
    mat4 model;
    #ifdef NIRENSTEIN_USE_MULTI_VIEW_RENDERING
        mat4 view[5];
    #else
        mat4 view;
    #endif
    mat4 projection;
} ubo;

layout(location = 0) in vec3 inPosition;

void main() {
    #ifdef NIRENSTEIN_USE_MULTI_VIEW_RENDERING
        gl_Position = ubo.projection * ubo.view[gl_ViewIndex] * ubo.model * vec4(inPosition, 1.0);
    #else
        gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPosition, 1.0);
    #endif
}