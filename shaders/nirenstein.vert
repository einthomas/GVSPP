#version 460

#extension GL_EXT_multiview : enable

layout(binding = 0, set = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view[5];
    mat4 projection;
} ubo;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = ubo.projection * ubo.view[gl_ViewIndex] * ubo.model * vec4(inPosition, 1.0);
}
