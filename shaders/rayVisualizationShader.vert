#version 460

layout(binding = 0, set = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec3 inColor;

layout(location = 0) flat out vec3 fragColor;

void main() {
    gl_Position = ubo.projection * ubo.view * vec4(inPosition, 1.0);
    fragColor = inColor;
}
