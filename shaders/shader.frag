#version 460

layout(binding = 0, set = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(binding = 1) uniform sampler2D tex;

layout(location = 0) flat in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 worldPos;

layout(location = 0) out vec4 color;

layout(push_constant) uniform PushConstants {
	layout(offset = 64) bool shadedRendering;
} pushConstants;

void main() {
    if (pushConstants.shadedRendering) {
        mat4 m = inverse(ubo.view);
        vec3 cameraWorldPos = vec3(m[3][0], m[3][1], m[3][2]);
        color = vec4(fragColor * max(0.0f, dot(fragNormal, normalize(cameraWorldPos - worldPos))), 1.0f);
    } else {
        color = vec4(fragColor, 1.0f);
    }
}
