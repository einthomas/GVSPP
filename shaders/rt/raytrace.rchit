#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

struct Vertex {
    vec3 pos;
    vec3 worldPos;
    vec3 normal;
    vec3 color;
    vec2 uv;
};

layout(location = 0) rayPayloadInNV vec4 hitInfo;

layout(binding = 1, set = 0) uniform cameraProperties {
    mat4 model;     // TODO: This is called camera properties, however this model matrix doesn't have anything to do with the camera, this is model specific. Instead, use a buffer of model matrices and access the buffer via gl_InstanceID (in case there are different models/instances) (see nvidia Vulkan tutorial)
    mat4 view;
    mat4 projection;
} camera;
layout(binding = 2, set = 0) buffer Vertices {
    vec4 v[];
} vertices;
layout(binding = 3, set = 0) buffer Indices {
    uint i[];
} indices;

hitAttributeNV vec3 attribs;

//#include "util.glsl"

void main() {
    const ivec3 index = ivec3(
        indices.i[3 * gl_PrimitiveID + 0],
        indices.i[3 * gl_PrimitiveID + 1],
        indices.i[3 * gl_PrimitiveID + 2]
    );

    const vec4 d0 = vertices.v[3 * index.x + 0];
    const vec4 d1 = vertices.v[3 * index.y + 1];
    const vec4 d2 = vertices.v[3 * index.z + 2];

    const vec3 barycentricCoords = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    const vec3 pos = d0.xyz * barycentricCoords.x + d1.xyz * barycentricCoords.y + d2.xyz * barycentricCoords.z;
    const vec3 worldPos = vec3(camera.model * vec4(pos, 1.0));

    hitInfo = vec4(worldPos, gl_PrimitiveID);
}
