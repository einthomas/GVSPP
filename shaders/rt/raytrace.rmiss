#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInNV uvec3 hitValue;

void main() {
    hitValue = uvec3(0, 0, 0);
}
