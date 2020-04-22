#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInNV vec4 hitInfo; //uint hitPrimitiveID;

void main() {
    //hitPrimitiveID = 0;
    hitInfo = vec4(-1.0);
}
