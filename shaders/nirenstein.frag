#version 460

#include "rt/defines.glsl"

layout(location = 0) out int color;

void main() {
    color = int(gl_PrimitiveID);
}
