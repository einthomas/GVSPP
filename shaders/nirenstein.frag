#version 460

layout(location = 0) out int color;

void main() {
    color = int(gl_PrimitiveID);
}
