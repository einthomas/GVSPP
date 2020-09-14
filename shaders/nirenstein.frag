#version 460

#extension GL_EXT_multiview : enable

#include "rt/defines.glsl"

layout(location = 0) out int color;

void main() {
    #ifdef NIRENSTEIN_USE_MULTI_VIEW_RENDERING
        if (
            ((gl_ViewIndex == 1 || gl_ViewIndex == 2) && gl_FragCoord.x > 512) ||
            ((gl_ViewIndex == 3 || gl_ViewIndex == 4) && gl_FragCoord.y > 512)
        ) {
            color = -1;
        } else {
            color = int(gl_PrimitiveID);
        }
    #else
        color = int(gl_PrimitiveID);
    #endif
}
