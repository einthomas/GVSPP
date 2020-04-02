#version 450

layout(binding = 1) uniform sampler2D tex;
layout(binding = 7, set = 0) readonly buffer pvsBuffer {
    int pvs[];
};

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 color;

void main() {
    //color = texture(tex, fragTexCoord);

    /*
    bool found = false;
    for (int i = 0; i < 10; i++) {
        if (pvs[i] == gl_PrimitiveID) {
            found = true;            
            break;
        }
    }
    if (found) {
        color = vec4(1.0);
    } else {
        discard;
    }
    */

    color = vec4(1.0);
}
