#version 460

#extension GL_ARB_viewport_array : enable

layout(triangles, invocations = 5) in;
layout(triangle_strip, max_vertices = 3) out;

layout(binding = 0, set = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view[5];
    mat4 projection;
} ubo;

void main() {	
    int a = 3;
	for (int i = 0; i < gl_in.length(); i++) {
        vec4 pos = gl_in[i].gl_Position;
        gl_Position = ubo.projection * ubo.view[a] * ubo.model * pos;

		// Set the viewport index that the vertex will be emitted to
		gl_ViewportIndex = a;
        gl_PrimitiveID = gl_PrimitiveIDIn;

		EmitVertex();
	}

	EndPrimitive();
}
