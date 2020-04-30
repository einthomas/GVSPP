#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec3 color;
    vec2 uv;
};

layout(location = 0) rayPayloadInNV vec4 hitInfo; //hitPrimitiveID;

layout(binding = 2, set = 0) uniform cameraProperties {
    mat4 model;     // TODO: This is called camera properties, however this model matrix doesn't have anything to do with the camera, this is model specific. Instead, use a buffer of model matrices and access the buffer via gl_InstanceID (in case there are different models/instances) (see nvidia Vulkan tutorial)
    mat4 view;
    mat4 projection;
} camera;
layout(binding = 3, set = 0) buffer Vertices {
    vec4 v[];
} vertices;
layout(binding = 4, set = 0) buffer Indices {
    uint i[];
} indices;

/*
layout(binding = 1, set = 1) buffer Vertices {
    vec4 v[];
} vertices;
*/

hitAttributeNV vec3 attribs;

Vertex unpackVertexData(int index) {
    vec4 d0 = vertices.v[3 * index + 0];
    vec4 d1 = vertices.v[3 * index + 1];
    vec4 d2 = vertices.v[3 * index + 2];

    Vertex vertex;
    vertex.pos = d0.xyz;
    vertex.normal = vec3(d0.w, d1.xy);
    vertex.color = vec3(d1.zw, d2.x);
    vertex.uv = d2.yz;

    return vertex;
}

void main() {
    /*
    vec3 barycentricCoords = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    ivec3 index = ivec3(
        indices.i[3 * gl_PrimitiveID + 0],
        indices.i[3 * gl_PrimitiveID + 1],
        indices.i[3 * gl_PrimitiveID + 2]
    );
	Vertex v0 = unpackVertexData(index.x);
	Vertex v1 = unpackVertexData(index.y);
	Vertex v2 = unpackVertexData(index.z);

    vec3 normal = normalize(
        v0.normal * barycentricCoords.x +
        + v1.normal * barycentricCoords.y
        + v2.normal * barycentricCoords.z
    );

    vec3 pos = normalize(
        v0.pos * barycentricCoords.x +
        + v1.pos * barycentricCoords.y
        + v2.pos * barycentricCoords.z
    );
    */

    // Diffuse shading
    //vec3 lightPos = vec3(30.0, 30.0, -30.0);
    //hitValue = max(0.0, dot(normalize(pos - lightPos), normal)) * vec3(1.0) + vec3(0.15);

    ivec3 index = ivec3(
        indices.i[3 * gl_PrimitiveID + 0],
        indices.i[3 * gl_PrimitiveID + 1],
        indices.i[3 * gl_PrimitiveID + 2]
    );
	Vertex v0 = unpackVertexData(index.x);
	Vertex v1 = unpackVertexData(index.y);
	Vertex v2 = unpackVertexData(index.z);

    vec3 barycentricCoords = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 pos = v0.pos * barycentricCoords.x + v1.pos * barycentricCoords.y + v2.pos * barycentricCoords.z;
    vec3 worldPos = vec3(camera.model * vec4(pos, 1.0));
    //worldPos = gl_WorldRayOriginNV + gl_WorldRayDirectionNV * gl_HitTNV;

    hitInfo = vec4(worldPos, gl_PrimitiveID);
}
