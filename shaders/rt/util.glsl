Vertex unpackVertexData(uint index) {
    vec4 d0 = vertices.v[3 * index + 0];
    vec4 d1 = vertices.v[3 * index + 1];
    vec4 d2 = vertices.v[3 * index + 2];

    Vertex vertex;
    vertex.pos = d0.xyz;
    vertex.worldPos = vec3(camera.model * vec4(vertex.pos, 1.0));
    vertex.normal = vec3(d0.w, d1.xy);
    vertex.color = vec3(d1.zw, d2.x);
    vertex.uv = d2.yz;

    return vertex;
}

vec3 getVertexPos(uint index) {
    return vertices.v[3 * index].xyz;
}

vec3 getVertexWorldPos(uint index) {
    return vec3(camera.model * vec4(
        vertices.v[3 * index + 0].xyz,
        1.0)
    );
}

bool intersectRayPlane(vec3 d, vec3 normal, vec3 rayOrigin, vec3 rayDir, out vec3 hitPoint) {
    float denom = dot(rayDir, normal);
    if (abs(denom) > 1e-6) {
        float t = dot(d - rayOrigin, normal) / denom;
        hitPoint = rayOrigin + t * rayDir;

        return true;
    }

    return false;
}

bool intersectRayPlane2(vec3 d, vec3 normal, vec3 rayOrigin, vec3 rayDir, out vec3 hitPoint, out float t) {
    float denom = dot(rayDir, normal);
    if (abs(denom) > 1e-6) {
        t = dot(d - rayOrigin, normal) / denom;
        hitPoint = rayOrigin + t * rayDir;

        return true;
    }

    return false;
}
