#ifndef SAMPLE_H
#define SAMPLE_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include <iostream>

struct Sample {
    alignas(4) int triangleID;
    alignas(16) glm::vec3 rayOrigin;        // Origin of the ray that hit the triangle
    alignas(16) glm::vec3 hitPos;       // Position where the triangle was hit
    alignas(16) glm::vec3 pos;       // Position of the sample itself

    friend std::ostream &operator<<(std::ostream &stream, const Sample &sample) {
        return stream
            << "triangleID: " << sample.triangleID
            << " rayOrigin: (" << sample.rayOrigin.x << ", " << sample.rayOrigin.y << ", " << sample.rayOrigin.z << ") "
            << "hitPos: (" << sample.hitPos.x << ", " << sample.hitPos.y << ", " << sample.hitPos.z << ") "
            << "pos: (" << sample.pos.x << ", " << sample.pos.y << ", " << sample.pos.z << ") ";
    }
};

#endif // SAMPLE_H
