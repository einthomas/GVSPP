#ifndef SAMPLE_H
#define SAMPLE_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
//#include <glm/gtx/hash.hpp>
//#include <glm/gtx/string_cast.hpp>

#include <iostream>

struct Sample {
    alignas(4) int triangleID;
    alignas(16) glm::vec3 rayOrigin;        // Origin of the ray that hit the triangle
    alignas(16) glm::vec3 hitPos;       // Position where the triangle was hit
    alignas(16) glm::vec3 pos;       // Position of the sample itself

    /*
    bool operator<(const Sample &s) const {
        return triangleID < s.triangleID;
    }
    */

    friend std::ostream &operator<<(std::ostream &stream, const Sample &sample) {
        return stream
            << "triangleID: " << sample.triangleID
            << " rayOrigin: (" << sample.rayOrigin.x << ", " << sample.rayOrigin.y << ", " << sample.rayOrigin.z << ") "
            << "hitPos: (" << sample.hitPos.x << ", " << sample.hitPos.y << ", " << sample.hitPos.z << ") "
            << "pos: (" << sample.pos.x << ", " << sample.pos.y << ", " << sample.pos.z << ") ";
    }
};

/*
class Sample {
public:
    alignas(4) int triangleID;
    alignas(16) glm::vec3 rayOrigin;        // Origin of the ray that hit the triangle
    alignas(16) glm::vec3 hitPos;       // Position where the triangle was hit
    alignas(16) glm::vec3 pos;       // Position of the sample itself
    Sample();
    Sample(int triangleID, glm::vec3 rayOrigin, glm::vec3 hitPos, glm::vec3 pos);

    bool operator==(const Sample &other) const;
    friend std::ostream &operator<<(std::ostream &ostream, const Sample &sample);
};


// See https://en.cppreference.com/w/cpp/utility/hash
namespace std {
    template<>
    struct hash<Sample> {
        size_t operator()(Sample const &sample) const {
            // Uniqueness is determined only by the triangle ID
            return (hash<int>()(sample.triangleID) << 1);
        }
    };
}
*/

#endif // SAMPLE_H
