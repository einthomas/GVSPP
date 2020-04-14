#ifndef SAMPLE_H
#define SAMPLE_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

class Sample {
public:
    unsigned int triangleID;
    alignas(16) glm::vec3 hitInfo;  // x == triangle ID, yzw == ray origin
    Sample();
    Sample(unsigned int triangleID, glm::vec3 hitInfo);

    bool operator==(const Sample &other) const;
};

// See https://en.cppreference.com/w/cpp/utility/hash
namespace std {
    template<>
    struct hash<Sample> {
        size_t operator()(Sample const &sample) const {
            return (hash<unsigned int>()(sample.triangleID) << 1);
        }
    };
}

#endif // SAMPLE_H
