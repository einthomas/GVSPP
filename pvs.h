#ifndef PVS_H
#define PVS_H

#include <unordered_set>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

class PVS {
public:
    std::unordered_set<glm::uvec3> triangles;

    PVS();
};

#endif // PVS_H
