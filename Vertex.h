#ifndef VERTEX_H
#define VERTEX_H

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <array>

class Vertex {
public:
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec3 texCoord;

    static VkVertexInputBindingDescription getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions();
    bool operator==(const Vertex &other) const;
};

// See https://en.cppreference.com/w/cpp/utility/hash
namespace std {
    template<>
    struct hash<Vertex> {
        size_t operator()(Vertex const &vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                (hash<glm::vec3>()(vertex.texCoord) << 1)) >> 1) ^
                (hash<glm::vec2>()(vertex.normal) << 1);
        }
    };
}

#endif // VERTEX_H
