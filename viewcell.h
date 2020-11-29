#ifndef VIEWCELL_H
#define VIEWCELL_H

#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/mat4x4.hpp>
#include <glm/geometric.hpp>
#include <array>

class ViewCell {
public:
    alignas(16) glm::vec3 pos;
    alignas(16) glm::vec3 size;
    alignas(16) glm::vec3 right;
    alignas(16) glm::vec3 up;
    alignas(16) glm::vec3 normal;

    ViewCell();

    ViewCell(glm::vec3 pos, glm::vec3 size, glm::vec3 right, glm::vec3 up, glm::vec3 normal);

    bool operator() (const ViewCell& lhs, const ViewCell& rhs) const {
        return glm::length(lhs.pos) < glm::length(lhs.pos);
    }

    bool operator <(const ViewCell& rhs) const {
        return glm::length(rhs.pos) < glm::length(pos);
    }
};

#endif // VIEWCELL_H
