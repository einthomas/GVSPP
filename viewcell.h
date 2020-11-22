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

    /*
    alignas(64) glm::mat4 model;

    ViewCell();
    ViewCell(glm::mat4 model);

    bool operator() (const ViewCell& lhs, const ViewCell& rhs) const
    {
        return glm::length(lhs.model[3]) < glm::length(lhs.model[3]);
    }

    bool operator <(const ViewCell& rhs) const
    {
        return glm::length(rhs.model[3]) < glm::length(model[3]);
    }
    */
};

#endif // VIEWCELL_H
