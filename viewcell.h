#ifndef VIEWCELL_H
#define VIEWCELL_H

#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/mat4x4.hpp>
#include <glm/geometric.hpp>
#include <array>

class ViewCell {
public:
    glm::mat4 model;

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
};

#endif // VIEWCELL_H
