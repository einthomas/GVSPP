#ifndef VIEWCELL_H
#define VIEWCELL_H

#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

class ViewCell {
public:
    alignas(16) glm::vec3 pos;
    alignas(8) glm::vec2 size;
    alignas(16) glm::vec3 normal;

    ViewCell();
    ViewCell(glm::vec3 pos, glm::vec2 size, glm::vec3 normal);
};

#endif // VIEWCELL_H
