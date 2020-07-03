#ifndef VIEWCELL_H
#define VIEWCELL_H

#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

class ViewCell {
public:
    alignas(16) glm::vec3 pos;
    alignas(16) glm::vec3 size;
    alignas(16) glm::vec3 normal;
    alignas(16) glm::vec3 tilePos;
    alignas(16) glm::vec3 tileSize;

    ViewCell();
    ViewCell(glm::vec3 pos, glm::vec3 size, glm::vec3 normal);
};

#endif // VIEWCELL_H
