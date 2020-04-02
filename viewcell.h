#ifndef VIEWCELL_H
#define VIEWCELL_H

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>

class ViewCell {
public:
    alignas(16) glm::vec3 pos;
    alignas(8) glm::vec2 size;
    alignas(16) glm::vec3 normal;

    ViewCell(glm::vec3 pos, glm::vec2 size, glm::vec3 normal);
};

#endif // VIEWCELL_H
