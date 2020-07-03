#include "viewcell.h"

#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

ViewCell::ViewCell()
{
}

ViewCell::ViewCell(glm::vec3 pos, glm::vec3 size, glm::vec3 normal)
    : pos(pos), size(size), normal(normal), tilePos(pos), tileSize(size)
{
}
