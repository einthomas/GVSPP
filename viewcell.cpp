#include "viewcell.h"

ViewCell::ViewCell()
{
}

ViewCell::ViewCell(glm::vec3 pos, glm::vec3 size, glm::vec3 normal)
    : pos(pos), size(size), normal(normal), tilePos(pos), tileSize(size)
{
}
