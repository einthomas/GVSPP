#include "viewcell.h"

ViewCell::ViewCell() {
}

ViewCell::ViewCell(glm::vec3 pos, glm::vec3 size, glm::vec3 right, glm::vec3 up, glm::vec3 normal)
    : pos(pos), size(size), right(right), up(up), normal(normal)
{
}
