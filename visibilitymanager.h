#ifndef VISIBILITYMANAGER_H
#define VISIBILITYMANAGER_H

#include <vector>

#include "viewcell.h"

class VisibilityManager {
public:
    std::vector<glm::vec2> haltonPoints;
    std::vector<ViewCell> viewCells;

    VisibilityManager();
    void addViewCell(glm::vec3 pos, glm::vec2 size, glm::vec3 normal);
    void generateHaltonPoints(int n, int p2 = 7);
};

#endif // VISIBILITYMANAGER_H
