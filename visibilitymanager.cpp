#include "visibilitymanager.h"
#include "viewcell.h"

VisibilityManager::VisibilityManager() {
}

void VisibilityManager::addViewCell(glm::vec3 pos, glm::vec2 size, glm::vec3 normal) {
    viewCells.push_back(ViewCell(pos, size, normal));
}

/*
 * From "Sampling with Hammersley and Halton Points" (Wong et al. 1997)
 */
void VisibilityManager::generateHaltonPoints(int n, int p2) {
    haltonPoints.resize(n);

    float p, u, v, ip;
    int k, kk, pos, a;
    for (k = 0, pos = 0; k < n; k++) {
        u = 0;
        for (p = 0.5, kk = k; kk; p *= 0.5, kk >>= 1) {
            if (kk & 1) {
                u += p;
            }
        }

        v = 0;
        ip = 1.0 / p2;
        for (p = ip, kk = k; kk; p *= ip, kk /= p2) {
            if ((a = kk % p2)) {
                v += a * p;
            }
        }

        haltonPoints[pos].x = u;
        haltonPoints[pos].y = v;
        pos++;
    }
}
