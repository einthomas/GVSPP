#include "sample.h"

Sample::Sample() {
    triangleID = 0;
    rayOrigin = glm::vec3(0.0f);
    hitPos = glm::vec3(0.0f);
}

Sample::Sample(int triangleID, glm::vec3 hitInfo, glm::vec3 hitPos)
    : triangleID(triangleID), rayOrigin(hitInfo), hitPos(hitPos)
{
}

bool Sample::operator==(const Sample &other) const {
    // Uniqueness is determined only by the triangle ID
    return triangleID == other.triangleID;
}
