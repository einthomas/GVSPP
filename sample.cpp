#include "sample.h"

Sample::Sample() {
    triangleID = 0;
    rayOrigin = glm::vec3(0.0f);
    hitPos = glm::vec3(0.0f);
    //pos = glm::vec3(0.0f);
}

Sample::Sample(int triangleID, glm::vec3 hitInfo, glm::vec3 hitPos)//, glm::vec3 pos)
    : triangleID(triangleID), rayOrigin(hitInfo), hitPos(hitPos)//, pos(pos)
{
}

bool Sample::operator==(const Sample &other) const {
    // Uniqueness is determined only by the triangle ID
    return triangleID == other.triangleID;
}
