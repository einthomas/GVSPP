#include "sample.h"

Sample::Sample() {
    triangleID = 0;
    hitInfo = glm::vec3(0.0f);
}

Sample::Sample(unsigned int triangleID, glm::vec3 hitInfo)
    : triangleID(triangleID), hitInfo(hitInfo)
{
}

bool Sample::operator==(const Sample &other) const {
    return triangleID == other.triangleID;
}
