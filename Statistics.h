#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>
#include <array>
#include <chrono>

#include "StatisticsEntry.h"

enum OPERATION_TYPE {
    RANDOM_SAMPLING = 0,
    RANDOM_SAMPLING_INSERT = 1,
    ADAPTIVE_BORDER_SAMPLING = 2,
    ADAPTIVE_BORDER_SAMPLING_INSERT = 3,
    EDGE_SUBDIVISION = 4,
    EDGE_SUBDIVISION_INSERT = 5
};

class Statistics {
public:
    std::vector<StatisticsEntry> entries;

    Statistics(int samplesPerLine);
    void update();
    void print();
    void startOperation(OPERATION_TYPE operationType);
    void endOperation(OPERATION_TYPE operationType);

private:
    int samplesPerLine;
    std::array<uint64_t, 6> elapsedTimes;
    std::array<std::chrono::steady_clock::time_point, 6> startTimes;
};

#endif // STATISTICS_H
