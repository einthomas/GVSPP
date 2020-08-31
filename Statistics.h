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
    EDGE_SUBDIVISION_INSERT = 5,
    HALTON_GENERATION = 6,
    VISIBILITY_SAMPLING = 7,
    GPU_HASH_SET_RESIZE = 8
};

class Statistics {
public:
    std::vector<StatisticsEntry> entries;

    Statistics(int samplesPerLine);
    void update();
    void print();
    void startOperation(OPERATION_TYPE operationType);
    void endOperation(OPERATION_TYPE operationType);
    int getTotalTracedRays();
    void reset();

private:
    int samplesPerLine;
    std::array<uint64_t, 9> elapsedTimes;
    std::array<std::chrono::steady_clock::time_point, 9> startTimes;
};

#endif // STATISTICS_H
