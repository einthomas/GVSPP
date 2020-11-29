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
    HALTON_GENERATION = 6,
    VISIBILITY_SAMPLING = 7,
    GPU_HASH_SET_RESIZE = 8,
    RASTER_VISIBILITY_RENDER = 9,
    RASTER_VISIBILITY_COMPUTE = 10
};

class Statistics {
public:
    std::vector<StatisticsEntry> entries;
    std::array<uint64_t, 11> elapsedTimes;

    Statistics(int samplesPerLine);
    void update();
    void addLine();
    void print();
    void startOperation(OPERATION_TYPE operationType);
    void endOperation(OPERATION_TYPE operationType);
    long long getTotalTracedRays();
    long getPVSSize();
    void reset();
    static void printElapsedTimes(const std::array<uint64_t, 11> &elapsedTimes);
    static void printAverageStatistics(const std::vector<Statistics> &statistics);

private:
    static const float div;
    int samplesPerLine;
    std::array<std::chrono::steady_clock::time_point, 11> startTimes;
};

#endif // STATISTICS_H
