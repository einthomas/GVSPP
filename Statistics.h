#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>

#include "StatisticsEntry.h"

enum SAMPLE_TYPE {
    SAMPLE_RANDOM,
    SAMPLE_ABS,
    SAMPLE_REVERSE,
    SAMPLE_EDGE_SUBDIV
};

class Statistics {
public:
    std::vector<StatisticsEntry> entries;

    Statistics(int samplesPerLine);
    void update();
    void print();

private:
    int samplesPerLine;
};

#endif // STATISTICS_H
