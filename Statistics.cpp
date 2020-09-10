#include "Statistics.h"

#include "StatisticsEntry.h"

#include <string>
#include <iostream>
#include <array>

const float Statistics::div = 1000000.0f;

Statistics::Statistics(int samplesPerLine)
    : samplesPerLine(samplesPerLine)
{
    entries.push_back(StatisticsEntry());

    for (int i = 0; i < elapsedTimes.size(); i++) {
        elapsedTimes[i] = 0;
    }
}

void Statistics::update() {
    if (entries.back().totalRays() >= samplesPerLine) {
        if (entries.size() == 1) {
            entries[0].newTriangles = entries[0].pvsSize;
        } else {
            entries.back().newTriangles = entries.back().pvsSize - entries[entries.size() - 2].pvsSize;
        }

        entries.push_back(StatisticsEntry());
    }
}

void Statistics::print() {
    if (entries.size() == 1) {
        entries[0].newTriangles = entries[0].pvsSize;
    } else {
        entries.back().newTriangles = entries.back().pvsSize - entries[entries.size() - 2].pvsSize;
        if (entries.back().numShaderExecutions == 0) {
            entries.pop_back();
        }
    }

    setlocale(LC_NUMERIC, "");
    setlocale(LC_ALL, "");
    printf("\n");
    printf(
        "%14s|| %14s%14s%14s%14s%14s%14s|| %14s%14s%14s%14s%14s|| %14s%14s\n", "Shader Calls",
        "Rand Rays", "ABS Rays", "ABS RS Rays", "ES Rays", "ES RS Rays", "Total Rays", "Rand Tri",
        "ABS Tri", "ABS RS Tri", "ES Tri", "ES RS Tri", "PVS Size", "New Tri"
    );
    std::array<int, 13> sums;
    for (int i = 0; i < sums.size(); i++) {
        sums[i] = 0;
    }
    for (int i = 0; i < entries.size(); i++) {
        printf(
            "%14i|| %14i%14i%14i%14i%14i%14i|| %14i%14i%14i%14i%14i|| %14i%14i\n",
            entries[i].numShaderExecutions,
            entries[i].rnsRays,
            entries[i].absRays,
            entries[i].absRsRays,
            entries[i].edgeSubdivRays,
            entries[i].edgeSubdivRsRays,
            entries[i].totalRays(),
            entries[i].rnsTris,
            entries[i].absTris,
            entries[i].absRsTris,
            entries[i].edgeSubdivTris,
            entries[i].edgeSubdivRsTris,
            entries[i].pvsSize,
            entries[i].newTriangles
        );
        sums[0] += entries[i].numShaderExecutions;
        sums[1] += entries[i].rnsRays;
        sums[2] += entries[i].absRays;
        sums[3] += entries[i].absRsRays;
        sums[4] += entries[i].edgeSubdivRays;
        sums[5] += entries[i].edgeSubdivRsRays;
        sums[6] += entries[i].totalRays();
        sums[7] += entries[i].rnsTris;
        sums[8] += entries[i].absTris;
        sums[9] += entries[i].absRsTris;
        sums[10] += entries[i].edgeSubdivTris;
        sums[11] += entries[i].edgeSubdivRsTris;
        sums[12] = entries[i].pvsSize;
    }
    printf("%s", "===\n");
    float totalRaySum = sums[6];
    float pvsSize = sums[12];

    for (int i = 0; i < sums.size(); i++) {
        if (i == 1 || i == 7 || i == 12) {
            printf("|| %14i%", sums[i]);
        } else {
            printf("%14i%", sums[i]);
        }
    }
    printf("\n");
    printf(
        "%14s|| %13.2f%%%13.2f%%%13.2f%%%13.2f%%%13.2f%%%14s||\n",
        "", (sums[1] / totalRaySum) * 100, (sums[2] / totalRaySum) * 100,
        (sums[3] / totalRaySum) * 100, (sums[4] / totalRaySum) * 100, (sums[5] / totalRaySum) * 100,
        "100%"
    );
    printf("\n\n");

    printElapsedTimes(elapsedTimes);

    setlocale(LC_NUMERIC, "en_US");
}

void Statistics::startOperation(OPERATION_TYPE operationType) {
    startTimes[operationType] = std::chrono::steady_clock::now();
}

void Statistics::endOperation(OPERATION_TYPE operationType) {
    auto end = std::chrono::steady_clock::now();
    elapsedTimes[operationType] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - startTimes[operationType]).count();
}

int Statistics::getTotalTracedRays() {
    int sum = 0;
    for (auto e : entries) {
        sum += e.totalRays();
    }
    return sum;
}

float Statistics::getTotalRayTime() {
    return (elapsedTimes[RANDOM_SAMPLING] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING] + elapsedTimes[EDGE_SUBDIVISION]) / div;
}

float Statistics::getTotalInsertTime() {
    return (elapsedTimes[RANDOM_SAMPLING_INSERT] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT] + elapsedTimes[EDGE_SUBDIVISION_INSERT]) / div;
}

float Statistics::getTotalTime() {
    return elapsedTimes[VISIBILITY_SAMPLING] / div;
}

void Statistics::reset() {
    entries.clear();
    entries.push_back(StatisticsEntry());
    for (int i = 0; i < elapsedTimes.size(); i++) {
        elapsedTimes[i] = 0;
    }
}

void Statistics::printElapsedTimes(const std::array<uint64_t, 9> &elapsedTimes) {
    auto randSum = (elapsedTimes[RANDOM_SAMPLING] + elapsedTimes[RANDOM_SAMPLING_INSERT]) / div;
    auto absSum = (elapsedTimes[ADAPTIVE_BORDER_SAMPLING] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT]) / div;
    auto esSum = (elapsedTimes[EDGE_SUBDIVISION] + elapsedTimes[EDGE_SUBDIVISION_INSERT]) / div;

    printf("%14s%14s%14s\n", "Rand", "Rand Insert", "Total (ms)");
    printf(
        "%14.1f%14.1f%14.1f\n", elapsedTimes[RANDOM_SAMPLING]/ div,
        elapsedTimes[RANDOM_SAMPLING_INSERT]/ div, randSum
    );
    printf("%14s%14s%14s\n", "ABS", "ABS Insert", "Total (ms)");
    printf(
        "%14.1f%14.1f%14.1f\n", elapsedTimes[ADAPTIVE_BORDER_SAMPLING] / div,
        elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT] / div, absSum
    );
    printf("%14s%14s%14s\n", "ES", "ES Insert", "Total (ms)");
    printf(
        "%14.1f%14.1f%14.1f\n", elapsedTimes[EDGE_SUBDIVISION] / div,
        elapsedTimes[EDGE_SUBDIVISION_INSERT] / div, esSum
    );
    printf("%14s", "===\n");
    printf(
        "%14.1f%14.1f%14.1f\n",
        (elapsedTimes[RANDOM_SAMPLING] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING] + elapsedTimes[EDGE_SUBDIVISION]) / div,
        (elapsedTimes[RANDOM_SAMPLING_INSERT] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT] + elapsedTimes[EDGE_SUBDIVISION_INSERT]) / div,
        randSum + absSum + esSum
    );

    printf("\n\n");

    printf("Halton sequence generation time: %.2fms\n", elapsedTimes[HALTON_GENERATION] / div);
    printf("GPU hash set resize: %.2fms\n", elapsedTimes[GPU_HASH_SET_RESIZE] / div);
    printf("Total time: %.2fms\n", elapsedTimes[VISIBILITY_SAMPLING] / div);
    printf("\n\n");
}

void Statistics::printAverageStatistics(const std::vector<Statistics> &statistics) {
    printf("================== AVERAGE ==================\n");
    std::array<uint64_t, 9> avgElapsedTimes;
    for (int i = 0; i < avgElapsedTimes.size(); i++) {
        avgElapsedTimes[i] = 0;
    }

    for (auto s : statistics) {
        for (int i = 0; i < s.elapsedTimes.size(); i++) {
            avgElapsedTimes[i] += s.elapsedTimes[i];
        }
    }

    for (int i = 0; i < avgElapsedTimes.size(); i++) {
        avgElapsedTimes[i] /= float(statistics.size());
    }

    Statistics::printElapsedTimes(avgElapsedTimes);
}
