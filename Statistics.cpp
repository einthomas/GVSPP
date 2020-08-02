#include "Statistics.h"

#include "StatisticsEntry.h"

#include <string>
#include <iostream>
#include <array>

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

    auto randSum = elapsedTimes[RANDOM_SAMPLING] + elapsedTimes[RANDOM_SAMPLING_INSERT];
    auto absSum = elapsedTimes[ADAPTIVE_BORDER_SAMPLING] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT];
    auto esSum = elapsedTimes[EDGE_SUBDIVISION] + elapsedTimes[EDGE_SUBDIVISION_INSERT];

    printf("%14s%14s%14s\n", "Rand", "Rand Insert", "Total (ms)");
    printf(
        "%14i%14i%14i\n", elapsedTimes[RANDOM_SAMPLING], elapsedTimes[RANDOM_SAMPLING_INSERT],
        elapsedTimes[RANDOM_SAMPLING] + elapsedTimes[RANDOM_SAMPLING_INSERT], randSum
    );
    printf("%14s%14s%14s\n", "ABS", "ABS Insert", "Total (ms)");
    printf(
        "%14i%14i%14i\n", elapsedTimes[ADAPTIVE_BORDER_SAMPLING],
        elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT], absSum

    );
    printf("%14s%14s%14s\n", "ES", "ES Insert", "Total (ms)");
    printf(
        "%14i%14i%14i\n", elapsedTimes[EDGE_SUBDIVISION], elapsedTimes[EDGE_SUBDIVISION_INSERT],
        esSum
    );
    printf("%14s", "===\n");
    printf(
        "%14i%14i%14i\n",
        elapsedTimes[RANDOM_SAMPLING] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING] + elapsedTimes[EDGE_SUBDIVISION],
        elapsedTimes[RANDOM_SAMPLING_INSERT] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT] + elapsedTimes[EDGE_SUBDIVISION_INSERT],
        randSum + absSum + esSum
    );

    printf("\n\n");

    setlocale(LC_NUMERIC, "en_US");
}

void Statistics::startOperation(OPERATION_TYPE operationType) {
    startTimes[operationType] = std::chrono::steady_clock::now();
}

void Statistics::endOperation(OPERATION_TYPE operationType) {
    auto end = std::chrono::steady_clock::now();
    elapsedTimes[operationType] += std::chrono::duration_cast<std::chrono::milliseconds>(end - startTimes[operationType]).count();
}

int Statistics::getTotalTracedRays() {
    int sum = 0;
    for (auto e : entries) {
        sum += e.totalRays();
    }
    return sum;
}
