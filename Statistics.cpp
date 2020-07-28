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
        entries.push_back(StatisticsEntry());
        entries.back().pvsSize += entries[entries.size() - 1].pvsSize;
    }
}

void Statistics::print() {
    setlocale(LC_NUMERIC, "");
    printf("\n");
    printf(
        "%-15s%-15s%-15s%-15s%-15s%-15s%-15s%-15s%-15s%-15s%-15s%-15s\n", "Shader Calls",
        "Rand Rays", "ABS Rays", "RS Rays", "ES Rays", "Total Rays", "Rand Triangles",
        "ABS Triangles", "RS Triangles", "ES Triangles", "PVS Size", "New Triangles"
    );
    std::array<int, 11> sums;
    for (int i = 0; i < sums.size(); i++) {
        sums[i] = 0;
    }
    for (int i = 0; i < entries.size(); i++) {
        printf(
            "%-15i%-15i%-15i%-15i%-15i%-15i%-15i%-15i%-15i%-15i%-15i%-15i\n",
            entries[i].numShaderExecutions,
            entries[i].rnsRays,
            entries[i].absRays,
            entries[i].rsRays,
            entries[i].edgeSubdivRays,
            entries[i].totalRays(),
            entries[i].rnsTris,
            entries[i].absTris,
            entries[i].rsTris,
            entries[i].edgeSubdivTris,
            entries[i].pvsSize,
            entries[i].newTriangles
        );
        sums[0] += entries[i].numShaderExecutions;
        sums[1] += entries[i].rnsRays;
        sums[2] += entries[i].absRays;
        sums[3] += entries[i].rsRays;
        sums[4] += entries[i].edgeSubdivRays;
        sums[5] += entries[i].totalRays();
        sums[6] += entries[i].rnsTris;
        sums[7] += entries[i].absTris;
        sums[8] += entries[i].rsTris;
        sums[9] += entries[i].edgeSubdivTris;
        sums[10] = entries[i].pvsSize;
    }
    printf("===\n");
    for (int i = 0; i < sums.size(); i++) {
        printf("%-15i%", sums[i]);
    }
    printf("\n\n");

    auto randSum = elapsedTimes[RANDOM_SAMPLING] + elapsedTimes[RANDOM_SAMPLING_INSERT];
    auto absSum = elapsedTimes[ADAPTIVE_BORDER_SAMPLING] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT];
    auto esSum = elapsedTimes[EDGE_SUBDIVISION] + elapsedTimes[EDGE_SUBDIVISION_INSERT];

    printf("%-15s%-15s%-15s\n", "Rand", "Rand Insert", "Total (ms)");
    printf(
        "%-15i%-15i%-15i\n", elapsedTimes[RANDOM_SAMPLING], elapsedTimes[RANDOM_SAMPLING_INSERT],
        elapsedTimes[RANDOM_SAMPLING] + elapsedTimes[RANDOM_SAMPLING_INSERT], randSum
    );
    printf("%-15s%-15s%-15s\n", "ABS", "ABS Insert", "Total (ms)");
    printf(
        "%-15i%-15i%-15i\n", elapsedTimes[ADAPTIVE_BORDER_SAMPLING],
        elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT], absSum

    );
    printf("%-15s%-15s%-15s\n", "ES", "ES Insert", "Total (ms)");
    printf(
        "%-15i%-15i%-15i\n", elapsedTimes[EDGE_SUBDIVISION], elapsedTimes[EDGE_SUBDIVISION_INSERT],
        esSum
    );
    printf("===\n");
    printf(
        "%-15i%-15i%-15i\n",
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
