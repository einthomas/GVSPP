#include "Statistics.h"

#include "StatisticsEntry.h"

#include <string>
#include <iostream>
#include <array>

Statistics::Statistics(int samplesPerLine)
    : samplesPerLine(samplesPerLine)
{
    entries.push_back(StatisticsEntry());
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
            "%'-15i%'-15i%'-15i%'-15i%'-15i%'-15i%'-15i%'-15i%'-15i%'-15i%'-15i%'-15i\n",
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
        printf("%'-15i%", sums[i]);
    }
    printf("\n\n");

    setlocale(LC_NUMERIC, "en_US");
}
