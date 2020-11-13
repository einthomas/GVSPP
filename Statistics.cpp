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
        addLine();
    }
}

void Statistics::addLine() {
    if (entries.size() == 1) {
        entries[0].newTriangles = entries[0].pvsSize;
    } else {
        entries.back().newTriangles = entries.back().pvsSize - entries[entries.size() - 2].pvsSize;
    }

    entries.push_back(StatisticsEntry());
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
        "%16s|| %16s|| %16s%16s%16s%16s%16s%16s|| %16s%16s%16s%16s%16s|| %16s%16s\n", "Ray Shader Calls", "Raster Cubes",
        "Rand Rays", "ABS Rays", "ABS RS Rays", "ES Rays", "ES RS Rays", "Total Rays", "Rand Tri",
        "ABS Tri", "ABS RS Tri", "ES Tri", "ES RS Tri", "PVS Size", "New Tri"
    );
    std::array<long long, 14> sums;
    for (int i = 0; i < sums.size(); i++) {
        sums[i] = 0;
    }
    for (int i = 0; i < entries.size(); i++) {
        printf(
            "%16lu|| %16lu|| %16lu%16lu%16lu%16lu%16lu%16lu|| %16lu%16lu%16lu%16lu%16lu|| %16lu%16lu\n",
            entries[i].numShaderExecutions,
            entries[i].rasterHemicubes,
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
        sums[1] += entries[i].rasterHemicubes;
        sums[2] += entries[i].rnsRays;
        sums[3] += entries[i].absRays;
        sums[4] += entries[i].absRsRays;
        sums[5] += entries[i].edgeSubdivRays;
        sums[6] += entries[i].edgeSubdivRsRays;
        sums[7] += entries[i].totalRays();
        sums[8] += entries[i].rnsTris;
        sums[9] += entries[i].absTris;
        sums[10] += entries[i].absRsTris;
        sums[11] += entries[i].edgeSubdivTris;
        sums[12] += entries[i].edgeSubdivRsTris;
        sums[13] = entries[i].pvsSize;
    }
    printf("%s", "============================================================================================================================================================================================================================================================\n");
    float totalRaySum = sums[7];

    for (int i = 0; i < sums.size(); i++) {
        if (i == 1 || i == 2 || i == 8 || i == 13) {
            printf("|| %16lu%", sums[i]);
        } else {
            printf("%16lu%", sums[i]);
        }
    }
    printf("\n");
    printf(
        "%16s|| %16s|| %15.2f%%%15.2f%%%15.2f%%%15.2f%%%15.2f%%%16s||\n", "",
        "", (sums[2] / totalRaySum) * 100, (sums[3] / totalRaySum) * 100,
        (sums[4] / totalRaySum) * 100, (sums[5] / totalRaySum) * 100, (sums[6] / totalRaySum) * 100,
        "100%"
    );
    printf("\n\n");

    printElapsedTimes(elapsedTimes);
    printf("\n\n");

    setlocale(LC_NUMERIC, "en_US");
}

void Statistics::startOperation(OPERATION_TYPE operationType) {
    startTimes[operationType] = std::chrono::steady_clock::now();
}

void Statistics::endOperation(OPERATION_TYPE operationType) {
    auto end = std::chrono::steady_clock::now();
    elapsedTimes[operationType] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - startTimes[operationType]).count();
}

long long Statistics::getTotalTracedRays() {
    long long sum = 0;
    for (auto e : entries) {
        sum += e.totalRays();
    }
    return sum;
}

long Statistics::getPVSSize() {
    long sum = 0;
    for (auto e : entries) {
        sum += e.pvsSize;
    }
    return sum;
}

void Statistics::reset() {
    entries.clear();
    entries.push_back(StatisticsEntry());
    for (int i = 0; i < elapsedTimes.size(); i++) {
        elapsedTimes[i] = 0;
    }
}

void Statistics::printElapsedTimes(const std::array<uint64_t, 11> &elapsedTimes) {
    auto rasterSum = (elapsedTimes[RASTER_VISIBILITY_RENDER] + elapsedTimes[RASTER_VISIBILITY_COMPUTE]) / div;
    auto randSum = (elapsedTimes[RANDOM_SAMPLING] + elapsedTimes[RANDOM_SAMPLING_INSERT]) / div;
    auto absSum = (elapsedTimes[ADAPTIVE_BORDER_SAMPLING] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT]) / div;
    auto esSum = (elapsedTimes[EDGE_SUBDIVISION] + elapsedTimes[EDGE_SUBDIVISION_INSERT]) / div;

    printf("%24s%24s%24s\n", "Raster", "Raster", "Total (ms)");
    printf("%24s%24s%24s\n", "(Hemicube Rendering)", "(Compute)", "");
    printf(
        "%24.1f%24.1f%24.1f\n\n", elapsedTimes[RASTER_VISIBILITY_RENDER]/ div,
        elapsedTimes[RASTER_VISIBILITY_COMPUTE]/ div, rasterSum
    );
    printf("%24s%24s%24s\n", "Random", "Random", "Total (ms)");
    printf("%24s%24s%24s\n", "(Shader)", "(Sample Queue Insert)", "");
    printf(
        "%24.1f%24.1f%24.1f\n\n", elapsedTimes[RANDOM_SAMPLING]/ div,
        elapsedTimes[RANDOM_SAMPLING_INSERT]/ div, randSum
    );
    printf("%24s%24s%24s\n", "ABS", "ABS", "Total (ms)");
    printf("%24s%24s%24s\n", "(Shader)", "(Sample Queue Insert)", "");
    printf(
        "%24.1f%24.1f%24.1f\n\n", elapsedTimes[ADAPTIVE_BORDER_SAMPLING] / div,
        elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT] / div, absSum
    );
    printf("%24s%24s%24s\n", "Recursive Edge Subdiv.", "Recursive Edge Subdiv.", "Total (ms)");
    printf("%24s%24s%24s\n", "(Shader)", "(Sample Queue Insert)", "");
    printf(
        "%24.1f%24.1f%24.1f\n", elapsedTimes[EDGE_SUBDIVISION] / div,
        elapsedTimes[EDGE_SUBDIVISION_INSERT] / div, esSum
    );
    printf("                   =====                   =====                   =====\n");
    printf(
        "%24.1f%24.1f%24.1f\n",
        (elapsedTimes[RANDOM_SAMPLING] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING] + elapsedTimes[EDGE_SUBDIVISION]) / div,
        (elapsedTimes[RANDOM_SAMPLING_INSERT] + elapsedTimes[ADAPTIVE_BORDER_SAMPLING_INSERT] + elapsedTimes[EDGE_SUBDIVISION_INSERT]) / div,
        rasterSum + randSum + absSum + esSum
    );

    printf("\n");

    printf("Halton sequence generation time: %.2fms\n", elapsedTimes[HALTON_GENERATION] / div);
    printf("GPU hash set resize: %.2fms\n", elapsedTimes[GPU_HASH_SET_RESIZE] / div);
    printf("Total time: %.2fms\n", elapsedTimes[VISIBILITY_SAMPLING] / div);
}

void Statistics::printAverageStatistics(const std::vector<Statistics> &statistics) {
    printf("==================================== AVERAGE> ====================================\n");
    std::array<uint64_t, 11> avgElapsedTimes;
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

    int pvsSize = 0;
    for (auto s : statistics) {
        pvsSize += s.entries.back().pvsSize;
    }
    pvsSize /= statistics.size();
    printf("PVS size: %i\n", pvsSize);
    printf("==================================== <AVERAGE ====================================\n\n");
}
