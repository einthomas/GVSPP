#ifndef STATISTICSENTRY_H
#define STATISTICSENTRY_H

class StatisticsEntry {
public:
    long long rasterHemicubes = 0;

    long long rnsRays = 0;
    long long absRays = 0;
    long long absRsRays = 0;
    long long edgeSubdivRays = 0;
    long long edgeSubdivRsRays = 0;

    long long rnsTris = 0;
    long long absTris = 0;
    long long absRsTris = 0;
    long long edgeSubdivTris = 0;
    long long edgeSubdivRsTris = 0;

    long long newTriangles = 0;
    long long pvsSize = 0;

    long long numShaderExecutions = 0;

    long long totalRays() {
        return rnsRays + absRays + absRsRays + edgeSubdivRays + edgeSubdivRsRays;
    }

    long long totalFoundTriangles() {
        return rnsTris + absTris + absRsTris + edgeSubdivTris + edgeSubdivRsTris;
    }
};

#endif // STATISTICSENTRY_H
