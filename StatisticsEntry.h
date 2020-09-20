#ifndef STATISTICSENTRY_H
#define STATISTICSENTRY_H

class StatisticsEntry {
public:
    long rasterHemicubes = 0;

    long rnsRays = 0;
    long absRays = 0;
    long absRsRays = 0;
    long edgeSubdivRays = 0;
    long edgeSubdivRsRays = 0;

    long rnsTris = 0;
    long absTris = 0;
    long absRsTris = 0;
    long edgeSubdivTris = 0;
    long edgeSubdivRsTris = 0;

    long newTriangles = 0;
    long pvsSize = 0;

    long numShaderExecutions = 0;

    long totalRays() {
        return rnsRays + absRays + absRsRays + edgeSubdivRays + edgeSubdivRsRays;
    }

    long totalFoundTriangles() {
        return rnsTris + absTris + absRsTris + edgeSubdivTris + edgeSubdivRsTris;
    }
};

#endif // STATISTICSENTRY_H
