#ifndef STATISTICSENTRY_H
#define STATISTICSENTRY_H

class StatisticsEntry {
public:
    int rnsRays = 0;
    int absRays = 0;
    int absRsRays = 0;
    int edgeSubdivRays = 0;
    int edgeSubdivRsRays = 0;

    int rnsTris = 0;
    int absTris = 0;
    int absRsTris = 0;
    int edgeSubdivTris = 0;
    int edgeSubdivRsTris = 0;

    int newTriangles = 0;
    int pvsSize = 0;

    int numShaderExecutions = 0;

    int totalRays() {
        return rnsRays + absRays + absRsRays + edgeSubdivRays + edgeSubdivRsRays;
    }

    int totalFoundTriangles() {
        return rnsTris + absTris + absRsTris + edgeSubdivTris + edgeSubdivRsTris;
    }
};

#endif // STATISTICSENTRY_H
