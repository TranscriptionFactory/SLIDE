#include "dsdp5.h"
#include <stdlib.h>

/* All arrays outlive the solver: DSDP retains input pointers until destruction. */
int slide_dsdp_diagonal(int n, double *cov, int maxit, double gap, double feas,
                        double *solution, double *stats, int *status) {
    DSDP solver = NULL;
    SDPCone sdp;
    LPCone lp;
    DSDPTerminationReason reason;
    DSDPSolutionType type;
    int rc = 0, iterations = 0;
    int *diagonal = calloc(n, sizeof(int));
    int *pointers = calloc(n + 2, sizeof(int));
    int *rows = calloc(3 * n, sizeof(int));
    double *values = calloc(3 * n, sizeof(double));
    double one = 1.0;
    if (!diagonal || !pointers || !rows || !values) { rc = -100; goto cleanup; }
#define CALL(expression) do { rc = (expression); if (rc) goto cleanup; } while (0)
    CALL(DSDPCreate(n, &solver));
    CALL(DSDPCreateSDPCone(solver, 1, &sdp));
    CALL(SDPConeSetBlockSize(sdp, 0, n));
    CALL(SDPConeSetStorageFormat(sdp, 0, 'P'));
    CALL(SDPConeSetSparsity(sdp, 0, n + 1));
    CALL(SDPConeSetADenseVecMat(sdp, 0, 0, n, 1.0, cov, n * (n + 1) / 2));
    pointers[1] = n;
    for (int i = 0; i < n; ++i) {
        diagonal[i] = i * (i + 1) / 2 + i;
        CALL(SDPConeSetASparseVecMat(sdp, 0, i + 1, n, 1.0, 0, diagonal + i, &one, 1));
        CALL(DSDPSetDualObjective(solver, i + 1, 1.0));
        rows[i] = n + i; values[i] = 1.0;
        rows[n + 2*i] = i; values[n + 2*i] = -1.0;
        rows[n + 2*i + 1] = n + i; values[n + 2*i + 1] = 1.0;
        pointers[i + 2] = n + 2*(i + 1);
    }
    CALL(DSDPCreateLPCone(solver, &lp));
    CALL(LPConeSetData(lp, 2*n, pointers, rows, values));
    CALL(DSDPSetMaxIts(solver, maxit));
    CALL(DSDPSetGapTolerance(solver, gap));
    CALL(DSDPSetPTolerance(solver, feas));
    CALL(DSDPSetRTolerance(solver, feas));
    CALL(DSDPSetup(solver));
    CALL(DSDPSolve(solver));
    CALL(DSDPComputeX(solver));
    CALL(DSDPStopReason(solver, &reason));
    CALL(DSDPGetSolutionType(solver, &type));
    CALL(DSDPGetIts(solver, &iterations));
    status[0] = reason; status[1] = type; status[2] = iterations;
    CALL(DSDPGetY(solver, solution, n));
    CALL(DSDPGetDObjective(solver, stats));
    CALL(DSDPGetPObjective(solver, stats + 1));
    CALL(DSDPGetFinalErrors(solver, stats + 2));
cleanup:
    if (solver) { int destroy_rc = DSDPDestroy(solver); if (!rc) rc = destroy_rc; }
    free(diagonal); free(pointers); free(rows); free(values);
    return rc;
}
