/* Numeric arrays only cross the DLL boundary; keep CRT objects local. */
#include <math.h>
#include <stdio.h>
#include <string.h>

extern char *openblas_get_config(void);
extern void dgemm_(const char *, const char *, const int *, const int *, const int *,
                   const double *, const double *, const int *, const double *,
                   const int *, const double *, double *, const int *);
extern void dpotrf_(const char *, const int *, double *, const int *, int *);

int main(void) {
    const char no = 'N', lower = 'L';
    const int n = 2;
    const double one = 1.0, zero = 0.0;
    const double a[4] = {1, 3, 2, 4}, b[4] = {5, 7, 6, 8};
    const double expected[4] = {19, 43, 22, 50};
    double c[4] = {0, 0, 0, 0}, spd[4] = {4, 2, 2, 5};
    int info = -1, i;
    const char *config = openblas_get_config();
    if (!config || !strstr(config, "OpenBLAS ") ||
        !strstr(config, "DYNAMIC_ARCH") || strstr(config, "USE64BITINT") ||
        sizeof(int) != 4) {
        fprintf(stderr, "Unexpected OpenBLAS LP64 configuration: %s\n", config ? config : "missing");
        return 1;
    }
    printf("%s\n", config);
    dgemm_(&no, &no, &n, &n, &n, &one, a, &n, b, &n, &zero, c, &n);
    for (i = 0; i < 4; ++i) {
        if (!isfinite(c[i]) || fabs(c[i] - expected[i]) > 1e-12) {
            fprintf(stderr, "dgemm mismatch at %d: %.17g\n", i, c[i]);
            return 1;
        }
    }
    puts("dgemm ok");
    dpotrf_(&lower, &n, spd, &n, &info);
    if (info != 0 || !isfinite(spd[0]) || !isfinite(spd[1]) || !isfinite(spd[3]) ||
        fabs(spd[0] - 2) > 1e-12 || fabs(spd[1] - 1) > 1e-12 || fabs(spd[3] - 2) > 1e-12) {
        fprintf(stderr, "dpotrf mismatch: info=%d L=(%.17g, %.17g, %.17g)\n", info, spd[0], spd[1], spd[3]);
        return 1;
    }
    puts("dpotrf ok");
    return 0;
}
