// Correctness test for gpuRandomStandardNormal: check sample mean ~0, variance ~1.

#include "utils/gpu_data_types.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_ctx_init.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
    int gpuId = 0;
    if (argc >= 2)
        gpuId = atoi(argv[1]);

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || gpuId < 0 || gpuId >= devCount)
    {
        fprintf(stderr, "Invalid gpu_id=%d (available: 0..%d)\n", gpuId, devCount > 0 ? devCount - 1 : 0);
        return 1;
    }
    if (cudaSetDevice(gpuId) != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice(%d) failed\n", gpuId);
        return 1;
    }

    initGPUMemPool();

    const int N = 10000;
    const int scale = 12;
    const int bout = 48;

    u8 *d_res = (u8 *)gpuMalloc((size_t)N * sizeof(i64));
    gpuRandomStandardNormal(N, scale, bout, d_res);

    i64 *h_res = (i64 *)moveToCPU(d_res, (size_t)N * sizeof(i64), (Stats *)NULL);
    gpuFree(d_res);

    double sum = 0.0, sumSq = 0.0;
    for (int i = 0; i < N; i++)
    {
        double x = (double)h_res[i] / (1LL << scale);
        sum += x;
        sumSq += x * x;
    }
    double mean = sum / N;
    double var = (sumSq / N) - (mean * mean);

    printf("N = %d, scale = %d, bout = %d\n", N, scale, bout);
    printf("First 10 (fixed-point): ");
    for (int i = 0; i < 10; i++)
        printf("%ld ", (long)h_res[i]);
    printf("\n");
    printf("First 10 (float): ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", (double)h_res[i] / (1LL << scale));
    printf("\n");
    printf("Sample mean  = %.6f (expect ~0)\n", mean);
    printf("Sample var   = %.6f (expect ~1)\n", var);

    int ok = 1;
    if (fabs(mean) > 0.1)
    {
        printf("FAIL: |mean| = %.4f > 0.1\n", fabs(mean));
        ok = 0;
    }
    if (var < 0.7 || var > 1.3)
    {
        printf("FAIL: variance = %.4f not in [0.7, 1.3]\n", var);
        ok = 0;
    }
    if (ok)
        printf("PASS: standard normal stats OK.\n");

    return ok ? 0 : 1;
}
