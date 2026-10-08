// GPU context init: standard normal RNG (CURAND).

#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>
#include <cassert>

#include "curand_utils.h"
#include "gpu_data_types.h"
#include "gpu_mem.h"
#include "gpu_ctx_init.h"

static curandGenerator_t s_normalGen = nullptr;

static curandGenerator_t getNormalGen()
{
    if (s_normalGen == nullptr)
    {
        CURAND_CHECK(curandCreateGenerator(&s_normalGen, CURAND_RNG_PSEUDO_XORWOW));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(s_normalGen, 54321ULL));
    }
    return s_normalGen;
}

__global__ void floatToFixedKernel(int N, int scale, int bout, const float *d_float, i64 *d_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        double v = (double)d_float[i];
        double scaled = v * (1LL << scale);
        i64 raw = (i64)__double2ll_rn(scaled);
        i64 lo = -(1LL << (bout - 1));
        i64 hi = (1LL << (bout - 1)) - 1;
        if (raw < lo) raw = lo;
        if (raw > hi) raw = hi;
        d_out[i] = raw;
    }
}

void gpuRandomStandardNormal(int N, int scale, int bout, u8 *res)
{
    gpuRandomStandardNormalWithSeed(N, scale, bout, res, 54321ULL);
}

void gpuRandomStandardNormalWithSeed(int N, int scale, int bout, u8 *res, unsigned long long seed)
{
    assert(N > 0 && scale >= 0 && bout >= 2 && bout <= 64 && res != nullptr);

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    float *d_float = (float *)gpuMalloc((size_t)N * sizeof(float));
    CURAND_CHECK(curandGenerateNormal(gen, d_float, (size_t)N, 0.0f, 1.0f));

    i64 *d_out = (i64 *)res;
    int block = 256;
    int grid = (N + block - 1) / block;
    floatToFixedKernel<<<grid, block>>>(N, scale, bout, d_float, d_out);
    cudaError_t e = cudaDeviceSynchronize();
    (void)e;
    assert(e == cudaSuccess);

    gpuFree(d_float);
    CURAND_CHECK(curandDestroyGenerator(gen));
}

__global__ void identityMatrixKernel(int d, int scale, int bout, i64 *res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d * d)
    {
        int i = idx / d;
        int j = idx % d;
        i64 raw = (i == j) ? (1LL << scale) : 0;
        i64 lo = -(1LL << (bout - 1));
        i64 hi = (1LL << (bout - 1)) - 1;
        if (raw < lo) raw = lo;
        if (raw > hi) raw = hi;
        res[idx] = raw;
    }
}

void gpuIdentityMatrix(int d, int scale, int bout, u8 *res)
{
    assert(d > 0 && scale >= 0 && bout >= 2 && bout <= 64 && res != nullptr);

    i64 *d_out = (i64 *)res;
    int n = d * d;
    int block = 256;
    int grid = (n + block - 1) / block;
    identityMatrixKernel<<<grid, block>>>(d, scale, bout, d_out);
    cudaError_t e = cudaDeviceSynchronize();
    (void)e;
    assert(e == cudaSuccess);
}
