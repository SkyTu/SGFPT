// GPU context init: standard normal RNG and related helpers.
// Requires CURAND (standard normal generation).

#pragma once

#include "gpu_data_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Fill device buffer with standard normal samples in fixed-point.
 *  Uses CURAND normal generator; output is signed fixed-point: value / 2^scale = N(0,1) sample.
 *
 *  N     - number of samples (vector length)
 *  scale - fractional bits (same meaning as elsewhere: value = float * 2^scale)
 *  bout  - output bit width (signed); values are clamped to [-2^(bout-1), 2^(bout-1)-1]
 *  res   - device pointer; must have at least N * sizeof(i64) bytes. Output type is i64[]. */
void gpuRandomStandardNormal(int N, int scale, int bout, u8 *res);

/** Same as gpuRandomStandardNormal but with a given seed for reproducibility.
 *  Both parties must use the same seed to obtain the same Z in the Sample protocol. */
void gpuRandomStandardNormalWithSeed(int N, int scale, int bout, u8 *res, unsigned long long seed);

/** Fill device buffer with a d x d identity matrix in fixed-point (row-major).
 *  Diagonal elements = 1 (stored as 2^scale); off-diagonal = 0.
 *
 *  d     - matrix dimension (output is d x d)
 *  scale - fractional bits; diagonal value = 1 * 2^scale
 *  bout  - output bit width (signed); values are clamped to [-2^(bout-1), 2^(bout-1)-1]
 *  res   - device pointer; must have at least d * d * sizeof(i64) bytes. Row-major: res[i*d+j]. */
void gpuIdentityMatrix(int d, int scale, int bout, u8 *res);

#ifdef __cplusplus
}
#endif
