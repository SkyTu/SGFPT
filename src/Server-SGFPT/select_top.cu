// Algorithm 2 Pi^SelectTop: aggregate [a], rank by comparisons, select top-mu Y and Z.
// [A] lambda x p (accuracy, m=1+scale), [Y],[Z] lambda x d. Output [Y_mu],[Z_mu] mu x d.

#include "select_top.h"
#include "utils/gpu_data_types.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/misc_utils.h"
#include "fss/gpu_relu.h"
#include "fss/gpu_truncate.h"
#include "fss/gpu_mul.h"
#include "fss/gpu_select.h"
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

namespace spt {

static const int bin_cmp = 64;

// Map linear index k in [0, n_comp) to pair (i, j) with i < j.
__device__ __host__ inline void pairKToIJ(int k, int lambda, int* i, int* j) {
    int n_before = 0;
    for (int ii = 0; ii < lambda; ii++) {
        int block_sz = lambda - 1 - ii;
        if (k < n_before + block_sz) {
            *i = ii;
            *j = ii + 1 + (k - n_before);
            return;
        }
        n_before += block_sz;
    }
    *i = lambda - 2;
    *j = lambda - 1;
}


template <typename T>
__global__ void sumLambdaRowsKernel(int lambda, int bw, int d, const T* in, T* out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < d) {
        T sum = 0;
        for (int row = 0; row < lambda; ++row) {
            sum += in[row * d + col];
        }
        gpuMod(sum, bw);
        out[col] = sum;
    }
}

// (i, j) with i < j -> linear index k.
__device__ __host__ inline int pairIJToK(int i, int j, int lambda) {
    return i * (2 * lambda - i - 1) / 2 + (j - i - 1);
}

// Used in both keygen (input_mask -> d_diff_mask) and run (d_a -> d_diff). Same (i,j) ordering so mask and computation match.
template <typename T>
__global__ void diffMaskFromInputKernel(int lambda, int bin, const T* d_A_mask, T* d_diff_mask) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n_comp = lambda * (lambda - 1) / 2;
    if (k >= n_comp) return;
    int i, j;
    pairKToIJ(k, lambda, &i, &j);
    T d = (T)((u64)d_A_mask[i] - (u64)d_A_mask[j]);
    gpuMod(d, bin);
    d_diff_mask[k] = d;
}

__global__ void packReluBitsToU8(const u32* d_relu_res, int n_comp, int n_s_, u8* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_comp) {
        u32 v = d_relu_res[i];
        #pragma unroll
        for (int b = 0; b < n_s_; ++b) {
            int bit_idx = i * n_s_ + b;
            int byte_idx = bit_idx >> 3;
            int off = bit_idx & 7;
            u8 bit = (v >> b) & 1;
            if (bit) out[byte_idx] |= (1 << off);
        }
    }
}

// reduceReluMaskToLambdaKernel: keygen mask reduction.
//   rank_mask[i] = sum_{j>i} reluMask[pairIJToK(i,j)]  (mask of drelu(a_i-a_j))
//                + sum_{j<i} -reluMask[pairIJToK(j,i)]  (mask of 1-drelu(a_j-a_i))
// reduceReluToLambdaKernel: run phase, counts how many elements i beats.
//   rank[i] = sum_{j>i} drelu(a_i-a_j) + sum_{j<i} (1-drelu(a_j-a_i))

template <typename T>
__global__ void reduceReluMaskToLambdaKernel(int lambda, int n_s, const T* d_reluMask, T* d_lambda_mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lambda) return;
    T sum = 0;
    // j > i: mask for drelu(a_i - a_j) stored at pairIJToK(i, j)
    for (int j = i + 1; j < lambda; j++) {
        sum += (T)(d_reluMask[pairIJToK(i, j, lambda)]);
        gpuMod<T>(sum, n_s);
    }
    // j < i: mask for (1 - drelu(a_j - a_i)) = -mask of drelu(a_j - a_i)
    for (int j = 0; j < i; j++) {
        sum += (T)(-d_reluMask[pairIJToK(j, i, lambda)]);
        gpuMod<T>(sum, n_s);
    }
    d_lambda_mask[i] = sum;
}

template <typename T>
__global__ void reduceReluToLambdaKernel(int lambda, int n_s, const T* d_relu, T* d_lambda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lambda) return;
    T sum = 0;
    // d_relu[k] = 1 - drelu(a_i - a_j), so i beats j iff d_relu[k] = 0
    // j > i: i beats j when d_relu[pairIJToK(i,j)] = 0, contribution = 1 - d_relu[k]
    for (int j = i + 1; j < lambda; j++) {
        sum += (T)(1 - d_relu[pairIJToK(i, j, lambda)]);
        gpuMod<T>(sum, n_s);
    }
    // j < i: i beats j when d_relu[pairIJToK(j,i)] = 1 (drelu(a_j-a_i)=0), contribution = d_relu[k]
    for (int j = 0; j < i; j++) {
        sum += (T)(d_relu[pairIJToK(j, i, lambda)]);
        gpuMod<T>(sum, n_s);
    }
    d_lambda[i] = sum;
}

// Unpack 1-bit-per-element from u32 packed array to u8[] (one byte 0/1 per element). For run() to feed reduceReluMaskToLambdaKernel.
template <typename T>
__global__ void unpackBitsKernel(int N, const u32* d_packed, T* d_out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    d_out[k] = (T)(1-((d_packed[k / 32] >> (k % 32)) & 1));
}

template <typename T>
__global__ void convertToBitwidth1Kernel(const T* input, u8* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = (u8)input[idx];
        gpuMod(output[idx], 1);
    }
}

// Replicate lambda row values to lambda*d: out[i] = in[i/d].
template <typename T>
__global__ void replicateRowToLambdaDKernel(int lambda, int d, const T* d_in, T* d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lambda * d)
        d_out[i] = d_in[i / d];
}

template <typename T>
__global__ void aggregateRowsKernel(int lambda, int p, int bw, const T* d_A, T* d_a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lambda) {
        T sum = 0;
        for (int j = 0; j < p; j++)
            sum += d_A[i * p + j];
        gpuMod(sum, bw);
        d_a[i] = sum;
    }
}

// Tile a lambda*d matrix mu times: out[r*lambda_d + k] = in[k % lambda_d]
template <typename T>
__global__ void tileRowsKernel(int lambda_d, int mu, const T* d_in, T* d_out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < mu * lambda_d)
        d_out[k] = d_in[k % lambda_d];
}

// Expand mu*lambda selection vector to 2*mu*lambda*d (same selection for Y and Z halves):
//   out[k] = select_all[(k % mu_lambda_d) / d]
// where mu_lambda_d = mu * lambda * d.
template <typename T>
__global__ void replicateSelectAllKernel(int mu_lambda_d, int d, const T* d_select_all, T* d_out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < 2 * mu_lambda_d)
        d_out[k] = d_select_all[(k % mu_lambda_d) / d];
}

// Sum lambda rows for each of nblocks blocks of size lambda*d → nblocks*d:
//   out[blk*d + col] = sum_{row=0}^{lambda-1} in[blk*lambda*d + row*d + col]
template <typename T>
__global__ void sumBatchedRowsKernel(int nblocks, int lambda, int bw, int d, const T* d_in, T* d_out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int blk = blockIdx.y;
    if (col >= d || blk >= nblocks) return;
    T sum = 0;
    for (int row = 0; row < lambda; ++row)
        sum += d_in[(u64)blk * lambda * d + (u64)row * d + col];
    gpuMod(sum, bw);
    d_out[blk * d + col] = sum;
}

template <typename T>
SelectTop<T>::SelectTop(const SelectTopParams& p) : p_(p) {
    n_s_ = (p_.lambda <= 1) ? 1 : (int)ceil(log2((double)p_.lambda));
    if (n_s_ < 1) n_s_ = 1;
}

template <typename T>
void SelectTop<T>::keygen(u8** key_as_bytes, int party, int bin, int bout, T* d_A_mask, CMAMask<T>& cma_mask, AESGlobalContext* gaes) {
    u8* start = *key_as_bytes;
    u8* prev = start;
    int lambda = p_.lambda;
    int mu = p_.mu;
    int d = p_.d;
    int m = 4 + p_.scale;
    int n_comp = lambda * (lambda - 1) / 2;
    int n_s_ = p_.n_s_;

    // 1) Ranking: one DReLU for n_comp comparisons
    T* d_diff_mask = (T*)gpuMalloc((size_t)n_comp * sizeof(T));
    diffMaskFromInputKernel<T><<<(n_comp + 255) / 256, 256>>>(lambda, m, d_A_mask, d_diff_mask);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_A_mask);

    auto d_dreluMask = gpuKeyGenDRelu(key_as_bytes, party, m, n_comp, d_diff_mask, gaes);
    auto d_extMask = genGPUZeroExtKey<T>(key_as_bytes, party, 1, n_s_, n_comp, d_dreluMask, gaes);
    T* d_lambda_mask = (T*)gpuMalloc((size_t)lambda * sizeof(T));
    reduceReluMaskToLambdaKernel<<<(lambda + 255) / 256, 256>>>(lambda, n_s_, d_extMask, d_lambda_mask);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_extMask);
    // 2) Generate mu LUT keys; for rank r the input mask is d_lambda_mask - r (mod 2^n_s_)
    int lambda_d = lambda * d;
    int mu_lambda_d = mu * lambda_d;
    int total = 2 * mu_lambda_d;
    
    // All mu LUT keys share the same input mask d_lambda_mask.
    // In run(), the input is decremented by r for rank r, so the DPF evaluates at
    // (d_lambda - r) - d_lambda_mask = revealed - r, correctly selecting rank r.
    T* d_mask_select_all = (T*)gpuMalloc((size_t)mu * lambda * sizeof(T));
    for (int r = 0; r < mu; r++) {
        T* d_mask_select_r = gpuKeyGenLUT<T, T>(key_as_bytes, party, n_s_, bin, lambda, d_lambda_mask, gaes);
        checkCudaErrors(cudaMemcpy(d_mask_select_all + r * lambda, d_mask_select_r,
                                   lambda * sizeof(T), cudaMemcpyDeviceToDevice));
        gpuFree(d_mask_select_r);
    }
    gpuFree(d_lambda_mask);

    // 3) Expand select masks: mu*lambda → 2*mu*lambda*d (Y half + Z half)
    T* d_mask_select_rep_all = (T*)gpuMalloc((size_t)total * sizeof(T));
    replicateSelectAllKernel<T><<<(total + 255) / 256, 256>>>(mu_lambda_d, d, d_mask_select_all, d_mask_select_rep_all);
    gpuFree(d_mask_select_all);

    // 4) Build YZ mask: [Y_mask tiled mu, z_mask tiled mu] = 2*mu*lambda*d
    T* d_yz_mask = (T*)gpuMalloc((size_t)total * sizeof(T));
    tileRowsKernel<T><<<(mu_lambda_d + 255) / 256, 256>>>(lambda_d, mu, cma_mask.d_mask_Y, d_yz_mask);
    tileRowsKernel<T><<<(mu_lambda_d + 255) / 256, 256>>>(lambda_d, mu, cma_mask.d_mask_Z, d_yz_mask + mu_lambda_d);
    checkCudaErrors(cudaDeviceSynchronize());

    // 5) Single mul key for 2*mu*lambda*d elements
    T* d_product_mask = gpuKeygenMul<T>(key_as_bytes, party, bin, bout, total,
                                        d_mask_select_rep_all, d_yz_mask, TruncateType::None, gaes);
    gpuFree(d_mask_select_rep_all);
    gpuFree(d_yz_mask);

    // 6) Sum lambda rows for each of 2*mu blocks → 2*mu*d
    T* d_result_mask = (T*)gpuMalloc((size_t)2 * mu * d * sizeof(T));
    dim3 grid_sum((d + 255) / 256, 2 * mu);
    sumBatchedRowsKernel<T><<<grid_sum, 256>>>(2 * mu, lambda, bout, d, d_product_mask, d_result_mask);
    gpuFree(d_product_mask);
    checkCudaErrors(cudaDeviceSynchronize());

    // 7) Extract output masks: first mu*d = Y_mu, next mu*d = Z_mu
    checkCudaErrors(cudaMemcpy(cma_mask.d_mask_mu_Y, d_result_mask,
                               (size_t)mu * d * sizeof(T), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(cma_mask.d_mask_mu_Z, d_result_mask + mu * d,
                               (size_t)mu * d * sizeof(T), cudaMemcpyDeviceToDevice));
    gpuFree(d_result_mask);
}

template <typename T>
void SelectTop<T>::readkey(u8** key_as_bytes) {
    u8* start = *key_as_bytes;
    u8* prev = start;
    printf("readkey key_as_bytes = %p (start)\n", start);
    int lambda = p_.lambda;
    int d = p_.d;
    key.mu = p_.mu;
    int n_s_ = p_.n_s_;
    int n_comp = lambda * (lambda - 1) / 2;

    key.dreluKey = readGPUDReluKey(key_as_bytes);

    key.zeroExtendKey = readGPUZeroExtKey<T>(key_as_bytes);
    
    printf("read key lutKeys (%d keys)\n", key.mu);
    key.lutKeys.resize(key.mu);
    for (int r = 0; r < key.mu; r++) {
        key.lutKeys[r] = readGPULUTKey<T>(key_as_bytes);
    }

    u64 total = 2ULL * key.mu * lambda * d;
    printf("read key mulKey (N=%llu)\n", (unsigned long long)total);
    key.mulKey = readGPUMulKey<T>(key_as_bytes, total, total, total, TruncateType::None);
}

template <typename T>
void SelectTop<T>::init() {
    // This indicator LUT has no fixed-point scale. Do not read the optional,
    // historically uninitialized comparison-width field as a scale.
    d_topTab = genLUT<T, topRankNoScale<T>>(p_.n_s_, 0, 0);
}

template <typename T>
void SelectTop<T>::run(SigmaPeer* peer, int party, u8** key_as_bytes, int bin, int bout,
    T* d_A, CMAState<T>& cma_state, AESGlobalContext* gaes, Stats* s) {
    peer->sync();  // ensure both parties are in run() before any comm
    int lambda = p_.lambda;
    int mu = p_.mu;
    int d = p_.d;
    int p = p_.p;
    int m = 4 + p_.scale;
    int scale = p_.scale;
    int n_comp = lambda * (lambda - 1) / 2;
    u8* cur = *key_as_bytes;
    int n_s_ = p_.n_s_;

    // 1) Aggregate [a]
    T* d_a = (T*)gpuMalloc((size_t)lambda * sizeof(T));
    aggregateRowsKernel<T><<<(lambda + 255) / 256, 256>>>(lambda, p, m, d_A, d_a);
    // auto h_a = (T *)moveToCPU((u8 *)d_a, lambda * sizeof(T), NULL);
    // for (int i = 0; i < lambda; i++) {
    //     printf("d_a[%d] = %f\n", i, asFloat(h_a[i], m, scale));
    // }
    // cpuFree(h_a);
    checkCudaErrors(cudaDeviceSynchronize());

    // Same diff + DReLU + reduce as keygen so mask and computation match (random mask and run use same (i,j) ordering).
    T* d_diff = (T*)gpuMalloc((size_t)n_comp * sizeof(T));
    diffMaskFromInputKernel<T><<<(n_comp + 255) / 256, 256>>>(lambda, m, d_a, d_diff);
    // auto h_diff = (T *)moveToCPU((u8 *)d_diff, n_comp * sizeof(T), NULL);
    // for (int i = 0; i < n_comp; i++) {
    //     printf("d_diff[%d] = %f\n", i, asFloat(h_diff[i], m, scale));
    // }
    // checkCudaErrors(cudaDeviceSynchronize());
    // printf("begin dcf\n");
    std::vector<u32*> drelu_corrections{key.dreluKey.mask};
    // Include the dealer's correction and output the masked sign bit.
    // unpackBitsKernel complements it before the ascending-loss rank count.
    auto d_relu_res = gpuDcf<T, 1, dReluPrologue<0>, dReluEpilogue<0, true>>(key.dreluKey.dpfKey, party, d_diff, gaes, s, &drelu_corrections);
    // printf("end dcf\n");
    
    peer->reconstructInPlace(d_relu_res, 1, n_comp, s);
    // Extend the bit-width to n_s_
    T* d_relu_res_unpack = (T*)gpuMalloc((size_t)n_comp * sizeof(T));
    unpackBitsKernel<<<(n_comp + 255) / 256, 256>>>(n_comp, d_relu_res, d_relu_res_unpack);
    gpuZeroExt<T>(key.zeroExtendKey, party, peer, d_relu_res_unpack, gaes, s);
    gpuFree(d_relu_res);
    T* d_lambda = (T*)gpuMalloc((size_t)lambda * sizeof(T));
    reduceReluToLambdaKernel<<<(lambda + 255) / 256, 256>>>(lambda, n_s_, d_relu_res_unpack, d_lambda);
    // The count excludes the candidate itself, so ranks are 0..lambda-1.
    // Using lambda here creates an empty rank-0 row and drops the mu-th best row.
    gpuLinearComb(bin, lambda, d_lambda, T(-1), d_lambda, T(lambda - 1));
    auto h_lambda = (T *)moveToCPU((u8 *)d_lambda, lambda * sizeof(T), NULL);
    // for (int i = 0; i < lambda; i++) {
    //     printf("d_lambda[%d] = %d ", i, h_lambda[i]);
    // }
    // printf("\n");
    cpuFree(h_lambda);
    checkCudaErrors(cudaDeviceSynchronize());
    // 2) Collect mu selection vectors via LUT (one per rank), then batch-multiply.
    int lambda_d = lambda * d;
    int mu_lambda_d = mu * lambda_d;
    int total = 2 * mu_lambda_d;

    // Phase 1: run LUT mu times, decrementing d_lambda each iteration, collect into d_select_all.
    // reconstructInPlace is deferred and done once on the full mu*lambda array.
    T* d_select_all = (T*)gpuMalloc((size_t)mu * lambda * sizeof(T));
    for (int rank = 0; rank < mu; rank++) {
        T* d_select_top = gpuDpfLUT(key.lutKeys[rank], peer, party, d_lambda, d_topTab, gaes, s, false);
        checkCudaErrors(cudaMemcpy(d_select_all + rank * lambda, d_select_top,
                                   lambda * sizeof(T), cudaMemcpyDeviceToDevice));
        gpuFree(d_select_top);
        if (rank < mu - 1)
            gpuLinearComb(bin, lambda, d_lambda, T(1), d_lambda, T(-1));
    }
    gpuFree(d_lambda);
    // LUT output masks and the following multiplication use the bin-bit ring.
    // n_s_ is only the rank/LUT input width; truncating here destroys the mask.
    peer->reconstructInPlace(d_select_all, bin, mu * lambda, s);

    // Phase 2: expand select_all (mu*lambda) → d_select_rep_all (2*mu*lambda*d)
    T* d_select_rep_all = (T*)gpuMalloc((size_t)total * sizeof(T));
    replicateSelectAllKernel<T><<<(total + 255) / 256, 256>>>(mu_lambda_d, d, d_select_all, d_select_rep_all);
    gpuFree(d_select_all);

    // Phase 3: tile Y and Z (lambda*d each) to mu*lambda*d, concat → d_YZ (2*mu*lambda*d)
    T* d_YZ = (T*)gpuMalloc((size_t)total * sizeof(T));
    tileRowsKernel<T><<<(mu_lambda_d + 255) / 256, 256>>>(lambda_d, mu, cma_state.d_masked_Y, d_YZ);
    tileRowsKernel<T><<<(mu_lambda_d + 255) / 256, 256>>>(lambda_d, mu, cma_state.d_Z, d_YZ + mu_lambda_d);
    checkCudaErrors(cudaDeviceSynchronize());

    // Phase 4: single batched element-wise multiplication
    T* d_product = gpuMul<T>(peer, party, bin, scale, total, key.mulKey,
                              d_select_rep_all, d_YZ, TruncateType::None, gaes, s);
    gpuFree(d_select_rep_all);
    gpuFree(d_YZ);

    // Phase 5: sum lambda rows for each of 2*mu blocks → 2*mu*d
    T* d_result = (T*)gpuMalloc((size_t)2 * mu * d * sizeof(T));
    dim3 grid_sum((d + 255) / 256, 2 * mu);
    sumBatchedRowsKernel<T><<<grid_sum, 256>>>(2 * mu, lambda, bout, d, d_product, d_result);
    gpuFree(d_product);
    checkCudaErrors(cudaDeviceSynchronize());

    // Phase 6: extract Y_mu (first mu*d) and Z_mu (next mu*d)
    checkCudaErrors(cudaMemcpy(cma_state.d_masked_mu_Y, d_result,
                               (size_t)mu * d * sizeof(T), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(cma_state.d_masked_mu_Z, d_result + mu * d,
                               (size_t)mu * d * sizeof(T), cudaMemcpyDeviceToDevice));
    gpuFree(d_result);
}

template class SelectTop<u64>;
}  // namespace spt
