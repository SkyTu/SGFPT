// Algorithm 3 Pi^Update: m-update, sigma-update, C-update.

#include "update.h"
#include "utils/gpu_data_types.h"
#include "utils/gpu_random.h"
#include "utils/misc_utils.h"
#include "utils/helper_cuda.h"
#include "fss/gpu_truncate.h"
#include "fss/gpu_lut.h"
#include "fss/gpu_mul.h"
#include "fss/dcf/gpu_truncate.h"
#include <cassert>
#include <cuda_runtime.h>
#include <cmath>

namespace spt {

using T = u64;
using TIn = u64;
using TOut = u64;

// Y_w[i*d + k] = w[i] * Y_mu[i*d + k] mod 2^bw
// w: shape (mu,), Y_mu: shape (mu, d), Y_w: shape (mu, d)
template <typename T>
__global__ void scaleRowsByWeightKernel(int mu, int d, int bw, const T* __restrict__ w, const T* __restrict__ Y_mu, T* Y_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mu * d) {
        int i = idx / d;
        T tmp = w[i] * Y_mu[idx];
        gpuMod(tmp, bw);
        Y_w[idx] = tmp;
    }
}

// out[k] = scalar * vec[k]  mod 2^bw
// scalar: plaintext T value (fixed-point), vec/out: shape (n,)
template <typename T>
__global__ void plaintextScaleKernel(int n, int bw, T scalar,
                                     const T* __restrict__ vec,
                                     T* out)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        u64 val = (u64)scalar * (u64)vec[k];
        gpuMod(val, bw);
        out[k] = (T)val;
    }
}

// out[0] = Σ_{j=0}^{n-1} vec[j]  mod 2^bw  (scalar reduction)
// vec: shape (n,), out: shape (1,)
template <typename T>
__global__ void vecSumKernel(int n, int bw, const T* __restrict__ vec, T* out)
{
    // single-block reduction with shared memory
    extern __shared__ T shmem[];
    int tid = threadIdx.x;
    T acc = 0;
    for (int i = tid; i < n; i += blockDim.x)
        acc += vec[i];
    shmem[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        T result = shmem[0];
        gpuMod(result, bw);
        *out = (T)result;
    }
}

// z_w[k] = Σ_{i=0}^{mu-1} coeff[i] * Z_mu[i*d + k]  mod 2^bw
// coeff = c_sigma2 * w[i], precomputed as d_csigma2_w
// coeff: shape (mu,), Z_mu: shape (mu, d), z_w: shape (d,)
template <typename T>
__global__ void weightedColSumKernel(int mu, int d, int bw,
                                     T* coeff,
                                     T* Z_mu,
                                     T* z_w)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < d) {
        u64 sum = 0;
        for (int i = 0; i < mu; i++)
            sum += (u64)coeff[i] * (u64)Z_mu[i * d + k];
        gpuMod(sum, bw);
        z_w[k] = (T)sum;
    }
}

template <typename T>
__global__ void mulConstantKernel(int bw, int N, const T* inp1, const T cons, T* out)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        T val = inp1[k] * cons;
        gpuMod(val, bw);
        out[k] = (T)val;
    }
}

template <typename T>
Update<T>::Update() {}

template <typename T>
void Update<T>::keygen(u8** key_as_bytes, int party, CMAMask<T>& cma_mask, AESGlobalContext* gaes)
{
    // m-update mask
    auto d_yw_mask = (T*)gpuMalloc((size_t)cma_mask.d * sizeof(T));
    MatmulParams p;
    p.M = 1;
    p.K = cma_mask.mu;
    p.N = cma_mask.d;
    p.batchSz = 1;
    stdInit(p, cma_mask.ring, cma_mask.scale);
    d_yw_mask = gpuMatmulPlaintext(p, cma_mask.d_w, cma_mask.d_mask_mu_Y, (T *)NULL, false);
    T* d_yw_tr_mask = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrFloor, cma_mask.ring, cma_mask.ring, cma_mask.scale, cma_mask.d, d_yw_mask, gaes);
    T* d_delta_m_mask = gpuKeygenMul<T>(key_as_bytes, party, cma_mask.ring, cma_mask.scale, cma_mask.d, d_yw_tr_mask, cma_mask.d_mask_sigma, TruncateType::TrFloor, gaes);
    gpuLinearComb(cma_mask.ring, cma_mask.d, cma_mask.d_mask_m, T(1), d_delta_m_mask, T(1), cma_mask.d_mask_m);
    gpuFree(d_yw_mask);

    // sigma-update mask
    T* d_Z_w_mask = (T*)gpuMalloc(cma_mask.d * sizeof(T));
    weightedColSumKernel<T><<<(cma_mask.d + 255) / 256, 256>>>(cma_mask.mu, cma_mask.d, cma_mask.ring, cma_mask.d_csigma2_w, cma_mask.d_mask_mu_Z, d_Z_w_mask);
    T* d_cmask_csigma1_p_sigma = (T*)gpuMalloc(cma_mask.d * sizeof(T));
    plaintextScaleKernel<T><<<(cma_mask.d + 255) / 256, 256>>>(cma_mask.d, cma_mask.ring, cma_mask.c_sigma1, cma_mask.d_mask_p_sigma, d_cmask_csigma1_p_sigma);
    gpuLinearComb(cma_mask.ring, cma_mask.d, d_cmask_csigma1_p_sigma, T(1), d_cmask_csigma1_p_sigma, T(1), d_Z_w_mask);
    gpuFree(d_Z_w_mask);
    T* d_p_sigma_mask = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrFloor, cma_mask.ring, cma_mask.ring, cma_mask.scale, cma_mask.d, d_cmask_csigma1_p_sigma, gaes);
    // Persist the output mask for decoding and the next generation's keygen.
    checkCudaErrors(cudaMemcpy(cma_mask.d_mask_p_sigma, d_p_sigma_mask,
                               (size_t)cma_mask.d * sizeof(T), cudaMemcpyDeviceToDevice));
    T* d_p_sigma_sq_mask = gpuKeygenMul<T>(key_as_bytes, party, cma_mask.ring, cma_mask.scale, cma_mask.d, d_p_sigma_mask, d_p_sigma_mask, TruncateType::None, gaes);
    gpuFree(d_p_sigma_mask);
    T* d_p_sigma_sq_sum = (T*)gpuMalloc(sizeof(T));
    vecSumKernel<T><<<1, 256, 256 * sizeof(T)>>>(cma_mask.d, cma_mask.ring, d_p_sigma_sq_mask, d_p_sigma_sq_sum);
    gpuFree(d_p_sigma_sq_mask);
    T* d_p_sigma_sq_tr_mask = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrFloor, cma_mask.ring, cma_mask.ring, cma_mask.scale, 1, d_p_sigma_sq_sum, gaes);
    T* d_sqrt_mask = (T*)gpuMalloc(sizeof(T));
    scaleDownKernel<T><<<(1 + 255) / 256, 256>>>(cma_mask.sqrt_bw_update, cma_mask.scale, cma_mask.sqrt_scale_update, 1, d_p_sigma_sq_tr_mask, d_sqrt_mask);
    gpuFree(d_p_sigma_sq_tr_mask);
    d_sqrt_mask = gpuKeyGenLUT<T, T>(key_as_bytes, party, cma_mask.sqrt_bw_update, cma_mask.ring, 1, d_sqrt_mask, gaes);
    mulConstantKernel<T><<<(1 + 255) / 256, 256>>>(cma_mask.ring, 1, d_sqrt_mask, cma_mask.b_sigma1, d_sqrt_mask);
    // Adding a public constant online changes the plaintext, not its mask.
    T* d_sqrt_tr_mask = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrFloor, cma_mask.ring, cma_mask.ring, cma_mask.scale, 1, d_sqrt_mask, gaes);
    scaleDownKernel<T><<<(1 + 255) / 256, 256>>>(cma_mask.exp_bw, cma_mask.scale, cma_mask.exp_scale, 1, d_sqrt_tr_mask, d_sqrt_tr_mask);
    T* d_exp_mask = gpuKeyGenLUT<T, T>(key_as_bytes, party, cma_mask.exp_bw, cma_mask.ring, 1, d_sqrt_tr_mask, gaes);
    T* d_updated_sigma_mask = gpuKeygenMul<T>(key_as_bytes, party, cma_mask.ring, cma_mask.scale, 1, d_exp_mask, cma_mask.d_mask_sigma, TruncateType::TrFloor, gaes);
    gpuFree(d_exp_mask);
    T* d_updated_sigma_mask_rep = (T*)gpuMalloc((size_t)cma_mask.d * sizeof(T));
    replicateVecKernel<T><<<(cma_mask.d + 255) / 256, 256>>>(1, cma_mask.d, d_updated_sigma_mask, cma_mask.d_mask_sigma);
    T* d_tmp_mask_1 = (T*)gpuMalloc(cma_mask.d * sizeof(T));
    T* d_tmp_mask_2 = (T*)gpuMalloc(cma_mask.d * sizeof(T));
    mulConstantKernel<T><<<(cma_mask.d + 255) / 256, 256>>>(cma_mask.ring, cma_mask.d, cma_mask.d_mask_p_c, cma_mask.c_c1, d_tmp_mask_1);
    mulConstantKernel<T><<<(cma_mask.d + 255) / 256, 256>>>(cma_mask.ring, cma_mask.d, d_yw_tr_mask, cma_mask.c_c2, d_tmp_mask_2);
    gpuLinearComb(cma_mask.ring, cma_mask.d, cma_mask.d_mask_p_c, T(1), d_tmp_mask_1, T(1), d_tmp_mask_2);
    gpuFree(d_tmp_mask_1);
    gpuFree(d_tmp_mask_2);
    gpuFree(d_updated_sigma_mask);
    gpuFree(d_updated_sigma_mask_rep);
    cma_mask.d_mask_p_c = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrFloor, cma_mask.ring, cma_mask.ring, cma_mask.scale, cma_mask.d, cma_mask.d_mask_p_c, gaes);
    T* Y_mu_sq = gpuKeygenMul<T>(key_as_bytes, party, cma_mask.ring, cma_mask.scale, cma_mask.mu * cma_mask.d, cma_mask.d_mask_mu_Y, cma_mask.d_mask_mu_Y, TruncateType::TrFloor, gaes);
    T* d_mask_cmu = (T*)gpuMalloc(cma_mask.d * sizeof(T));
    weightedColSumKernel<T><<<(cma_mask.d + 255) / 256, 256>>>(cma_mask.mu, cma_mask.d, cma_mask.ring, cma_mask.d_cmu_w, Y_mu_sq, d_mask_cmu);
    T* d_mask_c1 = gpuKeygenMul<T>(key_as_bytes, party, cma_mask.ring, cma_mask.scale, cma_mask.d, cma_mask.d_mask_p_c, cma_mask.d_mask_p_c, TruncateType::TrFloor, gaes);
    mulConstantKernel<T><<<(cma_mask.d + 255) / 256, 256>>>(cma_mask.ring, cma_mask.d, d_mask_c1, cma_mask.c_1, d_mask_c1);
    mulConstantKernel<T><<<(cma_mask.d + 255) / 256, 256>>>(cma_mask.ring, cma_mask.d, cma_mask.d_mask_C_diag, cma_mask.c, cma_mask.d_mask_C_diag);
    gpuLinearComb(cma_mask.ring, cma_mask.d, cma_mask.d_mask_C_diag, T(1), d_mask_c1,T(1), d_mask_cmu, T(1), cma_mask.d_mask_C_diag);
    cma_mask.d_mask_C_diag = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrFloor, cma_mask.ring, cma_mask.ring, cma_mask.scale, cma_mask.d, cma_mask.d_mask_C_diag, gaes);
    gpuFree(d_mask_cmu);
    gpuFree(d_mask_c1);
    gpuFree(Y_mu_sq);
}

template <typename T>
void Update<T>::readkey(u8** key_as_bytes, CMAState<T>& cma_state)
{
    key.sttrYwKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
    key.mulYsigmaKey = readGPUMulKey<T>(key_as_bytes, cma_state.d, cma_state.d, cma_state.d, TruncateType::TrFloor);
    key.sttrPsigmaKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
    key.mulPsigSqKey = readGPUMulKey<T>(key_as_bytes, cma_state.d, cma_state.d, cma_state.d, TruncateType::None);
    key.sttrSqsumKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
    key.sqrtKey = readGPULUTKey<T>(key_as_bytes);
    key.sttrSqrtKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
    key.expKey = readGPULUTKey<T>(key_as_bytes);
    key.sigmaUpdateKey = readGPUMulKey<T>(key_as_bytes, 1, 1, 1, TruncateType::TrFloor);
    key.sttrPcKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
    key.mulYmuSqKey = readGPUMulKey<T>(key_as_bytes, cma_state.mu * cma_state.d, cma_state.mu * cma_state.d, cma_state.mu * cma_state.d, TruncateType::TrFloor);
    key.mulPcSqKey = readGPUMulKey<T>(key_as_bytes, cma_state.d, cma_state.d, cma_state.d, TruncateType::TrFloor);
    key.sttrCdiagKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
}

template <typename T>
void Update<T>::run(SigmaPeer* peer, int party,
                   CMAState<T>& cma_state,
                   AESGlobalContext* gaes, Stats* s)
{
    // std::cout << "Running Update" << std::endl;
    // std::cout << "cma_state.mu = " << cma_state.mu << std::endl;
    // std::cout << "cma_state.d = " << cma_state.d << std::endl;
    // std::cout << "cma_state.ring = " << cma_state.ring << std::endl;
    // std::cout << "cma_state.scale = " << cma_state.scale << std::endl;
    // m-update: y_w = w^T · Y_mu, shape (1×mu) · (mu×d) → (d,)
    MatmulParams p;
    p.M = 1;
    p.K = cma_state.mu;
    p.N = cma_state.d;
    p.batchSz = 1;
    stdInit(p, cma_state.ring, cma_state.scale);
    T* d_yw = gpuMatmulPlaintext(p, cma_state.d_w, cma_state.d_masked_mu_Y, (T *)NULL, false);
    d_yw = gpuTruncate(cma_state.ring, cma_state.ring, TruncateType::TrFloor, key.sttrYwKey, cma_state.scale, peer, party, cma_state.d, d_yw, gaes, s);
    T* d_delta_m = gpuMul(peer, party, cma_state.ring, cma_state.scale, cma_state.d, key.mulYsigmaKey, d_yw, cma_state.d_masked_sigma, TruncateType::TrFloor, gaes, s);
    gpuLinearComb(cma_state.ring, cma_state.d, cma_state.d_masked_m, T(1), d_delta_m, T(1), cma_state.d_masked_m);
    gpuFree(d_delta_m);
    
    // sigma-update: z_w, p_sigma, s_p, n_p, sqrt, b, sigma *= (1+b)
    T* d_zw = (T*)gpuMalloc(cma_state.d * sizeof(T));
    weightedColSumKernel<T><<<(cma_state.d + 255) / 256, 256>>>(cma_state.mu, cma_state.d, cma_state.ring, cma_state.d_csigma2_w, cma_state.d_masked_mu_Z, d_zw);
    T* d_csigma1_p_sigma = (T*)gpuMalloc(cma_state.d * sizeof(T));
    plaintextScaleKernel<T><<<(cma_state.d + 255) / 256, 256>>>(cma_state.d, cma_state.ring, cma_state.c_sigma1, cma_state.d_masked_p_sigma, d_csigma1_p_sigma);
    gpuLinearComb(cma_state.ring, cma_state.d, d_csigma1_p_sigma, T(1), d_csigma1_p_sigma, T(1), d_zw);
    cma_state.d_masked_p_sigma = gpuTruncate(cma_state.ring, cma_state.ring, TruncateType::TrFloor, key.sttrPsigmaKey, cma_state.scale, peer, party, cma_state.d, d_csigma1_p_sigma, gaes, s);
    // auto h_p_sigma = (T*)moveToCPU((u8*)d_p_sigma, cma_state.d * sizeof(T), nullptr);
    // for (int i = 0; i < cma_state.d; i++) {
    //     printf("p_sigma[%d] = %f\n", i, asFloat(h_p_sigma[i], cma_state.ring, cma_state.scale));
    // }
    // cpuFree(h_p_sigma);
    gpuFree(d_zw);
    // auto h_p_sigma = (T*)moveToCPU((u8*)cma_state.d_masked_p_sigma, cma_state.d * sizeof(T), nullptr);
    // for (int i = 0; i < 10; i++) {
    //     printf("p_sigma[%d] = %f\n", i, asFloat(h_p_sigma[i], cma_state.ring, cma_state.scale));
    // }
    // cpuFree(h_p_sigma);
    T* d_p_sigma_sq = gpuMul(peer, party, cma_state.ring, cma_state.scale, cma_state.d, key.mulPsigSqKey, cma_state.d_masked_p_sigma, cma_state.d_masked_p_sigma, TruncateType::None, gaes, s);
    T* d_sq_sum = (T*)gpuMalloc(sizeof(T));
    vecSumKernel<T><<<1, 256, 256 * sizeof(T)>>>(cma_state.d, cma_state.ring, d_p_sigma_sq, d_sq_sum);
    gpuFree(d_p_sigma_sq);
    T* d_sq_sum_tr = gpuTruncate(cma_state.ring, cma_state.ring, TruncateType::TrFloor, key.sttrSqsumKey, cma_state.scale, peer, party, 1, d_sq_sum, gaes, s);
    // auto h_sq_sum_tr = (T*)moveToCPU((u8*)d_sq_sum_tr, sizeof(T), nullptr);
    // printf("sq_sum_tr = %f\n", asFloat(h_sq_sum_tr[0], cma_state.ring, cma_state.scale));
    // cpuFree(h_sq_sum_tr);
    auto h_sq_sum_tr = (T*)moveToCPU((u8*)d_sq_sum_tr, sizeof(T), nullptr);
    printf("sqrt input = %f\n", asFloat(h_sq_sum_tr[0], cma_state.ring, cma_state.scale));
    cpuFree(h_sq_sum_tr);
    scaleDownKernel<T><<<(1 + 255) / 256, 256>>>(cma_state.sqrt_bw_update, cma_state.scale, cma_state.sqrt_scale_update, 1, d_sq_sum_tr, d_sq_sum_tr);
    T* d_sqrt = gpuDpfLUT<T, T>(key.sqrtKey, peer, party, d_sq_sum_tr, d_sqrtTab, gaes, s, true);
    auto h_sqrt = (T*)moveToCPU((u8*)d_sqrt, sizeof(T), nullptr);
    printf("sqrt = %f\n", asFloat(h_sqrt[0], cma_state.ring, cma_state.scale));
    cpuFree(h_sqrt);
    // exponent = b_sigma1*||p_sigma|| + b_sigma2（b_sigma2 在 cma_state 中存为 -b2）
    mulConstantKernel<T><<<(1 + 255) / 256, 256>>>(cma_state.ring, 1, d_sqrt, cma_state.b_sigma1, d_sqrt);
    T* d_sqrt_tr =gpuTruncate(cma_state.ring, cma_state.ring, TruncateType::TrFloor, key.sttrSqrtKey, cma_state.scale, peer, party, 1, d_sqrt, gaes, s);
    // auto h_sqrt_tr = (T*)moveToCPU((u8*)d_sqrt_tr, sizeof(T), nullptr);
    // printf("sqrt_tr = %f\n", asFloat(h_sqrt_tr[0], cma_state.ring, cma_state.scale));
    // cpuFree(h_sqrt_tr);
    gpuLinearComb(cma_state.ring, 1, d_sqrt_tr, T(1), d_sqrt_tr, cma_state.b_sigma2);
    // h_sqrt_tr = (T*)moveToCPU((u8*)d_sqrt_tr, sizeof(T), nullptr);
    // printf("sqrt_tr + b_sigma2 = %f\n", asFloat(h_sqrt_tr[0], cma_state.ring, cma_state.scale));
    // cpuFree(h_sqrt_tr);
    scaleDownKernel<T><<<(1 + 255) / 256, 256>>>(cma_state.exp_bw, cma_state.scale, cma_state.exp_scale, 1, d_sqrt_tr, d_sqrt_tr);
    T* d_exp = gpuDpfLUT<T, T>(key.expKey, peer, party, d_sqrt_tr, d_expTab, gaes, s, true);
    auto h_exp = (T*)moveToCPU((u8*)d_exp, sizeof(T), nullptr);
    printf("exp = %f\n", asFloat(h_exp[0], cma_state.ring, cma_state.scale));
    cpuFree(h_exp);
    T* d_masked_sigma = gpuMul(peer, party, cma_state.ring, cma_state.scale, 1, key.sigmaUpdateKey, d_exp, cma_state.d_masked_sigma, TruncateType::TrFloor, gaes, s);
    // {
    //     auto h_masked_sigma = (T*)moveToCPU((u8*)d_masked_sigma, sizeof(T), nullptr);
    //     printf("[update] sigma[0] = %f\n", asFloat(h_masked_sigma[0], cma_state.ring, cma_state.scale));
    //     cpuFree(h_masked_sigma, true);
    // }
    replicateVecKernel<T><<<(cma_state.d + 255) / 256, 256>>>(1, cma_state.d, d_masked_sigma, cma_state.d_masked_sigma);
    // auto h_sigma = (T*)moveToCPU((u8*)cma_state.d_masked_sigma, cma_state.d * sizeof(T), nullptr);
    // for (int i = 0; i < cma_state.d; i++) {
    //     printf("sigma[%d] = %f\n", i, asFloat(h_sigma[i], cma_state.ring, cma_state.scale));
    // }
    // cpuFree(h_sigma);
    // C-update: p_c, Y_sq, C_mu, C_1, C
    T * d_tmp_1 = (T*)gpuMalloc(cma_state.d * sizeof(T));
    T * d_tmp_2 = (T*)gpuMalloc(cma_state.d * sizeof(T));
    mulConstantKernel<T><<<(cma_state.d + 255) / 256, 256>>>(cma_state.ring, cma_state.d, cma_state.d_masked_p_c, cma_state.c_c1, d_tmp_1);
    mulConstantKernel<T><<<(cma_state.d + 255) / 256, 256>>>(cma_state.ring, cma_state.d, d_yw, cma_state.c_c2, d_tmp_2);
    gpuLinearComb(cma_state.ring, cma_state.d, cma_state.d_masked_p_c, T(1), d_tmp_1, T(1), d_tmp_2);
    gpuFree(d_tmp_1);
    gpuFree(d_tmp_2);
    cma_state.d_masked_p_c = gpuTruncate(cma_state.ring, cma_state.ring, TruncateType::TrFloor, key.sttrPcKey, cma_state.scale, peer, party, cma_state.d, cma_state.d_masked_p_c, gaes, s);
    // auto h_p_c = (T*)moveToCPU((u8*)cma_state.d_masked_p_c, cma_state.d * sizeof(T), nullptr);
    // for (int i = 0; i < cma_state.d; i++) {
    //     printf("p_c[%d] = %f\n", i, asFloat(h_p_c[i], cma_state.ring, cma_state.scale));
    // }
    // cpuFree(h_p_c);
    T* Y_mu_sq = gpuMul(peer, party, cma_state.ring, cma_state.scale, cma_state.mu * cma_state.d, key.mulYmuSqKey, cma_state.d_masked_mu_Y, cma_state.d_masked_mu_Y, TruncateType::TrFloor, gaes, s);
    T* d_cmu = (T*)gpuMalloc(cma_state.d * sizeof(T));
    weightedColSumKernel<T><<<(cma_state.d + 255) / 256, 256>>>(cma_state.mu, cma_state.d, cma_state.ring, cma_state.d_cmu_w, Y_mu_sq, d_cmu);
    T* d_c1 = gpuMul(peer, party, cma_state.ring, cma_state.scale, cma_state.d, key.mulPcSqKey, cma_state.d_masked_p_c, cma_state.d_masked_p_c, TruncateType::TrFloor, gaes, s);
    mulConstantKernel<T><<<(cma_state.d + 255) / 256, 256>>>(cma_state.ring, cma_state.d, d_c1, cma_state.c_1, d_c1);
    mulConstantKernel<T><<<(cma_state.d + 255) / 256, 256>>>(cma_state.ring, cma_state.d, cma_state.d_masked_C_diag, cma_state.c, cma_state.d_masked_C_diag);
    gpuLinearComb(cma_state.ring, cma_state.d, cma_state.d_masked_C_diag, T(1), d_c1, T(1), d_cmu, T(1), cma_state.d_masked_C_diag);
    cma_state.d_masked_C_diag = gpuTruncate(cma_state.ring, cma_state.ring, TruncateType::TrFloor, key.sttrCdiagKey, cma_state.scale, peer, party, cma_state.d, cma_state.d_masked_C_diag, gaes, s);
    auto h_C_diag = (T*)moveToCPU((u8*)cma_state.d_masked_C_diag, cma_state.d * sizeof(T), nullptr);

    printf("C_diag[0] = %f\n", asFloat(h_C_diag[0], cma_state.ring, cma_state.scale));
    cpuFree(h_C_diag);
    gpuFree(d_cmu);
    gpuFree(d_c1);
    gpuFree(Y_mu_sq);
}

template <typename T>
void Update<T>::init(CMAState<T>& cma_state) {
    d_sqrtTab = genLUT<T, sqrt<T>>(cma_state.sqrt_bw_update, cma_state.sqrt_scale_update, cma_state.scale);
    d_expTab = genLUT<T, exp<T>>(cma_state.exp_bw, cma_state.exp_scale, cma_state.scale);
}

template class Update<u64>;

}  // namespace spt
