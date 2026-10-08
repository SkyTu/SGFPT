// Sample protocol implementation (Algorithm 1 Pi^Sample).
// Include gpu_random.h first so randomGEOnGpu is visible to fss code (e.g. gpu_sstab.h).

#include "sample.h"
#include "utils/gpu_ctx_init.h"
#include "utils/gpu_data_types.h"
#include "utils/gpu_random.h"
#include "utils/misc_utils.h"
#include "utils/gpu_file_utils.h"

#include "fss/gpu_truncate.h"  
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>

namespace spt {



// [q] * Z: Yprime[j*d+k] = q[k] * Z[j*d+k]; Z is u64 (public), q is T (share).
template <typename T>
__global__ void scaleByQKernel(int lambda, int d, int bw, const T* d_q, const u64* d_Z, T* d_Yprime)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lambda * d)
    {
        int k = i % d;
        u64 z = d_Z[i];
        u64 prod = d_q[k] * z;
        gpuMod(prod, bw);
        d_Yprime[i] = prod;
    }
}

template <typename T>   
__global__ void addMeanKernel(int lambda, int d, int bw, const T* d_m, const T* d_sigmaY, T* d_X)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lambda * d)
    {
        int k = i % d;
        T sum = d_m[k] + d_sigmaY[i];
        gpuMod(sum, bw);
        d_X[i] = sum;
    }
}

template <typename T>
Sample<T>::Sample()
{
}


template <typename T>
void Sample<T>::keygen(u8** key_as_bytes, int party, CMAMask<T>& cma_mask, AESGlobalContext* gaes, int generation)
{
    if (generation < 0) throw std::invalid_argument("Sample generation must be nonnegative");
    int lambda = cma_mask.lambda;
    int d = cma_mask.d;
    int scale = cma_mask.scale;
    int bin = cma_mask.ring;
    int shift = cma_mask.scale;
    int NqZ = lambda * d;
    int sqrt_bw = cma_mask.sqrt_bw_sample;
    int sqrt_scale = cma_mask.sqrt_scale_sample;

    u8* curPtr = *key_as_bytes;
    // 1. Sqrt on diagonal of C: use a temporary buffer to avoid corrupting d_mask_C_diag
    T* d_mask_C_scaled = (T*)gpuMalloc((size_t)d * sizeof(T));
    checkCudaErrors(cudaMemcpy(d_mask_C_scaled, cma_mask.d_mask_C_diag, (size_t)d * sizeof(T), cudaMemcpyDeviceToDevice));
    scaleDownKernel<<<(d + 255) / 256, 256>>>(sqrt_bw, scale, sqrt_scale, cma_mask.d, d_mask_C_scaled, d_mask_C_scaled);
    T* d_mask_q = gpuKeyGenLUT<T, T>(&curPtr, party, sqrt_bw, cma_mask.ring, cma_mask.d, d_mask_C_scaled, gaes);
    gpuFree(d_mask_C_scaled);
    

    // 2. Seed for Z (both parties must use same Z)
    unsigned long long seed = 12345678ULL + static_cast<unsigned long long>(generation) * 999999ULL;
    memcpy(curPtr, &seed, sizeof(seed));
    curPtr += sizeof(seed);

    // 3. Z and [q]*Z mask for StTR
    u8* d_Z_u8 = (u8*)gpuMalloc((size_t)NqZ * sizeof(u64));
    gpuRandomStandardNormalWithSeed(NqZ, scale, bin, d_Z_u8, seed);
    auto d_Z = (u64*)d_Z_u8;
    T* d_mask_Yprime = (T*)gpuMalloc((size_t)NqZ * sizeof(T));
    int block = 256;
    int grid = (NqZ + block - 1) / block;
    
    scaleByQKernel<<<grid, block>>>(lambda, d, bin, d_mask_q, d_Z, d_mask_Yprime);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_Z_u8);
    cma_mask.d_mask_Y = genGPUTruncateKey<T,T>(&curPtr, party, TruncateType::TrFloor, bin, bin, shift, NqZ, d_mask_Yprime, gaes);
    gpuFree(d_mask_Yprime);
    gpuFree(d_mask_q);
    cudaMemset(cma_mask.d_mask_Z, 0, NqZ * sizeof(T));
    // 4. Mul [sigma]*[Y]: mask sigma (d), replicate to NqZ; mask Y we have
    T* d_mask_sigma_rep = (T*)gpuMalloc((size_t)NqZ * sizeof(T));
    replicateVecKernel<<<(NqZ + 255) / 256, 256>>>(lambda, d, cma_mask.d_mask_sigma, d_mask_sigma_rep);
    checkCudaErrors(cudaDeviceSynchronize());
    auto d_mask_sigmaY = gpuKeygenMul<T>(&curPtr, party, bin, scale, NqZ, d_mask_sigma_rep, cma_mask.d_mask_Y, TruncateType::TrFloor, gaes);
    cma_mask.d_mask_X = (T*)gpuMalloc((size_t)NqZ * sizeof(T));
    addMeanKernel<<<(NqZ + 255) / 256, 256>>>(lambda, d, bin, cma_mask.d_mask_m, d_mask_sigmaY, cma_mask.d_mask_X);
    checkCudaErrors(cudaDeviceSynchronize());
    
    gpuFree(d_mask_sigmaY);
    gpuFree(d_mask_sigma_rep);

    *key_as_bytes = curPtr;
}

template <typename T>
void Sample<T>::readkey(u8** key_as_bytes, CMAMask<T>& cma_mask)
{
    u64 NqZ = cma_mask.lambda * cma_mask.d;
    // Order must match keygen: LUT key, then d_mask_diag shares, then seed, then StTR, then mul.
    key.sqrtKey = readGPULUTKey<T>(key_as_bytes);
    // key.d_mask_diag = (T*)moveToGPU(*key_as_bytes, (size_t)cma_mask.d * sizeof(T), nullptr);
    // *key_as_bytes += (size_t)cma_mask.d * sizeof(T);
    key.seed = *(unsigned long long*)*key_as_bytes;
    *key_as_bytes += sizeof(unsigned long long);  // skip seed written in keygen
    key.sttrKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
    key.mulKey = readGPUMulKey<T>(key_as_bytes, (u64)NqZ, (u64)NqZ, (u64)NqZ, TruncateType::TrFloor);
}

template <typename T>
void Sample<T>::run(SigmaPeer* peer, int party, CMAState<T>& cma_state,
                   AESGlobalContext* gaes, Stats* s, int generation)
{
    if (generation < 0 || key.seed != 12345678ULL + static_cast<unsigned long long>(generation) * 999999ULL)
        throw std::invalid_argument("Sample key does not match generation; generate fresh preprocessing for this generation");
    int lambda = cma_state.lambda;
    int d = cma_state.d;
    int scale = cma_state.scale;
    int bin = cma_state.ring;
    int shift = cma_state.scale;
    int NqZ = cma_state.lambda * cma_state.d;
    int sqrt_bw = cma_state.sqrt_bw_sample;
    int sqrt_scale = cma_state.sqrt_scale_sample;

    // 2. Sqrt on diagonal -> q; use a temporary buffer to avoid corrupting d_masked_C_diag
    T* d_sqrtTab = genLUT<T, sqrt<T>>(sqrt_bw, sqrt_scale, cma_state.scale);
    T* d_C_scaled = (T*)gpuMalloc((size_t)d * sizeof(T));
    checkCudaErrors(cudaMemcpy(d_C_scaled, cma_state.d_masked_C_diag, (size_t)d * sizeof(T), cudaMemcpyDeviceToDevice));
    scaleDownKernel<<<(d + 255) / 256, 256>>>(sqrt_bw, scale, sqrt_scale, cma_state.d, d_C_scaled, d_C_scaled);
    T* d_q = gpuDpfLUT<T, T>(key.sqrtKey, peer, party, (T*)d_C_scaled, d_sqrtTab, gaes, s, true);
    gpuFree(d_C_scaled);

    gpuFree(d_sqrtTab);
    
    // 3. Z and [q]*Z (local)
    // The dealer already used this generation's seed for the q*Z truncation mask.
    gpuRandomStandardNormalWithSeed(NqZ, scale, bin, (u8*)cma_state.d_Z, key.seed);
    // verify correctness of normalization
    // auto h_Z = (u64*)moveToCPU((u8*)d_Z, (size_t)NqZ * sizeof(u64), nullptr);
    // for (int i = 0; i < NqZ; i++) {
    //     cout << "Z[" << i << "] = " << asFloat(h_Z[i], bin, scale) << endl;
    // }
    // cpuFree(h_Z);

    T* d_Yprime = (T*)gpuMalloc((size_t)NqZ * sizeof(T));
    int block = 256;
    int grid = (NqZ + block - 1) / block;
    scaleByQKernel<<<grid, block>>>(lambda, d, bin, d_q, cma_state.d_Z, d_Yprime);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_q);


    // 4. [Y] = StTR([Y'])
    cma_state.d_masked_Y = gpuTruncate(bin, bin, TruncateType::TrFloor, key.sttrKey, scale, peer, party, NqZ, d_Yprime, gaes, s);
    // auto h_Y = (T*)moveToCPU((u8*)cma_state.d_masked_Y, (size_t)NqZ * sizeof(T), nullptr);
    // for (int i = 0; i < NqZ; i++) {
    //     cout << "Y[" << i << "] = " << asFloat(h_Y[i],bin,scale) << endl;
    // }
    // free(h_Y);

    // 5. [sigma]*[Y] (element-wise, sigma replicated)
    T* d_sigma_rep = (T*)gpuMalloc((size_t)NqZ * sizeof(T));
    replicateVecKernel<<<(NqZ + 255) / 256, 256>>>(lambda, d, cma_state.d_masked_sigma, d_sigma_rep);
    checkCudaErrors(cudaDeviceSynchronize());
    T* d_sigmaY = gpuMul<T>(peer, party, bin, scale, NqZ, key.mulKey, d_sigma_rep, cma_state.d_masked_Y, TruncateType::TrFloor, gaes, s);
    gpuFree(d_sigma_rep);
    // auto h_sigmaY = (T*)moveToCPU((u8*)d_sigmaY, (size_t)NqZ * sizeof(T), nullptr);
    // for (int i = 0; i < NqZ; i++) {
    //     cout << "sigmaY[" << i << "] = " << asFloat(h_sigmaY[i],bin,scale) << endl;
    // }
    // free(h_sigmaY);

    // 6. [X] = [m] + d_sigmaY (add mean row-wise)
    
    addMeanKernel<<<grid, block>>>(lambda, d, bin, cma_state.d_masked_m, d_sigmaY, cma_state.d_masked_X);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_sigmaY);
}

template class Sample<u64>;

}  // namespace spt
