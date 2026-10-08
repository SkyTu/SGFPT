// Two-party test for Sample protocol: keygen + run with random shares.
// Usage: ./test_sample <party:0|1> <peer_ip> [gpu_id]
// Run twice: party 0 and party 1, with peer_ip pointing to the other machine (or 127.0.0.1 for local).

#include "Server-SGFPT/sample.h"
#include "utils/gpu_data_types.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"
#include "utils/misc_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <party:0|1> <peer_ip> [gpu_id]\n", argv[0]);
        return 1;
    }

    int gpuId = 0;
    if (argc >= 4)
    {
        gpuId = atoi(argv[3]);
        int devCount = 0;
        if (cudaGetDeviceCount(&devCount) != cudaSuccess || gpuId < 0 || gpuId >= devCount)
        {
            fprintf(stderr, "Invalid gpu_id=%d\n", gpuId);
            return 1;
        }
        cudaSetDevice(gpuId);
    }

    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    
    
    int party = atoi(argv[1]);
    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);

    int lambda = 10, mu = 5, d = 10, scale = 24, bin = 64, bout = 64, sqrt_bin = 22, sqrt_scale = 16, exp_bin = 21, exp_scale = 16;
    CMAConfig cfg;
    cfg.d = d; cfg.lambda = lambda; cfg.mu = mu; cfg.scale = scale; cfg.ring = bin;
    cfg.sqrt_scale_sample = cfg.sqrt_scale_update = sqrt_scale;
    cfg.sqrt_bw_sample = cfg.sqrt_bw_update = sqrt_bin;
    cfg.exp_scale = exp_scale; cfg.exp_bw = exp_bin;
    CMAMask<u64> cma_mask(cfg);
    CMAState<u64> cma_state(cfg);
    cma_mask.init();
    cma_state.init();
    spt::Sample<u64> sample;

    u8* startPtr = nullptr;
    u8* curPtr = nullptr;
    getKeyBuf(&startPtr, &curPtr, 4 * OneGB);
    
    setZeroRandomness(true);
    cma_mask.d_mask_C_diag = (u64*)randomGEOnGpu<u64>(d, bin);
    cma_mask.d_mask_m = (u64*)randomGEOnGpu<u64>(d, bin);
    cma_mask.d_mask_sigma = (u64*)randomGEOnGpu<u64>(d, bin);
    
    
    sample.keygen(&curPtr, party, cma_mask, &g);

    size_t keySize = (size_t)(curPtr - startPtr);
    fprintf(stderr, "Party %d keygen done, key size = %zu bytes\n", party, keySize);

    u8* readPtr = startPtr;


    u64* h_m = (u64*)cpuMalloc(d * sizeof(u64));
    u64* h_sigma = (u64*)cpuMalloc(d * sizeof(u64));
    u64* h_C = (u64*)cpuMalloc(d * d * sizeof(u64));

    setZeroRandomness(false);
    // cma_state.d_masked_m = (u64*)getMaskedInputOnGpu<u64>(d, bin, cma_mask.d_mask_m, &h_m, true, scale+3);
    // cma_state.d_masked_sigma = (u64*)getMaskedInputOnGpu<u64>(d, bin, cma_mask.d_mask_sigma, &h_sigma, true, scale+3 );
    // cma_state.d_masked_C_diag = (u64*)getMaskedInputOnGpu<u64>(d, bin, cma_mask.d_mask_C_diag, &h_C, true, scale+3);
    

    checkCudaErrors(cudaMemset(cma_state.d_masked_m,       0, (size_t)d * sizeof(u64)));
    // p_c = 0, p_sigma = 0 (evolution paths)
    checkCudaErrors(cudaMemset(cma_state.d_masked_p_c,     0, (size_t)d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_masked_p_sigma, 0, (size_t)d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_masked_mu_Z, 0, (size_t)mu * d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_masked_mu_Y, 0, (size_t)mu * d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_masked_X, 0, (size_t)lambda * d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_masked_Y, 0, (size_t)lambda * d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_Z, 0, (size_t)lambda * d * sizeof(u64)));
    // sigma = 1.0, C_diag = 1.0  (both as fixed-point: 1 << scale)
    u64 one_fp;
    one_fp = 1 << scale;
        
    u64* h_ones = new u64[d];
    for (int i = 0; i < d; i++) h_ones[i] = one_fp;
    cma_state.d_masked_C_diag = (u64*)moveToGPU((u8*)h_ones, (size_t)d * sizeof(u64), nullptr);
    cma_state.d_masked_sigma = (u64*)moveToGPU((u8*)h_ones, (size_t)d * sizeof(u64), nullptr);
    peer->sync();
    sample.readkey(&readPtr, cma_mask);
    sample.run(peer, party, cma_state, &g, (Stats*)NULL);

    fprintf(stderr, "Party %d run done.\n", party);

    
    gpuLinearComb(bin, d * lambda, cma_state.d_masked_Y, u64(1), cma_state.d_masked_Y, u64(-1), cma_mask.d_mask_Y);
    gpuLinearComb(bin, d * lambda, cma_state.d_masked_X, u64(1), cma_state.d_masked_X, u64(-1), cma_mask.d_mask_X);
    

    u64* h_Y = (u64*)moveToCPU((u8*)cma_state.d_masked_Y, (size_t)d * lambda * sizeof(u64), NULL);
    u64* h_X = (u64*)moveToCPU((u8*)cma_state.d_masked_X, (size_t)d * lambda * sizeof(u64), NULL);
    u64* h_Z = (u64*)moveToCPU((u8*)cma_state.d_Z, (size_t)lambda * d * sizeof(u64), NULL);

    for (int i = 0; i < d * lambda; i++) {
        printf("Y[%d] = %f, X[%d] = %f\n", i, asFloat(h_Y[i],bin,scale), i, asFloat(h_X[i],bin,scale));
    }

    
    for (int i = 0; i < d * lambda; i++) {
        printf("Z[%d] = %f\n", i, asFloat(h_Z[i],bin,scale));
    }

    free(h_m);
    free(h_sigma);
    free(h_C);
    free(h_X);
    free(h_Y);
    
    destroyGPURandomness();

    printf("PASS: test_sample party %d\n", party);
    return 0;
}
