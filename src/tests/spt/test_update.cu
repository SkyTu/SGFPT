// Minimal test: keygen + run for Update protocol.
// Usage: ./test_update <party:0|1> <peer_ip> [gpu_id]

#include "Server-SGFPT/update.h"
#include "utils/gpu_data_types.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include <cmath>
#include "Server-SGFPT/cma_state.h"
#include "Server-SGFPT/cma_mask.h"

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

    int lambda = 10, mu = 5, d = 10, scale = 24, bout = 64, sqrt_bin = 22, sqrt_scale = 16, exp_bin = 21, exp_scale = 16;
    CMAConfig cfg;
    cfg.d = d; cfg.lambda = lambda; cfg.mu = mu; cfg.scale = scale; cfg.ring = bout;
    cfg.sqrt_scale_sample = cfg.sqrt_scale_update = sqrt_scale;
    cfg.sqrt_bw_sample = cfg.sqrt_bw_update = sqrt_bin;
    cfg.exp_scale = exp_scale; cfg.exp_bw = exp_bin;
    CMAState<u64> cma_state(cfg);
    cma_state.init();
    CMAMask<u64> cma_mask(cfg);
    cma_mask.init();
    setZeroRandomness(true);
    cma_mask.d_mask_m = (u64*)randomGEOnGpu<u64>(d, bout);
    cma_mask.d_mask_sigma = (u64*)randomGEOnGpu<u64>(d, bout);
    cma_mask.d_mask_C_diag = (u64*)randomGEOnGpu<u64>(d, bout);
    cma_mask.d_mask_p_c = (u64*)randomGEOnGpu<u64>(d, bout);
    cma_mask.d_mask_p_sigma = (u64*)randomGEOnGpu<u64>(d, bout);
    cma_mask.d_mask_mu_Y = (u64*)randomGEOnGpu<u64>(mu * d, bout);
    cma_mask.d_mask_mu_Z = (u64*)randomGEOnGpu<u64>(mu * d, bout);

    setZeroRandomness(false);
    u64 *h_m = nullptr, *h_sigma = nullptr, *h_C_diag = nullptr;
    u64 *h_p_c = nullptr, *h_p_sigma = nullptr, *h_mu_Y = nullptr, *h_mu_Z = nullptr;
    cma_state.d_masked_m      = (u64*)getMaskedInputOnGpu<u64>(d,      bout, cma_mask.d_mask_m,      &h_m,      true, scale+1);
    cma_state.d_masked_sigma  = (u64*)getMaskedInputOnGpu<u64>(d,      bout, cma_mask.d_mask_sigma,  &h_sigma,  true, scale+3);
    cma_state.d_masked_C_diag = (u64*)getMaskedInputOnGpu<u64>(d,      bout, cma_mask.d_mask_C_diag, &h_C_diag, true, scale+1);
    cma_state.d_masked_p_c    = (u64*)getMaskedInputOnGpu<u64>(d,      bout, cma_mask.d_mask_p_c,    &h_p_c,    true, scale+1);
    cma_state.d_masked_p_sigma= (u64*)getMaskedInputOnGpu<u64>(d,      bout, cma_mask.d_mask_p_sigma,&h_p_sigma,true, scale+1);
    cma_state.d_masked_mu_Y   = (u64*)getMaskedInputOnGpu<u64>(mu * d, bout, cma_mask.d_mask_mu_Y,   &h_mu_Y,   true, scale+1);
    cma_state.d_masked_mu_Z   = (u64*)getMaskedInputOnGpu<u64>(mu * d, bout, cma_mask.d_mask_mu_Z,   &h_mu_Z,   true, scale+1);
    
    auto h_masked_sigma = (u64*)moveToCPU((u8*)cma_state.d_masked_sigma, d *  sizeof(u64), nullptr);
    for (int i = 0; i < d; i++) {
        printf("sigma[%d] = %f\n", i, asFloat(h_masked_sigma[i], bout,scale));
    }
    fprintf(stderr, "\n");
    free(h_masked_sigma);

    spt::Update<u64> update;
    u8* startPtr = nullptr;
    u8* curPtr = nullptr;
    getKeyBuf(&startPtr, &curPtr, 4 * OneGB);
    setZeroRandomness(true);
    update.keygen(&curPtr, party, cma_mask, &g);

    update.readkey(&startPtr, cma_state);
    update.init(cma_state);

    update.run(peer, party, cma_state, &g, (Stats*)NULL);
    size_t keySize = (size_t)(curPtr - startPtr);
    fprintf(stderr, "Party %d keygen done, key size = %zu bytes\n", party, keySize);

    printf("PASS: test_update party %d\n", party);
    return 0;
}
