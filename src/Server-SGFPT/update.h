// Algorithm 3 Pi^Update: update [m], [sigma], [C] from [Y_mu], [Z_mu], [p_c], [p_sigma].
// All matrices row-major. C is diagonal (d elements).

#pragma once

#include "utils/gpu_data_types.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "fss/gpu_matmul.h"
#include "fss/gpu_truncate.h"
#include "fss/gpu_lut.h"
#include "fss/gpu_mul.h"
#include "Server-SGFPT/cma_state.h"
#include "Server-SGFPT/cma_mask.h"

struct AESGlobalContext;
struct Stats;

template <typename T>
struct UpdateKey {
    GPUTruncateKey<T> sttrYwKey;
    GPUMulKey<T> mulYsigmaKey;
    GPUTruncateKey<T> sttrPsigmaKey;
    GPUMulKey<T> mulPsigSqKey;
    GPUTruncateKey<T> sttrSqsumKey;
    GPULUTKey<T> sqrtKey;
    GPUTruncateKey<T> sttrSqrtKey;
    GPULUTKey<T> expKey;
    GPUMulKey<T> sigmaUpdateKey;
    GPUTruncateKey<T> sttrPcKey;
    GPUMulKey<T> mulYmuSqKey;
    GPUMulKey<T> mulPcSqKey;
    GPUTruncateKey<T> sttrCdiagKey;
};

namespace spt {

template <typename T>
class Update {
public:
    explicit Update();
    UpdateKey<T> key;
    void keygen(u8** key_as_bytes, int party, CMAMask<T>& cma_mask, AESGlobalContext* gaes);
    void readkey(u8** key_as_bytes, CMAState<T>& cma_state);
    void run(SigmaPeer* peer, int party,
            CMAState<T>& cma_state,
            AESGlobalContext* gaes, Stats* s);
    void init(CMAState<T>& cma_state);
    T *d_sqrtTab;
    T *d_expTab;
    
            
};

}  // namespace spt

#include "update.cu"
