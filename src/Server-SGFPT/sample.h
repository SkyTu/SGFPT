// Sample protocol (Algorithm 1 Pi^Sample): secret-shared [m], [sigma], [C] -> [X], [Y].
// Uses NDG (standard normal), Pi^Sqrt on Diag([C]), [q]*Z, Pi^StTR, Pi^EleMul, then add [m].
// All matrices row-major; Z, [Y], [X] are lambda x d.

#pragma once

#include "utils/gpu_data_types.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "fss/gpu_lut.h"
#include "fss/gpu_mul.h"
#include "fss/dcf/gpu_truncate.h"
#include "Server-SGFPT/cma_state.h"
#include "Server-SGFPT/cma_mask.h"

struct AESGlobalContext;
struct Stats;

namespace spt {

template <typename T>
struct SampleKey {
    T *d_mask_diag;
    GPULUTKey<T> sqrtKey;
    GPUTruncateKey<T> sttrKey;
    GPUMulKey<T> mulKey;
    unsigned long long seed;
};


// Sample protocol: keygen (dealer) and run (both parties).
// Keygen: call with party 0 or 1, key buffer, gaes; writes keys to key_as_bytes.
// Run: each party has key buffer, d_m (d), d_sigma (d), d_C (d*d row-major);
//      outputs d_X (lambda*d), d_Y (lambda*d). Caller allocates d_X, d_Y.
template <typename T = u64>
class Sample {
public:
    SampleKey<T> key;

    explicit Sample();

    // Preprocessing must use the same generation as run(), and the current state masks.
    void keygen(u8** key_as_bytes, int party, CMAMask<T>& cma_mask, AESGlobalContext* gaes, int generation = 0);

    void readkey(u8** key_as_bytes, CMAMask<T>& cma_mask);

    void run(SigmaPeer* peer, int party, CMAState<T>& cma_state, AESGlobalContext* gaes, Stats* s, int generation = 0);


};

}  // namespace spt

#include "sample.cu"
