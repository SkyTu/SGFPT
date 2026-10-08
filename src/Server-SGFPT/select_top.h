// Algorithm 2 Pi^SelectTop: from [Y], [Z], [A] select top-mu rows by aggregated accuracy.
// [Y] lambda x d, [Z] lambda x d, [A] lambda x p (accuracy, m = 1+scale).
// Output [Y_mu], [Z_mu] each mu x d. n_s = ceil(log2(lambda)).

#pragma once

#include <vector>
#include "utils/gpu_data_types.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "fss/gpu_lut.h"
#include "fss/gpu_mul.h"
#include "fss/gpu_select.h"
#include "fss/gpu_relu.h"
#include "fss/gpu_truncate.h"
#include "Server-SGFPT/cma_state.h"
#include "Server-SGFPT/cma_mask.h"

struct AESGlobalContext;
struct Stats;

namespace spt {

template <typename T>
struct SelectTopKey {
    int mu;
    int n_s_;
    GPUDReluKey dreluKey;
    GPUZeroExtKey<T> zeroExtendKey;
    // mu LUT keys, one per rank (rank 0 = highest, …, rank mu-1)
    std::vector<GPULUTKey<T>> lutKeys;
    // single mul key for 2*mu*lambda*d elements (Y half + Z half)
    GPUMulKey<T> mulKey;
};

struct SelectTopParams {
    int lambda;
    int mu;
    int d;
    int p;
    int scale;
    int bin_cmp;  // bit-width for DReLU comparison (must cover max_fitness * 2^scale)
    int bout;
    int n_s_;
};

template <typename T = u64>
class SelectTop {
public:
    SelectTopKey<T> key;
    explicit SelectTop(const SelectTopParams& p);
    void keygen(u8** key_as_bytes, int party, int bin, int bout, T* d_A_mask, CMAMask<T>& cma_mask, AESGlobalContext* gaes);
    void readkey(u8** key_as_bytes);
    void run(SigmaPeer* peer, int party, u8** key_as_bytes, int bin, int bout,
                        T* d_A, CMAState<T>& cma_state,
                           AESGlobalContext* gaes, Stats* s);
    void init();
    T *d_topTab;
private:
    SelectTopParams p_;
    int n_s_;
};

}  // namespace spt

#include "select_top.cu"
