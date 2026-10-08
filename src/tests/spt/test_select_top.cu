// SelectTop test: dealer generates per-DH masks; keygen receives their row sums.
// Inputs and outputs use additive masks. Nonzero masks are the default.
// Usage: ./test_select_top <party:0|1> <peer_ip> [gpu_id] [zero|nonzero]
// Run both parties (e.g. party 0 and 1 with peer_ip 127.0.0.1 for local).

#include "Server-SGFPT/select_top.h"
#include "utils/gpu_data_types.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <cuda_runtime.h>
#include "Server-SGFPT/cma_state.h"
#include "Server-SGFPT/cma_mask.h"

// Optional read-only audit of the original test's communication path.
// It records before/after values and always delegates to the existing implementation.
class ReconstructionTracePeer : public GpuPeer {
public:
    explicit ReconstructionTracePeer(int party_id, bool enabled)
        : GpuPeer(true), trace_party(party_id), trace_enabled(enabled) {}
    using GpuPeer::reconstructInPlace;
    const void* score_input = nullptr;
    int score_reconstruction_calls = 0;
    int reconstruction_calls = 0;

    void reconstructInPlace(u64* ptr, int bw, u64 n, Stats* stats) override {
        record<u64>(ptr, bw, n, "before");
        GpuPeer::reconstructInPlace(ptr, bw, n, stats);
        record<u64>(ptr, bw, n, "after");
    }
    void reconstructInPlace(u32* ptr, int bw, u64 n, Stats* stats) override {
        record<u32>(ptr, bw, n, "before");
        GpuPeer::reconstructInPlace(ptr, bw, n, stats);
        record<u32>(ptr, bw, n, "after");
    }
private:
    int trace_party;
    bool trace_enabled;
    template<typename T>
    void record(T* ptr, int bw, u64 n, const char* phase) {
        if (strcmp(phase, "before") == 0) {
            ++reconstruction_calls;
            if (static_cast<const void*>(ptr) == score_input) ++score_reconstruction_calls;
        }
        if (!trace_enabled) return;
        // Bit-widths 1/2 use packed u32 storage; other widths use one T per value.
        const size_t words = bw <= 2 ? (bw * n + 31) / 32 : n;
        const size_t shown = std::min<size_t>(words, 8);
        std::vector<T> values(shown);
        checkCudaErrors(cudaMemcpy(values.data(), ptr, shown * sizeof(T), cudaMemcpyDeviceToHost));
        printf("RECON_TRACE {\"party\":%d,\"call\":%d,\"phase\":\"%s\",\"word_bits\":%zu,\"bw\":%d,\"n\":%llu,\"is_score_input\":%s,\"values\":[",
               trace_party, reconstruction_calls, phase, sizeof(T)*8, bw,
               (unsigned long long)n, static_cast<const void*>(ptr) == score_input ? "true" : "false");
        for(size_t i=0;i<shown;++i) printf("%s%llu", i ? "," : "", (unsigned long long)values[i]);
        printf("]}\n");fflush(stdout);
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <party:0|1> <peer_ip> [gpu_id] [zero|nonzero]\n", argv[0]);
        return 1;
    }
    int gpuId = (argc > 3) ? atoi(argv[3]) : 0;
    bool zero_masks = argc > 4 && strcmp(argv[4], "zero") == 0;
    if (argc > 4 && !zero_masks && strcmp(argv[4], "nonzero") != 0) {
        fprintf(stderr, "Mask mode must be zero or nonzero\n");
        return 2;
    }
    printf("gpuId = %d\n", gpuId);
    cudaSetDevice(gpuId);

    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    u8* startPtr = nullptr;
    u8* curPtr = nullptr;
    getKeyBuf(&startPtr, &curPtr, 4 * OneGB);
    initGPURandomness();

    int party = atoi(argv[1]);
    auto peer = new ReconstructionTracePeer(party, std::getenv("SPT_TRACE_RECONSTRUCT") != nullptr);
    peer->connect(party, argv[2]);
    
    int lambda = 10, mu = 5, d = 10, scale = 24, bin = 64, bout = 64, sqrt_bin = 22, sqrt_scale = 16, exp_bin = 21, exp_scale = 16;
    CMAConfig cfg;
    cfg.d = d; cfg.lambda = lambda; cfg.mu = mu; cfg.scale = scale; cfg.ring = bout;
    cfg.sqrt_scale_sample = cfg.sqrt_scale_update = sqrt_scale;
    cfg.sqrt_bw_sample = cfg.sqrt_bw_update = sqrt_bin;
    cfg.exp_scale = exp_scale; cfg.exp_bw = exp_bin;
    CMAState<u64> cma_state(cfg);
    CMAMask<u64> cma_mask(cfg);
    cma_state.init();
    cma_mask.init();
    spt::SelectTopParams p;
    p.lambda = lambda;
    p.mu = mu;
    p.d = d;
    p.p = 3;
    p.scale = scale;
    p.bout = bout;
    p.n_s_ = (lambda <= 1) ? 1 : (int)ceil(log2((double)lambda));
    p.n_s_ = max(p.n_s_, 8);
    int m = 4 + scale;  // Same score ring as SelectTop::keygen/run.
    spt::SelectTop<u64> st(p);
    // The two processes emulate a dealer using the same test RNG seed.
    // Keep the per-DH masks for input construction; keygen consumes only their sums.
    setZeroRandomness(zero_masks);
    u64* d_mask_A_per_dh = randomGEOnGpu<u64>(p.lambda * p.p, bout);
    auto h_mask_A_per_dh = (u64*)moveToCPU((u8*)d_mask_A_per_dh,
                                         (size_t)p.lambda * p.p * sizeof(u64), nullptr);
    std::vector<u64> h_mask_A_sum(p.lambda, 0);
    int nonzero_dh_masks = 0;
    for (int j = 0; j < p.lambda; ++j) {
        for (int h = 0; h < p.p; ++h) {
            u64 r = h_mask_A_per_dh[j * p.p + h];
            h_mask_A_sum[j] += r;
            nonzero_dh_masks += r != 0;
        }
        h_mask_A_sum[j] &= (u64(1) << m) - 1;
    }
    assert(nonzero_dh_masks == (zero_masks ? 0 : p.lambda * p.p));
    printf("MASK_SETUP mode=%s per_DH_nonzero=%d/%d score_bits=%d\n",
           zero_masks ? "zero" : "nonzero", nonzero_dh_masks, p.lambda * p.p, m);
    u64* d_mask_A = (u64*)moveToGPU((u8*)h_mask_A_sum.data(),
                                   (size_t)p.lambda * sizeof(u64), nullptr);

    cma_mask.d_mask_Y = (u64*)randomGEOnGpu<u64>(p.lambda * p.d, bout);
    cma_mask.d_mask_Z = (u64*)randomGEOnGpu<u64>(p.lambda * p.d, bout);
    cma_mask.d_mask_mu_Y = (u64*)randomGEOnGpu<u64>(p.mu * p.d, bout);
    cma_mask.d_mask_mu_Z = (u64*)randomGEOnGpu<u64>(p.mu * p.d, bout);
    
    st.keygen(&curPtr, party, bin, bout, d_mask_A, cma_mask, &g);           
    // keygen frees the aggregate mask; the per-DH masks remain available below.
    size_t keySize = (size_t)(curPtr - startPtr);
    fprintf(stderr, "SelectTop keygen done, key size = %zu bytes\n", keySize);

    // Read key into st.key for run()
    u8* readPtr = startPtr;
    std::cout << "Begin readkey" << std::endl;
    st.readkey(&readPtr);
    st.init();
    std::cout << "End readkey" << std::endl;
    sleep(2);
    // Random plaintext test inputs; this does not change the already generated masks.
    setZeroRandomness(false);
    u64* h_Y_plain = nullptr;
    cma_state.d_masked_Y = (u64*)getMaskedInputOnGpu<u64>(p.lambda * p.d, bout, cma_mask.d_mask_Y, &h_Y_plain, true, 15);
 
    int num_parties = p.p; 
    std::vector<double> party_holdings(num_parties, 1.0 / num_parties); 

    std::mt19937 rng(42);  // fixed seed so both parties generate identical accuracies
    std::uniform_real_distribution<double> acc_dist(0.1, 0.9);
    std::vector<double> accuracies(p.lambda, 0.0);
    for (int j = 0; j < p.lambda; ++j) {
        accuracies[j] = acc_dist(rng);
    }
    // Exercise tied losses as well as selecting the smallest loss.
    accuracies[8] = accuracies[5];

    // h_A layout: h_A[model * num_parties + party], matching aggregateRowsKernel (d_A[model * p + party])
    std::vector<u64> h_A(num_parties * p.lambda, 0);
    for (int j = 0; j < p.lambda; ++j) {
        for (int i = 0; i < num_parties; ++i) {
            double val = party_holdings[i] * accuracies[j] * std::pow(2.0, p.scale);
            h_A[j * num_parties + i] = (u64)(val + 0.5);
        }
    }

    std::vector<double> merged_accuracies(p.lambda, 0.0);
    for (int j = 0; j < p.lambda; ++j) {
        double sum = 0.0;
        for (int i = 0; i < num_parties; ++i) {
            sum += static_cast<double>(h_A[j * num_parties + i]) / std::pow(2.0, p.scale);
        }
        merged_accuracies[j] = sum;
    }

    printf("Merged accuracies:\n");
    for (int j = 0; j < p.lambda; ++j) {
        printf("merged_accuracies[%d] = %.6f\n", j, merged_accuracies[j]);
    }

    u64* d_A = (u64*)moveToGPU((u8*)h_A.data(), p.lambda * num_parties * sizeof(u64), NULL);
    gpuLinearComb(m, p.lambda * num_parties, d_A, u64(1), d_A, u64(1), d_mask_A_per_dh);
    // Independently check the caller/keygen contract before entering SelectTop:
    // sum_h(A[j,h] + r[j,h]) - sum_h(r[j,h]) == sum_h(A[j,h]) mod 2^m.
    auto h_masked_A = (u64*)moveToCPU((u8*)d_A,
                                     (size_t)p.lambda * p.p * sizeof(u64), nullptr);
    const u64 score_ring_mask = (u64(1) << m) - 1;
    for (int j = 0; j < p.lambda; ++j) {
        u64 masked_sum = 0, plain_sum = 0;
        for (int h = 0; h < p.p; ++h) {
            masked_sum += h_masked_A[j * p.p + h];
            plain_sum += h_A[j * p.p + h];
        }
        assert(((masked_sum - h_mask_A_sum[j]) & score_ring_mask) ==
               (plain_sum & score_ring_mask));
    }
    printf("PASS input mask alignment: %d candidates, %d DHs\n", p.lambda, p.p);
    if (std::getenv("SPT_TRACE_RECONSTRUCT") != nullptr) {
        // Test-only dealer data for decoding the first comparison offline.
        // No protocol values, keys, or reconstruction calls are changed.
        printf("COMPARISON_REFERENCE {\"party\":%d,\"score_bits\":%d,\"score_sums\":[", party, m);
        for (int j = 0; j < p.lambda; ++j) {
            u64 sum = 0;
            for (int h = 0; h < p.p; ++h) sum += h_A[j * p.p + h];
            printf("%s%llu", j ? "," : "", (unsigned long long)(sum & score_ring_mask));
        }
        printf("],\"aggregate_masks\":[");
        for (int j = 0; j < p.lambda; ++j)
            printf("%s%llu", j ? "," : "", (unsigned long long)h_mask_A_sum[j]);
        printf("],\"correction_share_words\":[");
        const int n_comp = p.lambda * (p.lambda - 1) / 2;
        for (int word = 0; word < (n_comp + 31) / 32; ++word)
            printf("%s%u", word ? "," : "", st.key.dreluKey.mask[word]);
        printf("]}\n");
        fflush(stdout);
    }
    cpuFree(h_masked_A);
    cpuFree(h_mask_A_per_dh);
    gpuFree(d_mask_A_per_dh);

    peer->score_input = d_A;  // Observe whether this exact score array is ever reconstructed.
    // d_A即为最终矩阵
    u64* h_Z_plain = nullptr;
    cma_state.d_Z = (u64*)getMaskedInputOnGpu<u64>((u64)p.lambda * p.d, bout, cma_mask.d_mask_Z, &h_Z_plain, true, 15);
    printf("h_Y_plain:\n");
    for (int i = 0; i < p.lambda; i++){
        for(int j = 0; j < p.d; j++){
            printf("%f ", asFloat(h_Y_plain[i * p.d + j], bin, p.scale));
        }
        printf("\n");
    }

    printf("h_Z_plain:\n");
    for (int i = 0; i < p.lambda; i++){
        for(int j = 0; j < p.d; j++){
            printf("%f ", asFloat(h_Z_plain[i * p.d + j], bin, p.scale));
        }
        printf("\n");
    }

    peer->sync();
    printf("Before run\n");
    st.run(peer, party, &readPtr, bin, bout, d_A, cma_state, &g, (Stats*)NULL);
    fprintf(stderr, "SelectTop run done.\n");
    // Decode with the output masks generated by keygen before comparing to plaintext.
    gpuLinearComb(bout, p.mu * p.d, cma_state.d_masked_mu_Y,
                  u64(1), cma_state.d_masked_mu_Y, u64(-1), cma_mask.d_mask_mu_Y);
    gpuLinearComb(bout, p.mu * p.d, cma_state.d_masked_mu_Z,
                  u64(1), cma_state.d_masked_mu_Z, u64(-1), cma_mask.d_mask_mu_Z);
    auto h_Y_mu = (u64*)moveToCPU((u8*)cma_state.d_masked_mu_Y, p.mu * p.d * sizeof(u64), NULL);
    printf("h_Y_mu:\n");
    for (int i = 0; i < p.mu; i++){
        for(int j = 0; j < p.d; j++){
            printf("%f ", asFloat(h_Y_mu[i * p.d + j], bin, p.scale));
        }
        printf("\n");
    }
    auto h_Z_mu = (u64*)moveToCPU((u8*)cma_state.d_masked_mu_Z, p.mu * p.d * sizeof(u64), NULL);
    printf("h_Z_mu:\n");
    for (int i = 0; i < p.mu; i++){
        for(int j = 0; j < p.d; j++){
            printf("%f ", asFloat(h_Z_mu[i * p.d + j], bin, p.scale));
        }
        printf("\n");
    }
    // Compare selected payloads against an independent CPU ordering of the losses.
    std::vector<int> order(p.lambda);
    for (int i = 0; i < p.lambda; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return merged_accuracies[a] < merged_accuracies[b];
    });
    std::vector<bool> selected(p.lambda, false);
    int bad_rows = 0;
    for (int i = 0; i < p.mu; ++i) {
        int match = -1;
        for (int candidate = 0; candidate < p.lambda; ++candidate) {
            bool same = !selected[candidate];
            for (int j = 0; j < p.d; ++j) {
                same = same && h_Y_mu[i * p.d + j] == h_Y_plain[candidate * p.d + j];
                same = same && h_Z_mu[i * p.d + j] == h_Z_plain[candidate * p.d + j];
            }
            if (same) { match = candidate; break; }
        }
        // Tied candidates may use either order, but must have the expected loss
        // and preserve their Y/Z payload pair without duplicating a candidate.
        bool correct = match >= 0 && merged_accuracies[match] == merged_accuracies[order[i]];
        if (!correct) ++bad_rows;
        if (match >= 0) selected[match] = true;
        printf("CHECK rank=%d matched_candidate=%d expected_loss=%.9f status=%s\n",
               i, match, merged_accuracies[order[i]], correct ? "PASS" : "FAIL");
    }
    cpuFree(h_Y_mu);
    cpuFree(h_Z_mu);
    printf("RECON_SUMMARY party=%d total=%d score_input_calls=%d\n",
           party, peer->reconstruction_calls, peer->score_reconstruction_calls);
    assert(peer->score_reconstruction_calls == 0);
    peer->close();
    destroyGPURandomness();
    printf("%s: test_select_top party=%d mask=%s bad_rows=%d/%d\n",
           bad_rows ? "FAIL" : "PASS", party, zero_masks ? "zero" : "nonzero", bad_rows, p.mu);
    return bad_rows ? 1 : 0;
}
