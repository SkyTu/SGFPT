/*
 * SGFPT two-party research simulation driver.
 * Runs Sample -> Python DH evaluation -> SelectTop -> Update for G generations.
 * The inherited integration uses zero masks, reused preprocessing keys and
 * plaintext fitness responses. It is not the paper's private deployment.
 * Usage: sgfpt_client <party> <peer_ip> [host] [port] [lambda] [mu] [d]
 *                    [scale] [n_batches] [batch_size] [party_num] [gpu_id] [r_per_batch]
 * See README.md and scripts/smoke_service.py.
 */

#include <arpa/inet.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cuda_runtime.h>
#include "utils/gpu_data_types.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"
#include "Server-SGFPT/cma_config.h"
#include "Server-SGFPT/cma_state.h"
#include "Server-SGFPT/cma_mask.h"
#include "Server-SGFPT/sample.h"
#include "Server-SGFPT/select_top.h"
#include "Server-SGFPT/update.h"

__global__ void fillU64Kernel(u64* arr, int n, u64 val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

// ──────────────────────────────────────────────────────────────
// Protocol constants  (must match src/Client-Evaluation/protocol.py)
// ──────────────────────────────────────────────────────────────
static const char MAGIC[4] = {'B', '2', 'T', 'P'};

struct PromptHeader {
    char     magic[4];    // "B2TP"
    uint64_t lambda_;     // number of candidates
    uint64_t d;           // intrinsic dimension
    uint64_t bw;          // fixed-point bit-width (64)
    uint64_t scale;       // fixed-point scale     (24)
    uint64_t party_num;   // number of data-holding parties
} __attribute__((packed));  // 44 bytes, all little-endian on x86

// ──────────────────────────────────────────────────────────────
// Socket helpers
// ──────────────────────────────────────────────────────────────
static int send_all(int fd, const void *buf, size_t n)
{
    const char *p = reinterpret_cast<const char *>(buf);
    size_t sent = 0;
    while (sent < n) {
        ssize_t r = ::send(fd, p + sent, n - sent, 0);
        if (r <= 0) { perror("send"); return -1; }
        sent += static_cast<size_t>(r);
    }
    return 0;
}

static int recv_all(int fd, void *buf, size_t n)
{
    char *p = reinterpret_cast<char *>(buf);
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::recv(fd, p + got, n - got, MSG_WAITALL);
        if (r <= 0) { perror("recv"); return -1; }
        got += static_cast<size_t>(r);
    }
    return 0;
}

// ──────────────────────────────────────────────────────────────
// Fixed-point helpers  (mirror asFloat in misc_utils.h)
// ──────────────────────────────────────────────────────────────

// Encode a float as a u64 fixed-point value with the given scale.
static uint64_t float_to_fixed(double x, int scale)
{
    int64_t v = static_cast<int64_t>(x * (1LL << scale));
    return static_cast<uint64_t>(v);
}

// Decode a u64 fixed-point value back to double (available for debugging).
[[maybe_unused]] static double fixed_to_float(uint64_t x, int scale)
{
    return static_cast<double>(static_cast<int64_t>(x)) /
           static_cast<double>(1LL << scale);
}

typedef uint64_t T;
// ──────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);  // unbuffered so every printf is immediately visible
    // ── parse args ────────────────────────────────────────────────────────────
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <party> <peer_ip> [host] [port] [lambda] [mu] [d] "
            "[scale] [n_batches] [batch_size] [party_num] [gpu_id] [r_per_batch]\n", argv[0]);
        return 1;
    }
    int         party       = atoi(argv[1]);
    const char *peer_ip     = argv[2];
    const char *host        = (argc > 3)  ? argv[3]  : "127.0.0.1";
    int         port        = (argc > 4)  ? atoi(argv[4])  : 42200;
    uint64_t    lambda      = (argc > 5)  ? (uint64_t)atoi(argv[5]) : 30;
    uint64_t    mu          = (argc > 6)  ? (uint64_t)atoi(argv[6]) : 15;
    uint64_t    d           = (argc > 7)  ? (uint64_t)atoi(argv[7]) : 400;
    uint64_t    scale       = (argc > 8)  ? (uint64_t)atoi(argv[8]) : 24;
    int         n_batches   = (argc > 9)  ? atoi(argv[9])  : 39;    // default: cifar100 with batch_size=256
    int         batch_size  = (argc > 10) ? atoi(argv[10]) : 256;
    int         party_num   = (argc > 11) ? atoi(argv[11]) : 3;
    int         gpu_id      = (argc > 12) ? atoi(argv[12]) : 0;
    int         r_per_batch = (argc > 13) ? atoi(argv[13]) : 4;     // evaluation_time per batch

    // G = r_per_batch * n_batches (total evolution iterations)
    if ((party != 0 && party != 1) || lambda < 2 || mu < 1 || mu > lambda ||
        d < 4 || scale != 24 || n_batches < 1 || batch_size < 1 || party_num < 1 ||
        r_per_batch < 1 || port < 1 || port > 65535) {
        fprintf(stderr, "Invalid configuration: party=0/1, lambda>=2, 1<=mu<=lambda, d>=4, scale=24, positive batch/party counts required.\n");
        return 1;
    }
    printf("[sgfpt_client] Research simulation: zero masks and plaintext fitness.\n");
    int G = r_per_batch * n_batches;

    // ── select GPU ────────────────────────────────────────────────────────────
    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    if (gpu_id < 0 || gpu_id >= n_gpus) {
        fprintf(stderr, "[sgfpt_client] Invalid gpu_id=%d (available: 0..%d)\n",
                gpu_id, n_gpus - 1);
        return 1;
    }
    cudaSetDevice(gpu_id);
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpu_id);
        printf("[sgfpt_client] Using GPU %d: %s\n", gpu_id, prop.name);
    }

    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();

    auto peer = new GpuPeer(true);
    peer->connect(party, peer_ip);

    uint64_t bw   = 64;
    int      bin  = (int)bw;
    int      bout = (int)bw;

    CMAConfig cfg;
    cfg.d          = (int)d;
    cfg.lambda     = (int)lambda;
    cfg.mu         = (int)mu;
    cfg.scale      = (int)scale;
    cfg.ring       = (int)bw;
    cfg.sqrt_scale_sample = 18;
    cfg.sqrt_bw_sample    = 24;
    cfg.sqrt_scale_update = 14;   // can differ from sample (e.g. 16)
    cfg.sqrt_bw_update    = 24;

    cfg.exp_scale  = 22;
    cfg.exp_bw     = 25;

    CMAMask<u64>  cma_mask(cfg);
    CMAState<u64> cma_state(cfg);
    cma_mask.init();
    cma_state.init();
    spt::Sample<u64> sample;
    spt::Update<u64> update;

    u8* startPtr = nullptr;
    u8* curPtr = nullptr;
    getKeyBuf(&startPtr, &curPtr, 4 * OneGB);
    
    setZeroRandomness(true);


    spt::SelectTopParams p;
    p.lambda = lambda;
    p.mu = mu;
    p.d = d;
    p.p = party_num;
    p.scale = scale;
    p.bout = bout;
    p.n_s_ = (lambda <= 1) ? 1 : (int)ceil(log2((double)lambda));
    p.n_s_ = max(p.n_s_, 8);
    int m = 4 + scale;
    spt::SelectTop<u64> st(p);

    cma_mask.d_mask_Y = (u64*)randomGEOnGpu<u64>(p.lambda * p.d, bout);
    cma_mask.d_mask_Z = (u64*)randomGEOnGpu<u64>(p.lambda * p.d, bout);
    cma_mask.d_mask_mu_Y = (u64*)randomGEOnGpu<u64>(p.mu * p.d, bout);
    cma_mask.d_mask_mu_Z = (u64*)randomGEOnGpu<u64>(p.mu * p.d, bout);
    cma_mask.d_mask_m = (u64*)randomGEOnGpu<u64>(d, bout);
    cma_mask.d_mask_sigma = (u64*)randomGEOnGpu<u64>(d, bout);
    cma_mask.d_mask_C_diag = (u64*)randomGEOnGpu<u64>(d, bout);
    cma_mask.d_mask_p_c = (u64*)randomGEOnGpu<u64>(d, bout);
    cma_mask.d_mask_p_sigma = (u64*)randomGEOnGpu<u64>(d, bout);

    st.init();
    update.init(cma_state);
    u8* readPtr = nullptr;

    printf("[sgfpt_client] party=%d  peer=%s  server=%s:%d  gpu=%d\n"
           "               lambda=%llu  mu=%llu  d=%llu  scale=%llu\n"
           "               n_batches=%d  batch_size=%d  r_per_batch=%d  G=%d×%d=%d  party_num=%d\n",
           party, peer_ip, host, port, gpu_id,
           (unsigned long long)lambda, (unsigned long long)mu,
           (unsigned long long)d, (unsigned long long)scale,
           n_batches, batch_size, r_per_batch, r_per_batch, n_batches, G, party_num);

    // ── connect ──────────────────────────────────────────────────────────────
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return 1; }

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "Invalid address: %s\n", host);
        close(sock); return 1;
    }

    if (::connect(sock, reinterpret_cast<struct sockaddr *>(&addr),
                sizeof(addr)) < 0) {
        perror("connect"); close(sock); return 1;
    }
    printf("[sgfpt_client] Connected.\n");
    // ── Initialise CMA state (masked = plaintext when masks are zero) ─────────
    // m = 0 (mean starts at origin)
    checkCudaErrors(cudaMemset(cma_state.d_masked_m,       0, (size_t)d * sizeof(u64)));
    // p_c = 0, p_sigma = 0 (evolution paths)
    checkCudaErrors(cudaMemset(cma_state.d_masked_p_c,     0, (size_t)d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_masked_p_sigma, 0, (size_t)d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_masked_mu_Z, 0, (size_t)mu * d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_masked_mu_Y, 0, (size_t)mu * d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_masked_X, 0, (size_t)lambda * d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_masked_Y, 0, (size_t)lambda * d * sizeof(u64)));
    checkCudaErrors(cudaMemset(cma_state.d_Z, 0, (size_t)lambda * d * sizeof(u64)));
       
    setZeroRandomness(false);
    u64 * h_C_diag = (u64*)cpuMalloc(d * sizeof(u64));
    u64 * h_sigma = (u64*)cpuMalloc(d * sizeof(u64));
    u64 * h_m = (u64*)cpuMalloc(d * sizeof(u64));
    fillU64Kernel<<<(d + 255) / 256, 256>>>(cma_state.d_masked_C_diag, d, u64(1) << scale);
    fillU64Kernel<<<(d + 255) / 256, 256>>>(cma_state.d_masked_sigma, d, u64(1*(1<<scale)));
    // fillU64Kernel<<<(d + 255) / 256, 256>>>(cma_state.d_masked_m, d, u64(0 * (1 << scale)));
    peer->sync();
    
    for(int i = 0; i < G; i++)
    {
        // Simulation-only local dealer: fresh keys use this generation's Z and
        // the masks carried from the previous Update. Fitness remains unmasked.
        peer->sync();
        setZeroRandomness(true);
        curPtr = startPtr;
        auto d_mask_A = randomGEOnGpu<u64>(p.lambda, bout);
        sample.keygen(&curPtr, party, cma_mask, &g, i);
        st.keygen(&curPtr, party, bin, bout, d_mask_A, cma_mask, &g);
        update.keygen(&curPtr, party, cma_mask, &g);
        readPtr = startPtr;
        sample.readkey(&readPtr, cma_mask);
        st.readkey(&readPtr);
        update.readkey(&readPtr, cma_state);
        setZeroRandomness(false);
        auto h_sigma = (u64*)moveToCPU((u8*)cma_state.d_masked_sigma, d *  sizeof(u64), nullptr);
        
        printf("sigma[%d] = %f\n", 0, asFloat(h_sigma[0], bout,scale));
        printf("sigma[%d] = %f\n", 1, asFloat(h_sigma[1], bout,scale));
        auto h_X = (u64*)moveToCPU((u8*)cma_state.d_masked_X, lambda * d *  sizeof(u64), nullptr);
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {
                printf("%f ", asFloat(h_X[j * d + k], bw, scale));
            }
            printf("\n");
        }
        cpuFree(h_X);
        cpuFree(h_sigma);
        sample.run(peer, party, cma_state, &g, (Stats*)NULL, i);  // Pass generation index i
            // ── build + send header ───────────────────────────────────────────────────
        PromptHeader hdr{};
        memcpy(hdr.magic, MAGIC, 4);
        hdr.lambda_    = lambda;
        hdr.d          = d;
        hdr.bw         = bw;
        hdr.scale      = scale;
        hdr.party_num  = (uint64_t)party_num;
        if (send_all(sock, &hdr, sizeof(hdr)) < 0) { close(sock); return 1; }
        size_t X_bytes = lambda * d * sizeof(u64);
        u64* data = (u64*)moveToCPU((u8*)cma_state.d_masked_X, X_bytes, nullptr);
        // printf("[sgfpt_client] Sending X ...\n");
        // for(int j = 0; j < lambda; j++) {
        //     for(int k = 0; k < d; k++) {
        //         printf("%f ", asFloat(data[j * d + k], bw, scale));
        //     }
        //     printf("\n");
        // }
        if (send_all(sock, data, X_bytes) < 0) {
            cpuFree(data); close(sock); return 1;
        }
        cpuFree(data);  // cudaHostUnregister + free (moveToCPU pins memory)

        // ── receive fitness ───
        // Server returns lambda*party_num float64 values: for each candidate j
        // the DH evaluator splits loss[j] evenly into party_num parts.
        // SelectTop sums those contributions locally; no score masks are sent.
        double *fitness = new double[lambda * party_num];
        printf("[sgfpt_client] Waiting for fitness values ...\n");

        if (recv_all(sock, fitness, lambda * party_num * sizeof(double)) < 0) {
            delete[] fitness; close(sock); return 1;
        }

        // Diagnostic-only sum: the input array remains lambda*party_num elements.
        printf("[sgfpt_client] Received fitness (aggregated per candidate): ");
        for (int j = 0; j < (int)lambda; j++) {
            double agg = 0.0;
            for (int k = 0; k < party_num; k++) agg += fitness[j * party_num + k];
            printf("%f ", agg);
        }
        printf("\n");

        size_t acc_bytes = lambda * party_num * sizeof(u64);
        u64* accuracy = (u64*)cpuMalloc(acc_bytes);  // pinned by default
        for (int j = 0; j < (int)lambda; j++) {
            for (int k = 0; k < party_num; k++) {
                accuracy[j * party_num + k] = (u64)(fitness[j * party_num + k] * (1 << scale));
            }
        }
        delete[] fitness;
        u64* d_A = (u64*)moveToGPU((u8*)accuracy, acc_bytes, nullptr);
        cpuFree(accuracy);  // unregister + free pinned host buffer
        st.run(peer, party, &readPtr, bin, bout, d_A, cma_state, &g, (Stats*)NULL);
        gpuFree(d_A);
        update.run(peer, party, cma_state, &g, (Stats*)NULL);
        printf("[sgfpt_client] generation=%d complete\n", i + 1);
        fflush(stdout);
    }

    close(sock);
    return 0;
}
