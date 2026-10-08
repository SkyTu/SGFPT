#pragma once
#include "Server-SGFPT/cma_config.h"
#include <cstdio>
template <typename T = u64>
class CMAState {
public:
    T* d_masked_m      = nullptr;
    T* d_masked_sigma  = nullptr;
    T* d_masked_C_diag = nullptr;
    T* d_w             = nullptr;
    T* d_csigma2_w     = nullptr;  // c_sigma2 * w[i], precomputed plaintext
    T* d_cmu_w         = nullptr;  // c_mu * w[i], precomputed plaintext
    T* d_masked_p_c    = nullptr;
    T* d_masked_p_sigma = nullptr;
    T* d_masked_mu_Y   = nullptr;
    T* d_masked_mu_Z   = nullptr;
    T* d_masked_Y      = nullptr;
    T* d_masked_X      = nullptr;
    T* d_Z      = nullptr;
    int d;
    int lambda;
    int mu;
    int scale;
    int sqrt_scale_sample;
    int sqrt_bw_sample;
    int sqrt_scale_update;
    int sqrt_bw_update;
    int exp_scale;
    int exp_bw;
    int ring;
    T c_sigma1, c_sigma2, c_c1, c_c2, c_1, c_mu, c;
    T b_sigma1, b_sigma2;
    


    CMAState(const CMAConfig& cfg) : d(cfg.d), lambda(cfg.lambda), mu(cfg.mu), scale(cfg.scale), ring(cfg.ring), sqrt_scale_sample(cfg.sqrt_scale_sample), sqrt_bw_sample(cfg.sqrt_bw_sample), sqrt_scale_update(cfg.sqrt_scale_update), sqrt_bw_update(cfg.sqrt_bw_update), exp_scale(cfg.exp_scale), exp_bw(cfg.exp_bw) {
        d_masked_m     = (T*)gpuMalloc((size_t)d * sizeof(T));
        d_masked_sigma = (T*)gpuMalloc((size_t)d * sizeof(T));
        d_masked_C_diag  = (T*)gpuMalloc((size_t)d * sizeof(T));
        d_masked_p_c = (T*)gpuMalloc((size_t)d * sizeof(T));
        d_masked_p_sigma = (T*)gpuMalloc((size_t)d * sizeof(T));
        d_masked_mu_Y = (T*)gpuMalloc((size_t)mu * d * sizeof(T));
        d_masked_mu_Z = (T*)gpuMalloc((size_t)mu * d * sizeof(T));
        d_masked_Y = (T*)gpuMalloc((size_t)lambda * d * sizeof(T));
        d_masked_X = (T*)gpuMalloc((size_t)lambda * d * sizeof(T));
        d_Z = (T*)gpuMalloc((size_t)lambda * d * sizeof(T));
        d_w          = (T*)gpuMalloc((size_t)mu * sizeof(T));
        d_csigma2_w  = (T*)gpuMalloc((size_t)mu * sizeof(T));
        d_cmu_w      = (T*)gpuMalloc((size_t)mu * sizeof(T));
    }

    ~CMAState() {
        gpuFree(d_masked_m);
        gpuFree(d_masked_sigma);
        gpuFree(d_masked_C_diag);
        gpuFree(d_masked_p_c);
        gpuFree(d_masked_p_sigma);
        gpuFree(d_masked_mu_Y);
        gpuFree(d_masked_mu_Z);
        gpuFree(d_masked_Y);
        gpuFree(d_Z);
        gpuFree(d_w);
        gpuFree(d_csigma2_w);
        gpuFree(d_cmu_w);
    }

    void init() {
        // Step 1: compute w'[i] and normalized w[i]
        double sumw = 0, sum_w_sq = 0;
        double* w_norm = new double[mu];
        for (int i = 0; i < mu; i++) {
            w_norm[i] = std::log((double)mu + 0.5) - std::log((double)(i + 1));
            sumw += w_norm[i];
        }
        for (int i = 0; i < mu; i++) {
            w_norm[i] /= sumw;
            sum_w_sq += w_norm[i] * w_norm[i];
        }

        double delta_eff = 1.0 / sum_w_sq;

        // Step 2: compute c_sigma2 as double (needs delta_eff first)
        double c_sigma = (delta_eff + 2.0) / (d + delta_eff + 5.0);
        double c_sigma2_d = std::sqrt(c_sigma * (2.0 - c_sigma) * delta_eff);
        
        // Step 3: fill d_w = w[i] and d_csigma2_w = c_sigma2 * w[i]
        const double fp = (double)(1 << scale);
        T* h_w         = new T[mu];
        T* h_csigma2_w = new T[mu];
        T* h_cmu_w     = new T[mu];
        
        // rank-mu covariance learning rate (same formula used for the decay c below)
        double fc_1_tmp  = 2.0 / ((d + 1.3) * (d + 1.3) + delta_eff);
        double fc_mu_tmp = std::min(1.0 - fc_1_tmp,
                               2.0 * (delta_eff - 2.0 + 1.0 / delta_eff)
                               / ((d + 2.0) * (d + 2.0) + delta_eff));
        
        // First pass: compute quantized weights
        for (int i = 0; i < mu; i++) {
            h_w[i]         = T(w_norm[i]                * fp);
            h_csigma2_w[i] = T(c_sigma2_d * w_norm[i]   * fp);
            h_cmu_w[i]     = T(fc_mu_tmp  * w_norm[i]   * fp);
        }
        
        // Compensate for fixed-point quantization to ensure Σw ≈ 1
        double fp_w_sum = 0.0;
        for (int i = 0; i < mu; i++) {
            fp_w_sum += (double)h_w[i] / fp;
        }
        double w_error = 1.0 - fp_w_sum;
        
        // Apply correction to first weight
        h_w[0] = T((w_norm[0] + w_error) * fp);
        h_csigma2_w[0] = T(c_sigma2_d * (w_norm[0] + w_error) * fp);
        h_cmu_w[0] = T(fc_mu_tmp * (w_norm[0] + w_error) * fp);
        
        // Verify sum
        double final_sum = 0.0;
        for (int i = 0; i < mu; i++) {
            final_sum += (double)h_w[i] / fp;
        }
        printf("[CMA-ES] Weight normalization check: Σw = %.10f (error = %.2e)\n", final_sum, 1.0 - final_sum);
        double w0 = w_norm[0], w_last = w_norm[mu - 1];
        delete[] w_norm;
        checkCudaErrors(cudaMemcpy(d_w,         h_w,         (size_t)mu * sizeof(T), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_csigma2_w, h_csigma2_w, (size_t)mu * sizeof(T), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_cmu_w,     h_cmu_w,     (size_t)mu * sizeof(T), cudaMemcpyHostToDevice));
        delete[] h_w;
        delete[] h_cmu_w;
        delete[] h_csigma2_w;

        // Step-size path learning rate (c_sigma already computed above)
        c_sigma1 = T((1.0 - c_sigma)  * fp);
        c_sigma2 = T(c_sigma2_d        * fp);

        // Covariance path learning rate
        double c_c = (4.0 + delta_eff / d) / (d + 4.0 + 2.0 * delta_eff / d);
        c_c1 = T((1.0 - c_c)                               * fp);
        c_c2 = T(std::sqrt(c_c * (2.0 - c_c) * delta_eff) * fp);

        // Rank-one and rank-mu learning rates
        double fc_1  = 2.0 / ((d + 1.3) * (d + 1.3) + delta_eff);
        double fc_mu = std::min(1.0 - fc_1,
                                2.0 * (delta_eff - 2.0 + 1.0 / delta_eff) / ((d + 2.0) * (d + 2.0) + delta_eff));
        c_1  = T(fc_1  * fp);
        c_mu = T(fc_mu * fp);
        c = T((1 - fc_1 - fc_mu) * fp);
        // Step-size update coefficients: sigma <- sigma * exp(b1*||p_sigma|| - b2), 存 -b2 以便 linearComb 做 d_sqrt + b_sigma2 = d_sqrt - b2
        double v_sigma = 1.0 + c_sigma;
        double E_N = std::sqrt((double)d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d));
        double b_sigma1_d = c_sigma / (v_sigma * E_N);
        double b_sigma2_pos = c_sigma / v_sigma;
        b_sigma1 = T(b_sigma1_d * fp);
        {
            b_sigma2 = (uint64_t)(b_sigma2_pos * fp);
            b_sigma2 = (T(0) - b_sigma2);
        }

        // ----- 打印所有 CMA-ES 参数（与 pycma 对照用）-----
        printf("[CMA-ES params] d=%d lambda=%d mu=%d scale=%d fp=%.0f\n", d, lambda, mu, scale, fp);
        printf("  delta_eff = %.10f   (1/sum(w^2))\n", delta_eff);
        printf("  c_sigma   = %.10f   (step-size path lr)\n", c_sigma);
        printf("  c_sigma1  = %.10f   (1-c_sigma, fp=%lu)\n", (1.0 - c_sigma), (unsigned long)c_sigma1);
        printf("  c_sigma2  = %.10f   (sqrt(c_sigma*(2-c_sigma)*delta_eff), fp=%lu)\n", c_sigma2_d, (unsigned long)c_sigma2);
        printf("  c_c       = %.10f   (cov path lr)\n", c_c);
        printf("  c_c1      = %.10f   (1-c_c)\n", (1.0 - c_c));
        printf("  c_c2      = %.10f   (sqrt(c_c*(2-c_c)*delta_eff))\n", std::sqrt(c_c * (2.0 - c_c) * delta_eff));
        printf("  c_1       = %.10f   (rank-one lr)\n", fc_1);
        printf("  c_mu      = %.10f   (rank-mu lr)\n", fc_mu);
        printf("  c         = %.10f   (1-c_1-c_mu, decay)\n", 1.0 - fc_1 - fc_mu);
        printf("  v_sigma   = %.10f   (1+c_sigma)\n", v_sigma);
        printf("  E_N       = %.10f   (E||N(0,I)||)\n", E_N);
        printf("  b_sigma1  = %.10f   (c_sigma/(v_sigma*E_N), sigma update coeff)\n", b_sigma1_d);
        printf("  b_sigma2  = %.10f   (c_sigma/v_sigma, 存为环上 -b2)\n", b_sigma2_pos);
        printf("  w_norm[0]=%.6f w_norm[mu-1]=%.6f\n", w0, w_last);
        printf("[CMA-ES params] ----- end -----\n");
    }
};


