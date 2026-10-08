#pragma once
#include "Server-SGFPT/cma_config.h"
template <typename T = u64>
class CMAMask {
public:
    T* d_mask_m      = nullptr;
    T* d_mask_sigma  = nullptr;
    T* d_mask_C_diag = nullptr;
    T* d_w           = nullptr;
    T* d_csigma2_w   = nullptr;  // c_sigma2 * w[i], precomputed plaintext
    T* d_cmu_w       = nullptr;  // c_mu * w[i], precomputed plaintext
    T* d_mask_p_c    = nullptr;
    T* d_mask_p_sigma = nullptr;
    T* d_mask_mu_Y   = nullptr;
    T* d_mask_mu_Z   = nullptr;
    T* d_mask_Y      = nullptr;
    T* d_mask_X      = nullptr;
    T* d_mask_Z      = nullptr;
    int d;
    int lambda;
    int mu;
    int scale;
    int ring;
    int sqrt_scale_sample;
    int sqrt_bw_sample;
    int sqrt_scale_update;
    int sqrt_bw_update;
    int exp_scale;
    int exp_bw;
    T c_sigma1, c_sigma2, c_c1, c_c2, c_1, c_mu, c;
    T b_sigma1, b_sigma2;


    CMAMask(const CMAConfig& cfg) : d(cfg.d), lambda(cfg.lambda), mu(cfg.mu), scale(cfg.scale), ring(cfg.ring), sqrt_scale_sample(cfg.sqrt_scale_sample), sqrt_bw_sample(cfg.sqrt_bw_sample), sqrt_scale_update(cfg.sqrt_scale_update), sqrt_bw_update(cfg.sqrt_bw_update), exp_scale(cfg.exp_scale), exp_bw(cfg.exp_bw) {
        d_mask_m      = (T*)gpuMalloc((size_t)d * sizeof(T));
        d_mask_sigma  = (T*)gpuMalloc((size_t)d * sizeof(T));
        d_mask_C_diag = (T*)gpuMalloc((size_t)d * sizeof(T));
        d_mask_p_c    = (T*)gpuMalloc((size_t)d * sizeof(T));
        d_mask_p_sigma = (T*)gpuMalloc((size_t)d * sizeof(T));
        d_mask_mu_Y   = (T*)gpuMalloc((size_t)mu * d * sizeof(T));
        d_mask_mu_Z   = (T*)gpuMalloc((size_t)mu * d * sizeof(T));
        d_mask_Y      = (T*)gpuMalloc((size_t)lambda * d * sizeof(T));
        d_mask_X      = (T*)gpuMalloc((size_t)lambda * d * sizeof(T));
        d_mask_Z      = (T*)gpuMalloc((size_t)lambda * d * sizeof(T));
        d_w           = (T*)gpuMalloc((size_t)mu * sizeof(T));
        d_csigma2_w   = (T*)gpuMalloc((size_t)mu * sizeof(T));
        d_cmu_w       = (T*)gpuMalloc((size_t)mu * sizeof(T));
    }

    ~CMAMask() {
        gpuFree(d_mask_m);
        gpuFree(d_mask_sigma);
        gpuFree(d_mask_C_diag);
        gpuFree(d_mask_p_c);
        gpuFree(d_mask_p_sigma);
        gpuFree(d_mask_mu_Y);
        gpuFree(d_mask_mu_Z);
        gpuFree(d_w);
        gpuFree(d_csigma2_w);
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
        T* h_w          = new T[mu];
        T* h_csigma2_w  = new T[mu];
        T* h_cmu_w      = new T[mu];
        double fc_1_tmp  = 2.0 / ((d + 1.3) * (d + 1.3) + delta_eff);
        double fc_mu_tmp = std::min(1.0 - fc_1_tmp,
                               2.0 * (delta_eff - 2.0 + 1.0 / delta_eff)
                               / ((d + 2.0) * (d + 2.0) + delta_eff));
        for (int i = 0; i < mu; i++) {
            h_w[i]         = T(w_norm[i]               * fp);
            h_csigma2_w[i] = T(c_sigma2_d * w_norm[i]  * fp);
            h_cmu_w[i]     = T(fc_mu_tmp  * w_norm[i]  * fp);
        }
        // Match CMAState's fixed-point correction exactly: a one-bit coefficient
        // difference multiplies a full-width random mask during preprocessing.
        double fp_w_sum = 0.0;
        for (int i = 0; i < mu; ++i) fp_w_sum += (double)h_w[i] / fp;
        double w_error = 1.0 - fp_w_sum;
        h_w[0] = T((w_norm[0] + w_error) * fp);
        h_csigma2_w[0] = T(c_sigma2_d * (w_norm[0] + w_error) * fp);
        h_cmu_w[0] = T(fc_mu_tmp * (w_norm[0] + w_error) * fp);
        delete[] w_norm;
        checkCudaErrors(cudaMemcpy(d_w,         h_w,         (size_t)mu * sizeof(T), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_csigma2_w, h_csigma2_w, (size_t)mu * sizeof(T), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_cmu_w,     h_cmu_w,     (size_t)mu * sizeof(T), cudaMemcpyHostToDevice));
        delete[] h_w;
        delete[] h_csigma2_w;
        delete[] h_cmu_w;
        // Step-size path learning rate (c_sigma already computed above)
        c_sigma1 = T((1.0 - c_sigma) * fp);
        c_sigma2 = T(c_sigma2_d      * fp);

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
        b_sigma1 = T(c_sigma / (v_sigma * E_N) * fp);
        {
            uint64_t b2_pos = (uint64_t)(c_sigma / v_sigma * fp);
            b_sigma2 = (T(0) - T(b2_pos));
        }
    }
};
