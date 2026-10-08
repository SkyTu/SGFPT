#pragma once

struct CMAConfig {
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
};
