#pragma once

#include "../layers/transformer.cuh"
#include "../optim/adamw.cuh"
#include "../optim/lr_schedule.cuh"
#include "data_loader.cuh"
#include "diagnostics.cuh"

#include <cstdio>
#include <random>

namespace ct { namespace training {

// =================== Training Configuration =================

struct TrainConfig {
    layers::ModelConfig model;
    optim::AdamWConfig  optim;

    int seq_len;                // 256
    int grad_accum_steps;       // 32
    int max_steps;              // total optimizer steps
    int eval_interval;          // 500
    int checkpoint_interval;    // 2000
    int log_interval;           // 10
    int sample_interval;        // 1000
    int sample_len;             // 64 tokens
    float sample_temperature;   // 0 = greedy, >0 = temperature sampling
    bool gradient_checkpointing; // activation checkpointing toggle
    const char* checkpoint_layers; // all|none|csv indices

    float peak_lr;              // 3e-4
    float min_lr;               // 1e-5
    int   warmup_steps;         // 500

    bool  mixed_precision;      // enable mixed-precision training controls
    float loss_scale_init;      // initial dynamic loss scale
    float loss_scale_growth;    // growth factor after stable interval
    float loss_scale_backoff;   // backoff factor on overflow
    int   loss_scale_interval;  // successful steps before growth

    // diagnostics
    bool diag_enabled;
    int  diag_tier2_interval;
    int  diag_tier3_interval;
    int  diag_tier4_interval;
    int  diag_hist_bins;
    int  diag_attn_layers;
    int  diag_attn_heads;
    int  diag_attn_queries;
    bool diag_async_io;
    const char* diag_log_dir;

    const char* train_path;     // CTRF binary
    const char* val_path;       // CTRF binary (can be null)
    const char* checkpoint_dir; // where to save checkpoints
    const char* log_path;       // CSV log file

    size_t persistent_pool_mb;  // default 1200
    size_t scratch_pool_mb;     // default 300

    bool benchmark;             // CUDA event timing for research benchmarks

    uint64_t seed;
};

// =================== Trainer =================

struct Trainer {
    layers::Transformer  model;
    optim::AdamW         optimizer;
    optim::LRSchedule    lr_schedule;
    DataLoader           train_data;
    DataLoader           val_data;
    bool                 has_val;

    MemoryPool persistent;
    MemoryPool scratch;

    TrainConfig config;
    int current_step;
    FILE* log_file;
    float last_train_loss;
    float last_val_loss;
    float loss_scale;
    int   good_steps_since_scale;
    int   last_overflow_step;
    std::mt19937_64 sample_rng;
    Diagnostics diagnostics;

    // ---- benchmark timing (research instrumentation)
    cudaEvent_t ev_fwd_start, ev_fwd_end;
    cudaEvent_t ev_bwd_start, ev_bwd_end;
    cudaEvent_t ev_opt_start, ev_opt_end;
    float last_fwd_ms, last_bwd_ms, last_optim_ms;
    float last_vram_real_mb;

    void create(const TrainConfig& cfg);
    void destroy();
    void train();

private:
    float train_step(float& out_grad_norm, bool& out_overflow);
    float evaluate(int num_batches);
    void  run_diagnostics_snapshot();
    void  log_step(int step, float loss, float val_loss, float lr, float gnorm,
                   double elapsed_s, double tok_per_sec, float gpu_temp_c,
                   float cur_loss_scale, bool overflow);
    void  save_checkpoint();
    void  sample_text();
};

}} // namespace ct::training
