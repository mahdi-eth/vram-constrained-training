#pragma once

#include "../tensor.cuh"
#include "../memory.cuh"
#include "common.cuh"
#include "block.cuh"
#include <vector>
#include <random>

namespace ct { namespace layers {

// =================== Model Configuration =================

struct ModelConfig {
    int vocab_size;
    int max_seq_len;
    int d_model;
    int n_layers;
    int n_heads;
    int ffn_dim;
};

// =================== Full Transformer =================
// Embedding -> L blocks -> final LN -> weight-tied output projection
// Pre-norm GPT-2 architecture.

struct Transformer {
    ModelConfig config;

    // ---- embeddings (persistent) ----
    Tensor tok_embed;          // (V, D)
    Tensor tok_embed_f16;      // (V, D) fp16 model copy
    Tensor pos_embed;          // (max_seq, D)
    Tensor grad_tok_embed;     // (V, D) — accumulates from output proj + embedding backward
    Tensor grad_pos_embed;     // (max_seq, D)

    // ---- transformer blocks ----
    TransformerBlock* blocks;
    int n_blocks;

    // ---- final layernorm ----
    Tensor final_ln_gamma, final_ln_beta;
    Tensor grad_final_ln_gamma, grad_final_ln_beta;

    // ---- saved for backward ----
    Tensor saved_x_final;                      // (S, D) — layernorm output for output proj grad
    Tensor saved_final_ln_mean, saved_final_ln_rstd;
    Tensor saved_x_pre_ln;                     // (S, D) — input to final LN

    // ---- flat parameter list ----
    std::vector<ParamEntry> params;
    bool mixed_precision_enabled;
    bool gradient_checkpointing_enabled;
    std::vector<uint8_t> checkpoint_mask;
    std::vector<Tensor> saved_block_inputs;

    // ---- lifecycle ----
    void create(const ModelConfig& cfg, MemoryPool& persistent, bool inference = false);
    void destroy();

    // tokens: (S,) int -> logits: (S, V) fp32
    void forward(const int* tokens, int seq_len, Tensor& logits,
                 MemoryPool& scratch, cudaStream_t s = 0);

    // targets: (S,) int. returns mean loss (scalar, sync'd to host).
    float backward(const int* tokens, const int* targets, int seq_len,
                   MemoryPool& scratch, cudaStream_t s = 0,
                   float grad_scale = 1.0f);

    // forward + CE loss only, no backward. for evaluation.
    float compute_loss(const int* tokens, const int* targets, int seq_len,
                       MemoryPool& scratch, cudaStream_t s = 0);

    void zero_grads(cudaStream_t s = 0);
    void collect_params();
    void init_weights(uint64_t seed);
    void enable_mixed_precision(bool enabled);
    void configure_gradient_checkpointing(bool enabled, const std::vector<uint8_t>& mask);
    void sync_model_weights_from_master(cudaStream_t s = 0);
};

}} // namespace ct::layers
