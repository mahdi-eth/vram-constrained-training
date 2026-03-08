#pragma once

#include "../tensor.cuh"
#include "../memory.cuh"
#include "common.cuh"
#include "attention.cuh"
#include "ffn.cuh"
#include <vector>

namespace ct { namespace layers {

// =================== Transformer Block (Pre-Norm) =================
// x = x + Attention(LayerNorm(x))
// x = x + FFN(LayerNorm(x))

struct TransformerBlock {
    Attention attn;
    FFN       ffn;

    // ---- layernorm params (persistent) ----
    Tensor ln1_gamma, ln1_beta;
    Tensor ln2_gamma, ln2_beta;
    Tensor grad_ln1_gamma, grad_ln1_beta;
    Tensor grad_ln2_gamma, grad_ln2_beta;

    // ---- saved for backward (allocated from scratch) ----
    Tensor saved_x_in;              // (S, D) — block input
    Tensor saved_x_mid;             // (S, D) — after attention residual
    Tensor saved_x_ln1, saved_x_ln2; // (S, D) — layernorm outputs
    Tensor saved_ln1_mean, saved_ln1_rstd;
    Tensor saved_ln2_mean, saved_ln2_rstd;

    int d_model;

    void create(int d_model, int n_heads, int ffn_dim, MemoryPool& persistent, bool inference = false);

    // x_in: (S, D), x_out: (S, D)
    void forward(const Tensor& x_in, Tensor& x_out,
                 MemoryPool& scratch, cudaStream_t s = 0,
                 bool save_for_backward = true);

    // dx_out: (S, D) -> dx_in: (S, D)
    void backward(const Tensor& dx_out, Tensor& dx_in,
                  MemoryPool& scratch, cudaStream_t s = 0);

    // recompute internals from x_in and then run backward
    void backward_with_recompute(const Tensor& x_in, const Tensor& dx_out, Tensor& dx_in,
                                 MemoryPool& scratch, cudaStream_t s = 0);

    void collect_params(std::vector<ParamEntry>& params);
};

}} // namespace ct::layers
