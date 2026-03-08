#include "ct/layers/transformer.cuh"
#include "ct/init.cuh"
#include "ct/kernels/matmul.cuh"
#include "ct/kernels/elementwise.cuh"
#include "ct/kernels/layernorm.cuh"
#include "ct/kernels/embedding.cuh"
#include "ct/kernels/cross_entropy.cuh"
#include "ct/kernels/reduce.cuh"
#include "ct/fusion_config.cuh"

#include <cmath>

namespace ct { namespace layers {

// =================== Transformer Model =================

void Transformer::create(const ModelConfig& cfg, MemoryPool& persistent, bool inference) {
    config   = cfg;
    n_blocks = cfg.n_layers;
    mixed_precision_enabled = false;
    gradient_checkpointing_enabled = false;
    checkpoint_mask.assign(n_blocks, 0);
    saved_block_inputs.resize(n_blocks);

    // ---- embeddings ----
    tok_embed = persistent.alloc(cfg.vocab_size, cfg.d_model, DType::FP32);
    pos_embed = persistent.alloc(cfg.max_seq_len, cfg.d_model, DType::FP32);

    if (!inference) {
        tok_embed_f16  = persistent.alloc(cfg.vocab_size, cfg.d_model, DType::FP16);
        grad_tok_embed = persistent.alloc(cfg.vocab_size, cfg.d_model, DType::FP32);
        grad_pos_embed = persistent.alloc(cfg.max_seq_len, cfg.d_model, DType::FP32);
    }

    // ---- blocks ----
    blocks = new TransformerBlock[n_blocks];
    for (int i = 0; i < n_blocks; i++)
        blocks[i].create(cfg.d_model, cfg.n_heads, cfg.ffn_dim, persistent, inference);

    // ---- final layernorm ----
    final_ln_gamma = persistent.alloc(cfg.d_model, DType::FP32);
    final_ln_beta  = persistent.alloc(cfg.d_model, DType::FP32);

    if (!inference) {
        grad_final_ln_gamma = persistent.alloc(cfg.d_model, DType::FP32);
        grad_final_ln_beta  = persistent.alloc(cfg.d_model, DType::FP32);
    }
}

void Transformer::destroy() {
    delete[] blocks;
    blocks = nullptr;
    n_blocks = 0;
}

// ------------------- forward -------------------
// embed -> blocks -> final_ln -> logits (weight-tied)

void Transformer::forward(const int* tokens, int seq_len, Tensor& logits,
                          MemoryPool& scratch, cudaStream_t s) {
    int D = config.d_model;
    int V = config.vocab_size;

    // ---- save tensors that must survive until backward ----
    saved_x_pre_ln       = scratch.alloc(seq_len, D, DType::FP32);
    saved_x_final        = scratch.alloc(seq_len, D, DType::FP32);
    saved_final_ln_mean  = scratch.alloc(seq_len, DType::FP32);
    saved_final_ln_rstd  = scratch.alloc(seq_len, DType::FP32);

    // NOTE: no mark/restore here — blocks and sub-layers allocate their
    // own saved tensors from scratch, which must survive until backward.
    // The caller (training loop) resets scratch between steps.

    // ---- embedding: tok + pos -> x ----
    g_profiler.begin_section();
    Tensor x = scratch.alloc(seq_len, D, DType::FP32);
    kernels::embedding_forward(tokens, tok_embed.f32(), pos_embed.f32(),
                               x.f32(), seq_len, D, s);
    g_profiler.record(&g_profiler.embed_ms, 1);

    // ---- transformer blocks ----
    for (int i = 0; i < n_blocks; i++) {
        bool checkpointed = gradient_checkpointing_enabled
                            && i < static_cast<int>(checkpoint_mask.size())
                            && checkpoint_mask[i] != 0;
        if (checkpointed) {
            saved_block_inputs[i] = scratch.alloc(seq_len, D, DType::FP32);
            tensor_copy(x, saved_block_inputs[i], s);
        }
        Tensor x_out = scratch.alloc(seq_len, D, DType::FP32);
        blocks[i].forward(x, x_out, scratch, s, !checkpointed);
        x = x_out;
    }

    // ---- save pre-LN input and apply final layernorm ----
    tensor_copy(x, saved_x_pre_ln, s);
    g_profiler.begin_section();
    kernels::layernorm_forward(x.f32(), final_ln_gamma.f32(), final_ln_beta.f32(),
                               saved_x_final.f32(),
                               saved_final_ln_mean.f32(), saved_final_ln_rstd.f32(),
                               seq_len, D, 1e-5f, s);
    g_profiler.record(&g_profiler.ln_ms, 1);

    // ---- logits = x_final @ tok_embed^T (weight tying, no bias) ----
    if (mixed_precision_enabled) {
        kernels::matmul_f32_f16(saved_x_final.f32(), tok_embed_f16.f16(), logits.f32(),
                                seq_len, V, D,
                                kernels::TransMode::NT, nullptr, s);
    } else {
        kernels::matmul_f32(saved_x_final.f32(), tok_embed.f32(), logits.f32(),
                            seq_len, V, D,
                            kernels::TransMode::NT, nullptr, s);
    }
}

// ------------------- backward -------------------
// CE backward -> output proj grad -> final LN backward -> blocks backward -> embedding backward

float Transformer::backward(const int* tokens, const int* targets, int seq_len,
                            MemoryPool& scratch, cudaStream_t s, float grad_scale) {
    int D = config.d_model;
    int V = config.vocab_size;
    float ce_grad_scale = grad_scale / static_cast<float>(seq_len);

    size_t mark = scratch.save_mark();

    // ---- recompute logits (we only saved x_final, not logits themselves) ----
    Tensor logits = scratch.alloc(seq_len, V, DType::FP32);
    if (mixed_precision_enabled) {
        kernels::matmul_f32_f16(saved_x_final.f32(), tok_embed_f16.f16(), logits.f32(),
                                seq_len, V, D,
                                kernels::TransMode::NT, nullptr, s);
    } else {
        kernels::matmul_f32(saved_x_final.f32(), tok_embed.f32(), logits.f32(),
                            seq_len, V, D,
                            kernels::TransMode::NT, nullptr, s);
    }

    // ---- compute loss (for return value) ----
    g_profiler.begin_section();
    Tensor losses = scratch.alloc(seq_len, DType::FP32);
    if (g_fusion.cross_entropy) {
        kernels::cross_entropy_forward(logits.f32(), targets, losses.f32(),
                                       seq_len, V, s);
    } else {
        // ---- unfused: log-softmax then NLL ----
        Tensor log_probs = scratch.alloc(seq_len, V, DType::FP32);
        kernels::log_softmax_forward(logits.f32(), log_probs.f32(), seq_len, V, s);
        kernels::nll_forward(log_probs.f32(), targets, losses.f32(), seq_len, V, s);
    }
    Tensor loss_sum = scratch.alloc(1, DType::FP32);
    kernels::reduce_sum(losses.f32(), loss_sum.f32(), seq_len, s);
    g_profiler.record(&g_profiler.ce_ms, g_fusion.cross_entropy ? 2 : 3);

    float loss_host = 0.0f;
    CT_CHECK_CUDA(cudaMemcpyAsync(&loss_host, loss_sum.f32(), sizeof(float),
                                  cudaMemcpyDeviceToHost, s));
    CT_CHECK_CUDA(cudaStreamSynchronize(s));
    loss_host /= static_cast<float>(seq_len);

    // ---- CE backward -> d_logits ----
    g_profiler.begin_section();
    Tensor d_logits = scratch.alloc(seq_len, V, DType::FP32);
    if (g_fusion.cross_entropy) {
        kernels::cross_entropy_backward(logits.f32(), targets, d_logits.f32(),
                                        seq_len, V, ce_grad_scale, s);
    } else {
        // ---- unfused: recompute log_probs, then backward through log-softmax ----
        Tensor log_probs = scratch.alloc(seq_len, V, DType::FP32);
        kernels::log_softmax_forward(logits.f32(), log_probs.f32(), seq_len, V, s);
        kernels::log_softmax_backward(nullptr, log_probs.f32(), targets, d_logits.f32(),
                                      seq_len, V, ce_grad_scale, s);
    }
    g_profiler.record(&g_profiler.ce_ms, g_fusion.cross_entropy ? 1 : 3);

    // ---- output projection backward (weight-tied) ----
    // dx_final = d_logits @ tok_embed
    Tensor dx_final = scratch.alloc(seq_len, D, DType::FP32);
    kernels::matmul_f32(d_logits.f32(), tok_embed.f32(), dx_final.f32(),
                        seq_len, D, V,
                        kernels::TransMode::NN, nullptr, s);

    // grad_tok_embed += d_logits^T @ x_final
    Tensor tmp_dW = scratch.alloc(V, D, DType::FP32);
    kernels::matmul_f32(d_logits.f32(), saved_x_final.f32(), tmp_dW.f32(),
                        V, D, seq_len,
                        kernels::TransMode::TN, nullptr, s);
    kernels::vec_add(grad_tok_embed.f32(), tmp_dW.f32(), grad_tok_embed.f32(),
                     V * D, s);

    // ---- final layernorm backward ----
    Tensor dx = scratch.alloc(seq_len, D, DType::FP32);
    Tensor tmp_dg = scratch.alloc(D, DType::FP32);
    Tensor tmp_db = scratch.alloc(D, DType::FP32);
    kernels::layernorm_backward(dx_final.f32(), saved_x_pre_ln.f32(),
                                final_ln_gamma.f32(),
                                saved_final_ln_mean.f32(), saved_final_ln_rstd.f32(),
                                dx.f32(), tmp_dg.f32(), tmp_db.f32(),
                                seq_len, D, s);
    kernels::vec_add(grad_final_ln_gamma.f32(), tmp_dg.f32(),
                     grad_final_ln_gamma.f32(), D, s);
    kernels::vec_add(grad_final_ln_beta.f32(), tmp_db.f32(),
                     grad_final_ln_beta.f32(), D, s);

    // ---- blocks backward (reverse order) ----
    for (int i = n_blocks - 1; i >= 0; i--) {
        bool checkpointed = gradient_checkpointing_enabled
                            && i < static_cast<int>(checkpoint_mask.size())
                            && checkpoint_mask[i] != 0;
        Tensor dx_prev = scratch.alloc(seq_len, D, DType::FP32);
        if (checkpointed) {
            blocks[i].backward_with_recompute(saved_block_inputs[i], dx, dx_prev, scratch, s);
        } else {
            blocks[i].backward(dx, dx_prev, scratch, s);
        }
        dx = dx_prev;
    }

    // ---- embedding backward: scatter-add into grad_tok_embed + grad_pos_embed ----
    // grad_tok_embed already has output proj gradient, atomicAdd accumulates on top
    kernels::embedding_backward(tokens, dx.f32(), grad_tok_embed.f32(),
                                seq_len, D, V, s);
    kernels::pos_embedding_backward(dx.f32(), grad_pos_embed.f32(),
                                    seq_len, D, s);

    scratch.restore_mark(mark);
    return loss_host;
}

// ------------------- eval-only forward + loss -------------------

float Transformer::compute_loss(const int* tokens, const int* targets, int seq_len,
                                MemoryPool& scratch, cudaStream_t s) {
    int D = config.d_model;
    int V = config.vocab_size;

    size_t mark = scratch.save_mark();

    // ---- embedding ----
    Tensor x = scratch.alloc(seq_len, D, DType::FP32);
    kernels::embedding_forward(tokens, tok_embed.f32(), pos_embed.f32(),
                               x.f32(), seq_len, D, s);

    // ---- blocks (no saved activations needed) ----
    for (int i = 0; i < n_blocks; i++) {
        Tensor x_out = scratch.alloc(seq_len, D, DType::FP32);
        blocks[i].forward(x, x_out, scratch, s, false);
        x = x_out;
    }

    // ---- final layernorm ----
    Tensor x_normed = scratch.alloc(seq_len, D, DType::FP32);
    Tensor ln_mean  = scratch.alloc(seq_len, DType::FP32);
    Tensor ln_rstd  = scratch.alloc(seq_len, DType::FP32);
    kernels::layernorm_forward(x.f32(), final_ln_gamma.f32(), final_ln_beta.f32(),
                               x_normed.f32(), ln_mean.f32(), ln_rstd.f32(),
                               seq_len, D, 1e-5f, s);

    // ---- logits = x_normed @ tok_embed^T ----
    Tensor logits = scratch.alloc(seq_len, V, DType::FP32);
    if (mixed_precision_enabled) {
        kernels::matmul_f32_f16(x_normed.f32(), tok_embed_f16.f16(), logits.f32(),
                                seq_len, V, D,
                                kernels::TransMode::NT, nullptr, s);
    } else {
        kernels::matmul_f32(x_normed.f32(), tok_embed.f32(), logits.f32(),
                            seq_len, V, D,
                            kernels::TransMode::NT, nullptr, s);
    }

    // ---- cross-entropy loss ----
    Tensor losses = scratch.alloc(seq_len, DType::FP32);
    if (g_fusion.cross_entropy) {
        kernels::cross_entropy_forward(logits.f32(), targets, losses.f32(),
                                       seq_len, V, s);
    } else {
        Tensor log_probs = scratch.alloc(seq_len, V, DType::FP32);
        kernels::log_softmax_forward(logits.f32(), log_probs.f32(), seq_len, V, s);
        kernels::nll_forward(log_probs.f32(), targets, losses.f32(), seq_len, V, s);
    }
    Tensor loss_sum = scratch.alloc(1, DType::FP32);
    kernels::reduce_sum(losses.f32(), loss_sum.f32(), seq_len, s);

    float loss_host = 0.0f;
    CT_CHECK_CUDA(cudaMemcpyAsync(&loss_host, loss_sum.f32(), sizeof(float),
                                   cudaMemcpyDeviceToHost, s));
    CT_CHECK_CUDA(cudaStreamSynchronize(s));
    loss_host /= static_cast<float>(seq_len);

    scratch.restore_mark(mark);
    return loss_host;
}

// ------------------- zero all gradient buffers -------------------

void Transformer::zero_grads(cudaStream_t s) {
    tensor_zero(grad_tok_embed, s);
    tensor_zero(grad_pos_embed, s);
    tensor_zero(grad_final_ln_gamma, s);
    tensor_zero(grad_final_ln_beta, s);

    for (int i = 0; i < n_blocks; i++) {
        auto& b = blocks[i];
        tensor_zero(b.grad_ln1_gamma, s);
        tensor_zero(b.grad_ln1_beta, s);
        tensor_zero(b.grad_ln2_gamma, s);
        tensor_zero(b.grad_ln2_beta, s);

        auto zero_linear = [&](Linear& l) {
            tensor_zero(l.grad_weight, s);
            tensor_zero(l.grad_bias, s);
        };
        zero_linear(b.attn.proj_q);
        zero_linear(b.attn.proj_k);
        zero_linear(b.attn.proj_v);
        zero_linear(b.attn.proj_o);
        zero_linear(b.ffn.fc_up);
        zero_linear(b.ffn.fc_down);
    }
}

// ------------------- collect all (weight, grad) pairs -------------------

void Transformer::collect_params() {
    params.clear();

    // ---- embeddings ----
    params.push_back(ParamEntry(tok_embed.f32(), grad_tok_embed.f32(), tok_embed.numel()));
    params.push_back(ParamEntry(pos_embed.f32(), grad_pos_embed.f32(), pos_embed.numel()));

    // ---- blocks ----
    for (int i = 0; i < n_blocks; i++)
        blocks[i].collect_params(params);

    // ---- final LN (no weight decay) ----
    params.push_back(ParamEntry(final_ln_gamma.f32(), grad_final_ln_gamma.f32(), config.d_model, false));
    params.push_back(ParamEntry(final_ln_beta.f32(), grad_final_ln_beta.f32(), config.d_model, false));
}

// ------------------- weight initialization -------------------

void Transformer::init_weights(uint64_t seed) {
    std::mt19937 rng(seed);
    int L = config.n_layers;
    float std_base = 0.02f;
    float std_residual = 0.02f / sqrtf(2.0f * L);

    fill_normal(tok_embed, rng, 0.0f, std_base);
    fill_normal(pos_embed, rng, 0.0f, std_base);

    for (int i = 0; i < n_blocks; i++) {
        auto& b = blocks[i];

        // ---- LN: gamma=1, beta=0 ----
        fill_ones(b.ln1_gamma);  fill_zeros(b.ln1_beta);
        fill_ones(b.ln2_gamma);  fill_zeros(b.ln2_beta);

        // ---- attention projections ----
        fill_normal(b.attn.proj_q.weight, rng, 0.0f, std_base);
        fill_normal(b.attn.proj_k.weight, rng, 0.0f, std_base);
        fill_normal(b.attn.proj_v.weight, rng, 0.0f, std_base);
        fill_normal(b.attn.proj_o.weight, rng, 0.0f, std_residual); // residual path

        // ---- FFN ----
        fill_normal(b.ffn.fc_up.weight, rng, 0.0f, std_base);
        fill_normal(b.ffn.fc_down.weight, rng, 0.0f, std_residual); // residual path

        // ---- all biases to zero ----
        fill_zeros(b.attn.proj_q.bias);
        fill_zeros(b.attn.proj_k.bias);
        fill_zeros(b.attn.proj_v.bias);
        fill_zeros(b.attn.proj_o.bias);
        fill_zeros(b.ffn.fc_up.bias);
        fill_zeros(b.ffn.fc_down.bias);
    }

    fill_ones(final_ln_gamma);
    fill_zeros(final_ln_beta);

    sync_model_weights_from_master();

    // ---- pack QKV weights for fused projection ----
    if (g_fusion.qkv_proj) {
        for (int i = 0; i < n_blocks; i++)
            blocks[i].attn.sync_qkv_weights();
    }
}

void Transformer::enable_mixed_precision(bool enabled) {
    mixed_precision_enabled = enabled;
    for (int i = 0; i < n_blocks; i++) {
        auto& b = blocks[i];
        b.attn.proj_q.set_mixed_forward(enabled);
        b.attn.proj_k.set_mixed_forward(enabled);
        b.attn.proj_v.set_mixed_forward(enabled);
        b.attn.proj_o.set_mixed_forward(enabled);
        b.ffn.fc_up.set_mixed_forward(enabled);
        b.ffn.fc_down.set_mixed_forward(enabled);
    }
}

void Transformer::configure_gradient_checkpointing(bool enabled, const std::vector<uint8_t>& mask) {
    gradient_checkpointing_enabled = enabled;
    checkpoint_mask.assign(n_blocks, enabled ? 1u : 0u);
    if (enabled && !mask.empty()) {
        for (int i = 0; i < n_blocks && i < static_cast<int>(mask.size()); i++) {
            checkpoint_mask[i] = mask[i] ? 1u : 0u;
        }
    }
}

void Transformer::sync_model_weights_from_master(cudaStream_t s) {
    tensor_cast_f32_to_f16(tok_embed, tok_embed_f16, s);
    for (int i = 0; i < n_blocks; i++) {
        auto& b = blocks[i];
        b.attn.proj_q.sync_model_weight(s);
        b.attn.proj_k.sync_model_weight(s);
        b.attn.proj_v.sync_model_weight(s);
        b.attn.proj_o.sync_model_weight(s);
        b.ffn.fc_up.sync_model_weight(s);
        b.ffn.fc_down.sync_model_weight(s);

        // ---- keep QKV packed weights in sync after optimizer step ----
        if (g_fusion.qkv_proj)
            b.attn.sync_qkv_weights();
    }
}

}} // namespace ct::layers
