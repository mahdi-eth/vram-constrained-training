#include "ct/layers/block.cuh"
#include "ct/kernels/layernorm.cuh"
#include "ct/kernels/elementwise.cuh"
#include "ct/fusion_config.cuh"

namespace ct { namespace layers {

// =================== Transformer Block =================

void TransformerBlock::create(int d_model_, int n_heads, int ffn_dim,
                              MemoryPool& persistent, bool inference) {
    d_model = d_model_;
    attn.create(d_model, n_heads, persistent, inference);
    ffn.create(d_model, ffn_dim, persistent, inference);

    ln1_gamma = persistent.alloc(d_model, DType::FP32);
    ln1_beta  = persistent.alloc(d_model, DType::FP32);
    ln2_gamma = persistent.alloc(d_model, DType::FP32);
    ln2_beta  = persistent.alloc(d_model, DType::FP32);

    if (!inference) {
        grad_ln1_gamma = persistent.alloc(d_model, DType::FP32);
        grad_ln1_beta  = persistent.alloc(d_model, DType::FP32);
        grad_ln2_gamma = persistent.alloc(d_model, DType::FP32);
        grad_ln2_beta  = persistent.alloc(d_model, DType::FP32);
    }
}

// ------------------- forward -------------------
// Pre-norm: LN1 -> Attn -> residual, LN2 -> FFN -> residual

void TransformerBlock::forward(const Tensor& x_in, Tensor& x_out,
                               MemoryPool& scratch, cudaStream_t s,
                               bool save_for_backward) {
    int seq = x_in.dim(0);

    if (save_for_backward) {
        saved_x_in     = scratch.alloc(seq, d_model, DType::FP32);
        saved_x_mid    = scratch.alloc(seq, d_model, DType::FP32);
        saved_x_ln1    = scratch.alloc(seq, d_model, DType::FP32);
        saved_x_ln2    = scratch.alloc(seq, d_model, DType::FP32);
        saved_ln1_mean = scratch.alloc(seq, DType::FP32);
        saved_ln1_rstd = scratch.alloc(seq, DType::FP32);
        saved_ln2_mean = scratch.alloc(seq, DType::FP32);
        saved_ln2_rstd = scratch.alloc(seq, DType::FP32);
    } else {
        saved_x_in = Tensor{};
        saved_x_mid = Tensor{};
        saved_x_ln1 = Tensor{};
        saved_x_ln2 = Tensor{};
        saved_ln1_mean = Tensor{};
        saved_ln1_rstd = Tensor{};
        saved_ln2_mean = Tensor{};
        saved_ln2_rstd = Tensor{};
    }

    // ---- save input ----
    Tensor x_in_store = save_for_backward ? saved_x_in : scratch.alloc(seq, d_model, DType::FP32);
    tensor_copy(x_in, x_in_store, s);

    // ---- LN1 ----
    g_profiler.begin_section();
    Tensor x_ln1 = save_for_backward ? saved_x_ln1 : scratch.alloc(seq, d_model, DType::FP32);
    Tensor ln1_mean = save_for_backward ? saved_ln1_mean : scratch.alloc(seq, DType::FP32);
    Tensor ln1_rstd = save_for_backward ? saved_ln1_rstd : scratch.alloc(seq, DType::FP32);
    kernels::layernorm_forward(x_in.f32(), ln1_gamma.f32(), ln1_beta.f32(),
                               x_ln1.f32(), ln1_mean.f32(), ln1_rstd.f32(),
                               seq, d_model, 1e-5f, s);
    g_profiler.record(&g_profiler.ln_ms, 1);

    // ---- attention ----
    Tensor attn_out = scratch.alloc(seq, d_model, DType::FP32);
    attn.forward(x_ln1, attn_out, scratch, s);

    // ---- residual 1 + LN2 ----
    Tensor x_mid = save_for_backward ? saved_x_mid : scratch.alloc(seq, d_model, DType::FP32);
    Tensor x_ln2 = save_for_backward ? saved_x_ln2 : scratch.alloc(seq, d_model, DType::FP32);
    Tensor ln2_mean = save_for_backward ? saved_ln2_mean : scratch.alloc(seq, DType::FP32);
    Tensor ln2_rstd = save_for_backward ? saved_ln2_rstd : scratch.alloc(seq, DType::FP32);

    if (g_fusion.residual_ln) {
        // ---- fused: residual add + layernorm in one kernel ----
        g_profiler.begin_section();
        kernels::residual_layernorm_forward(x_in.f32(), attn_out.f32(),
                                            ln2_gamma.f32(), ln2_beta.f32(),
                                            x_mid.f32(), x_ln2.f32(),
                                            ln2_mean.f32(), ln2_rstd.f32(),
                                            seq, d_model, 1e-5f, s);
        g_profiler.record(&g_profiler.residual_ms, 1);
    } else {
        // ---- unfused: separate residual add then layernorm ----
        g_profiler.begin_section();
        kernels::residual_add(x_in.f32(), attn_out.f32(), x_mid.f32(),
                              seq * d_model, s);
        g_profiler.record(&g_profiler.residual_ms, 1);

        g_profiler.begin_section();
        kernels::layernorm_forward(x_mid.f32(), ln2_gamma.f32(), ln2_beta.f32(),
                                   x_ln2.f32(), ln2_mean.f32(), ln2_rstd.f32(),
                                   seq, d_model, 1e-5f, s);
        g_profiler.record(&g_profiler.ln_ms, 1);
    }

    // ---- FFN ----
    Tensor ffn_out = scratch.alloc(seq, d_model, DType::FP32);
    ffn.forward(x_ln2, ffn_out, scratch, s);

    // ---- residual 2 ----
    g_profiler.begin_section();
    kernels::residual_add(x_mid.f32(), ffn_out.f32(), x_out.f32(),
                          seq * d_model, s);
    g_profiler.record(&g_profiler.residual_ms, 1);
}

// ------------------- backward -------------------

void TransformerBlock::backward(const Tensor& dx_out, Tensor& dx_in,
                                MemoryPool& scratch, cudaStream_t s) {
    int seq = dx_out.dim(0);
    int n = seq * d_model;

    // ---- FFN arm backward ----
    Tensor d_x_ln2 = scratch.alloc(seq, d_model, DType::FP32);
    ffn.backward(dx_out, saved_x_ln2, d_x_ln2, scratch, s);

    // ---- LN2 backward ----
    g_profiler.begin_section();
    Tensor d_x_mid_from_ln = scratch.alloc(seq, d_model, DType::FP32);
    Tensor tmp_dg2 = scratch.alloc(d_model, DType::FP32);
    Tensor tmp_db2 = scratch.alloc(d_model, DType::FP32);
    kernels::layernorm_backward(d_x_ln2.f32(), saved_x_mid.f32(),
                                ln2_gamma.f32(), saved_ln2_mean.f32(), saved_ln2_rstd.f32(),
                                d_x_mid_from_ln.f32(), tmp_dg2.f32(), tmp_db2.f32(),
                                seq, d_model, s);
    kernels::vec_add(grad_ln2_gamma.f32(), tmp_dg2.f32(), grad_ln2_gamma.f32(), d_model, s);
    kernels::vec_add(grad_ln2_beta.f32(), tmp_db2.f32(), grad_ln2_beta.f32(), d_model, s);
    g_profiler.record(&g_profiler.ln_ms, 1);

    // ---- residual junction ----
    g_profiler.begin_section();
    Tensor dx_mid = scratch.alloc(seq, d_model, DType::FP32);
    kernels::vec_add(dx_out.f32(), d_x_mid_from_ln.f32(), dx_mid.f32(), n, s);
    g_profiler.record(&g_profiler.residual_ms, 1);

    // ---- attention arm backward ----
    Tensor d_x_ln1 = scratch.alloc(seq, d_model, DType::FP32);
    attn.backward(dx_mid, saved_x_ln1, d_x_ln1, scratch, s);

    // ---- LN1 backward ----
    g_profiler.begin_section();
    Tensor d_x_in_from_ln = scratch.alloc(seq, d_model, DType::FP32);
    Tensor tmp_dg1 = scratch.alloc(d_model, DType::FP32);
    Tensor tmp_db1 = scratch.alloc(d_model, DType::FP32);
    kernels::layernorm_backward(d_x_ln1.f32(), saved_x_in.f32(),
                                ln1_gamma.f32(), saved_ln1_mean.f32(), saved_ln1_rstd.f32(),
                                d_x_in_from_ln.f32(), tmp_dg1.f32(), tmp_db1.f32(),
                                seq, d_model, s);
    kernels::vec_add(grad_ln1_gamma.f32(), tmp_dg1.f32(), grad_ln1_gamma.f32(), d_model, s);
    kernels::vec_add(grad_ln1_beta.f32(), tmp_db1.f32(), grad_ln1_beta.f32(), d_model, s);
    g_profiler.record(&g_profiler.ln_ms, 1);

    // ---- residual junction ----
    g_profiler.begin_section();
    kernels::vec_add(dx_mid.f32(), d_x_in_from_ln.f32(), dx_in.f32(), n, s);
    g_profiler.record(&g_profiler.residual_ms, 1);
}

void TransformerBlock::backward_with_recompute(const Tensor& x_in, const Tensor& dx_out, Tensor& dx_in,
                                               MemoryPool& scratch, cudaStream_t s) {
    int seq = x_in.dim(0);
    size_t mark = scratch.save_mark();

    Tensor x_out_tmp = scratch.alloc(seq, d_model, DType::FP32);
    forward(x_in, x_out_tmp, scratch, s, true);
    backward(dx_out, dx_in, scratch, s);

    scratch.restore_mark(mark);
}

// ------------------- param collection -------------------

static void push_linear(Linear& l, std::vector<ParamEntry>& params) {
    params.push_back(ParamEntry(l.weight.f32(), l.grad_weight.f32(), l.weight.numel(), true));
    params.push_back(ParamEntry(l.bias.f32(), l.grad_bias.f32(), l.bias.numel(), false));
}

void TransformerBlock::collect_params(std::vector<ParamEntry>& params) {
    params.push_back(ParamEntry(ln1_gamma.f32(), grad_ln1_gamma.f32(), d_model, false));
    params.push_back(ParamEntry(ln1_beta.f32(), grad_ln1_beta.f32(), d_model, false));
    params.push_back(ParamEntry(ln2_gamma.f32(), grad_ln2_gamma.f32(), d_model, false));
    params.push_back(ParamEntry(ln2_beta.f32(), grad_ln2_beta.f32(), d_model, false));

    push_linear(attn.proj_q, params);
    push_linear(attn.proj_k, params);
    push_linear(attn.proj_v, params);
    push_linear(attn.proj_o, params);

    push_linear(ffn.fc_up, params);
    push_linear(ffn.fc_down, params);
}

}} // namespace ct::layers
