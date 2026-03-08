#include "ct/training/trainer.cuh"
#include "ct/training/checkpoint.cuh"
#include "ct/kernels/reduce.cuh"
#include "ct/fusion_config.cuh"

#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <sstream>

#ifdef _WIN32
#include <direct.h>
#define ct_mkdir(d) _mkdir(d)
#else
#include <sys/stat.h>
#define ct_mkdir(d) mkdir(d, 0755)
#endif

namespace ct { namespace training {

static float try_read_gpu_temp() {
#ifdef _WIN32
    FILE* pipe = _popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>nul", "r");
#else
    FILE* pipe = popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
#endif
    if (!pipe) return std::numeric_limits<float>::quiet_NaN();

    char buf[64] = {0};
    float temp = std::numeric_limits<float>::quiet_NaN();
    if (fgets(buf, sizeof(buf), pipe)) {
        temp = static_cast<float>(atof(buf));
        if (!std::isfinite(temp) || temp <= 0.0f) {
            temp = std::numeric_limits<float>::quiet_NaN();
        }
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return temp;
}

static std::vector<uint8_t> parse_checkpoint_layer_mask(const char* spec, int n_layers) {
    std::vector<uint8_t> mask(n_layers, 1u);
    if (!spec || !spec[0] || strcmp(spec, "all") == 0) {
        return mask;
    }
    if (strcmp(spec, "none") == 0) {
        std::fill(mask.begin(), mask.end(), 0u);
        return mask;
    }
    std::fill(mask.begin(), mask.end(), 0u);
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        int idx = atoi(tok.c_str());
        if (idx < 0 || idx >= n_layers) {
            fprintf(stderr, "[CT] invalid --grad-checkpoint-layers index: %d (n_layers=%d)\n", idx, n_layers);
            exit(EXIT_FAILURE);
        }
        mask[idx] = 1u;
    }
    return mask;
}

// =================== Trainer Lifecycle =================

void Trainer::create(const TrainConfig& cfg) {
    config = cfg;
    current_step = 0;
    last_train_loss = std::numeric_limits<float>::quiet_NaN();
    last_val_loss = std::numeric_limits<float>::quiet_NaN();
    loss_scale = cfg.loss_scale_init;
    good_steps_since_scale = 0;
    last_overflow_step = -1;
    sample_rng.seed(cfg.seed);

    // ---- memory pools ----
    size_t pers_bytes = cfg.persistent_pool_mb * 1024ULL * 1024ULL;
    size_t scr_bytes  = cfg.scratch_pool_mb    * 1024ULL * 1024ULL;
    persistent.create(pers_bytes, "persistent");
    scratch.create(scr_bytes, "scratch");

    // ---- model ----
    model.create(cfg.model, persistent);
    model.init_weights(cfg.seed);
    model.enable_mixed_precision(cfg.mixed_precision);
    std::vector<uint8_t> ckpt_mask = parse_checkpoint_layer_mask(cfg.checkpoint_layers, cfg.model.n_layers);
    model.configure_gradient_checkpointing(cfg.gradient_checkpointing, ckpt_mask);
    model.collect_params();

    int total_scalars = 0;
    for (auto& p : model.params) total_scalars += p.numel;
    printf("[CT] Model: %d scalars across %d param tensors\n",
           total_scalars, (int)model.params.size());

    // ---- optimizer ----
    optimizer.config = cfg.optim;
    optimizer.create(model.params, persistent);

    printf("[CT] Persistent pool: %.1f MB / %.1f MB\n",
           persistent.used() / (1024.0 * 1024.0),
           persistent.capacity() / (1024.0 * 1024.0));

    // ---- lr schedule ----
    lr_schedule.peak_lr      = cfg.peak_lr;
    lr_schedule.min_lr       = cfg.min_lr;
    lr_schedule.warmup_steps = cfg.warmup_steps;
    lr_schedule.total_steps  = cfg.max_steps;

    // ---- data ----
    train_data.create(cfg.train_path, cfg.seq_len, cfg.seed);

    has_val = (cfg.val_path != nullptr && cfg.val_path[0] != '\0');
    if (has_val) {
        val_data.create(cfg.val_path, cfg.seq_len, cfg.seed + 1);
    }

    // ---- checkpoint dir ----
    if (cfg.checkpoint_dir && cfg.checkpoint_dir[0]) {
        ct_mkdir(cfg.checkpoint_dir);
    }

    // ---- CSV log ----
    log_file = nullptr;
    if (cfg.log_path && cfg.log_path[0]) {
        log_file = fopen(cfg.log_path, "w");
        if (log_file) {
            fprintf(log_file, "step,loss,val_loss,val_ppl,lr,grad_norm,tok_per_sec,gpu_temp,vram_mb,elapsed_s,"
                    "loss_scale,overflow,fwd_ms,bwd_ms,optim_ms,vram_real_mb,"
                    "attn_qkv_ms,attn_scores_ms,attn_softmax_ms,attn_context_ms,attn_proj_ms,"
                    "ffn_up_ms,ffn_act_ms,ffn_down_ms,ln_ms,residual_ms,embed_ms,ce_ms,"
                    "kernel_launches\n");
            fflush(log_file);
        }
    }

    // ---- benchmark events + profiler ----
    last_fwd_ms = 0; last_bwd_ms = 0; last_optim_ms = 0; last_vram_real_mb = 0;
    if (cfg.benchmark) {
        cudaEventCreate(&ev_fwd_start); cudaEventCreate(&ev_fwd_end);
        cudaEventCreate(&ev_bwd_start); cudaEventCreate(&ev_bwd_end);
        cudaEventCreate(&ev_opt_start); cudaEventCreate(&ev_opt_end);
        g_profiler.active = true;
        g_profiler.create();
    }

    printf("[CT] Fusion: bias_gelu=%d residual_ln=%d causal_softmax=%d matmul_bias=%d "
           "cross_entropy=%d scale_softmax=%d qkv_proj=%d\n",
           g_fusion.bias_gelu, g_fusion.residual_ln, g_fusion.causal_softmax,
           g_fusion.matmul_bias, g_fusion.cross_entropy, g_fusion.scale_softmax,
           g_fusion.qkv_proj);

    DiagnosticsConfig dcfg = {};
    dcfg.enabled = cfg.diag_enabled;
    dcfg.tier2_interval = cfg.diag_tier2_interval;
    dcfg.tier3_interval = cfg.diag_tier3_interval;
    dcfg.tier4_interval = cfg.diag_tier4_interval;
    dcfg.hist_bins = cfg.diag_hist_bins;
    dcfg.attn_sample_layers = cfg.diag_attn_layers;
    dcfg.attn_sample_heads = cfg.diag_attn_heads;
    dcfg.attn_sample_queries = cfg.diag_attn_queries;
    dcfg.async_io = cfg.diag_async_io;
    dcfg.log_dir = cfg.diag_log_dir;
    cudaDeviceProp prop;
    CT_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    diagnostics.create(dcfg, cfg.model, prop.name);
}

void Trainer::destroy() {
    if (config.benchmark) {
        cudaEventDestroy(ev_fwd_start); cudaEventDestroy(ev_fwd_end);
        cudaEventDestroy(ev_bwd_start); cudaEventDestroy(ev_bwd_end);
        cudaEventDestroy(ev_opt_start); cudaEventDestroy(ev_opt_end);
        g_profiler.destroy();
    }
    if (log_file) { fclose(log_file); log_file = nullptr; }
    diagnostics.destroy();
    train_data.destroy();
    if (has_val) val_data.destroy();
    model.destroy();
    scratch.destroy();
    persistent.destroy();
}

// =================== Training Loop =================

void Trainer::train() {
    printf("\n");
    printf("============================================================\n");
    printf("  Training: %d steps, accum=%d, seq=%d, lr=%.1e -> %.1e, mp=%s\n",
           config.max_steps, config.grad_accum_steps, config.seq_len,
           config.peak_lr, config.min_lr,
           config.mixed_precision ? "on" : "off");
    printf("============================================================\n\n");

    auto t_start = std::chrono::high_resolution_clock::now();
    float best_train_loss = std::numeric_limits<float>::infinity();

    for (; current_step < config.max_steps; current_step++) {
        auto t_step = std::chrono::high_resolution_clock::now();

        // ---- one optimizer step (with gradient accumulation) ----
        float gnorm = 0.0f;
        bool overflow = false;
        float loss = train_step(gnorm, overflow);
        last_train_loss = loss;
        best_train_loss = std::min(best_train_loss, loss);

        if (!std::isfinite(loss)) {
            fprintf(stderr, "[CT] Non-finite loss at step %d: %.8f\n", current_step, loss);
            exit(EXIT_FAILURE);
        }
        if (!std::isfinite(gnorm)) {
            fprintf(stderr, "[CT] Non-finite grad norm at step %d: %.8f\n", current_step, gnorm);
            exit(EXIT_FAILURE);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double step_s = std::chrono::duration<double>(t_end - t_step).count();
        double total_s = std::chrono::duration<double>(t_end - t_start).count();
        double tok_s = static_cast<double>(config.seq_len * config.grad_accum_steps) / step_s;

        float lr = lr_schedule.get(current_step);
        float val_loss = std::numeric_limits<float>::quiet_NaN();

        // ---- evaluation ----
        if (has_val && config.eval_interval > 0
            && current_step % config.eval_interval == 0
            && current_step > 0) {
            scratch.reset();
            val_loss = evaluate(100);
            last_val_loss = val_loss;
            float val_ppl = expf(val_loss);
            printf("  [eval] val_loss %.4f | val_ppl %.1f\n", val_loss, val_ppl);
        }

        // ---- per-step logging ----
        if (current_step % config.log_interval == 0) {
            float gpu_temp = try_read_gpu_temp();
            log_step(current_step, loss, val_loss, lr, gnorm, total_s, tok_s,
                     gpu_temp, loss_scale, overflow);
            if (diagnostics.enabled()) {
                StepRecord rec = {};
                rec.step = static_cast<uint32_t>(current_step);
                rec.loss = loss;
                rec.val_loss = val_loss;
                rec.lr = lr;
                rec.grad_norm = gnorm;
                rec.throughput = static_cast<float>(tok_s);
                rec.vram_used_mb = static_cast<float>(persistent.used() + scratch.peak()) / (1024.0f * 1024.0f);
                rec.gpu_temp_c = gpu_temp;
                rec.loss_scale = loss_scale;
                rec.flags = overflow ? 1u : 0u;
                diagnostics.record_step(rec);
            }
        }

        if (diagnostics.enabled()) {
            if (config.diag_tier2_interval > 0 && current_step % config.diag_tier2_interval == 0) {
                run_diagnostics_snapshot();
                diagnostics.record_layer_stats(model, static_cast<uint32_t>(current_step));
            }
            if (config.diag_tier3_interval > 0 && current_step % config.diag_tier3_interval == 0) {
                run_diagnostics_snapshot();
                diagnostics.record_histograms(model, static_cast<uint32_t>(current_step));
            }
            if (config.diag_tier4_interval > 0 && current_step % config.diag_tier4_interval == 0) {
                run_diagnostics_snapshot();
                diagnostics.record_attention(model, static_cast<uint32_t>(current_step));
            }
        }

        // ---- text sample ----
        if (config.sample_interval > 0
            && current_step % config.sample_interval == 0
            && current_step > 0) {
            sample_text();
        }

        // ---- checkpoint ----
        if (config.checkpoint_dir && config.checkpoint_interval > 0
            && current_step % config.checkpoint_interval == 0
            && current_step > 0) {
            save_checkpoint();
        }
    }

    // ---- final checkpoint ----
    if (config.checkpoint_dir) {
        save_checkpoint();
    }

    printf("\n[CT] Training complete. %d steps. best_loss=%.6f", config.max_steps, best_train_loss);
    if (std::isfinite(last_val_loss)) {
        printf(" last_val_loss=%.6f", last_val_loss);
    }
    printf("\n");
}

void Trainer::run_diagnostics_snapshot() {
    DataLoader& src = has_val ? val_data : train_data;
    scratch.reset();
    src.next_batch();
    Tensor logits = scratch.alloc(config.seq_len, config.model.vocab_size, DType::FP32);
    model.forward(src.d_tokens, config.seq_len, logits, scratch);
    CT_CHECK_CUDA(cudaStreamSynchronize(0));
}

// =================== Single Training Step =================

float Trainer::train_step(float& out_grad_norm, bool& out_overflow) {
    out_overflow = false;
    model.zero_grads();
    g_profiler.reset();

    float total_loss = 0.0f;
    float fwd_accum = 0.0f, bwd_accum = 0.0f;

    for (int micro = 0; micro < config.grad_accum_steps; micro++) {
        scratch.reset();
        train_data.next_batch();

        // ---- forward (sets saved activations) ----
        if (config.benchmark) cudaEventRecord(ev_fwd_start);
        Tensor logits = scratch.alloc(config.seq_len, config.model.vocab_size, DType::FP32);
        model.forward(train_data.d_tokens, config.seq_len, logits, scratch);
        if (config.benchmark) cudaEventRecord(ev_fwd_end);

        // ---- backward (accumulates grads, returns mean loss) ----
        if (config.benchmark) cudaEventRecord(ev_bwd_start);
        float grad_scale = config.mixed_precision ? loss_scale : 1.0f;
        float micro_loss = model.backward(train_data.d_tokens, train_data.d_targets,
                                          config.seq_len, scratch, 0, grad_scale);
        if (config.benchmark) cudaEventRecord(ev_bwd_end);
        total_loss += micro_loss;

        if (config.benchmark) {
            float ms = 0;
            cudaEventSynchronize(ev_fwd_end);
            cudaEventElapsedTime(&ms, ev_fwd_start, ev_fwd_end);
            fwd_accum += ms;
            cudaEventSynchronize(ev_bwd_end);
            cudaEventElapsedTime(&ms, ev_bwd_start, ev_bwd_end);
            bwd_accum += ms;
        }
    }

    total_loss /= config.grad_accum_steps;

    // ---- scale grads by 1/accum_steps (and unscale if dynamic loss scaling is enabled) ----
    float post_scale = 1.0f / config.grad_accum_steps;
    if (config.mixed_precision && loss_scale > 1.0f) {
        post_scale /= loss_scale;
    }
    optim::grad_scale(model.params, post_scale);

    if (config.mixed_precision) {
        scratch.reset();
        if (!optim::grads_finite(model.params, scratch)) {
            out_overflow = true;
            last_overflow_step = current_step;
            loss_scale = fmaxf(1.0f, loss_scale * config.loss_scale_backoff);
            good_steps_since_scale = 0;
            model.zero_grads();
            out_grad_norm = 0.0f;
            if (config.benchmark) { last_fwd_ms = fwd_accum; last_bwd_ms = bwd_accum; last_optim_ms = 0; }
            return total_loss;
        }
    }

    // ---- grad norm + clip ----
    scratch.reset();
    out_grad_norm = optim::grad_norm(model.params, scratch);
    if (out_grad_norm > config.optim.grad_clip) {
        float clip_coef = config.optim.grad_clip / out_grad_norm;
        optim::grad_scale(model.params, clip_coef);
    }

    // ---- optimizer step ----
    if (config.benchmark) cudaEventRecord(ev_opt_start);
    float lr = lr_schedule.get(current_step);
    optimizer.step(model.params, lr);
    if (config.benchmark) cudaEventRecord(ev_opt_end);

    if (config.mixed_precision) {
        model.sync_model_weights_from_master();
        good_steps_since_scale++;
        if (good_steps_since_scale >= config.loss_scale_interval) {
            loss_scale *= config.loss_scale_growth;
            good_steps_since_scale = 0;
        }
    }

    // ---- collect benchmark timing ----
    if (config.benchmark) {
        float opt_ms = 0;
        cudaEventSynchronize(ev_opt_end);
        cudaEventElapsedTime(&opt_ms, ev_opt_start, ev_opt_end);
        last_fwd_ms = fwd_accum;
        last_bwd_ms = bwd_accum;
        last_optim_ms = opt_ms;
    }

    return total_loss;
}

// =================== Evaluation =================

float Trainer::evaluate(int num_batches) {
    float total_loss = 0.0f;

    for (int i = 0; i < num_batches; i++) {
        scratch.reset();
        val_data.next_batch();

        float loss = model.compute_loss(val_data.d_tokens, val_data.d_targets,
                                        config.seq_len, scratch);
        total_loss += loss;
    }

    return total_loss / num_batches;
}

// =================== Greedy Text Sampling =================

void Trainer::sample_text() {
    // ---- decode: start from random val offset, generate sample_len tokens ----
    int gen_len = config.sample_len;
    int max_seq = config.model.max_seq_len;
    int V = config.model.vocab_size;

    if (gen_len > max_seq - 1) gen_len = max_seq - 1;

    // ---- seed with a few tokens from val data (or training data) ----
    int ctx_len = 4;
    DataLoader& src = has_val ? val_data : train_data;
    src.next_batch();

    // ---- host-side token buffer ----
    std::vector<int> tokens(ctx_len + gen_len);
    CT_CHECK_CUDA(cudaMemcpy(tokens.data(), src.d_tokens,
                              ctx_len * sizeof(int), cudaMemcpyDeviceToHost));

    // ---- autoregressive generation ----
    int* d_tok;
    CT_CHECK_CUDA(cudaMalloc(&d_tok, max_seq * sizeof(int)));

    for (int step = 0; step < gen_len; step++) {
        int cur_len = ctx_len + step;

        scratch.reset();

        CT_CHECK_CUDA(cudaMemcpy(d_tok, tokens.data(),
                                  cur_len * sizeof(int), cudaMemcpyHostToDevice));

        Tensor logits = scratch.alloc(cur_len, V, DType::FP32);
        model.forward(d_tok, cur_len, logits, scratch);

        // ---- argmax of last position ----
        // copy last row of logits to host
        std::vector<float> last_logits(V);
        CT_CHECK_CUDA(cudaMemcpy(last_logits.data(),
                                  logits.f32() + (cur_len - 1) * V,
                                  V * sizeof(float), cudaMemcpyDeviceToHost));

        int next_tok = 0;
        if (config.sample_temperature <= 0.0f) {
            int best = 0;
            float best_val = last_logits[0];
            for (int j = 1; j < V; j++) {
                if (last_logits[j] > best_val) {
                    best_val = last_logits[j];
                    best = j;
                }
            }
            next_tok = best;
        } else {
            float temp = config.sample_temperature;
            float inv_temp = 1.0f / temp;
            float mx = last_logits[0];
            for (int j = 1; j < V; j++) mx = fmaxf(mx, last_logits[j]);

            std::vector<float> probs(V);
            float sum = 0.0f;
            for (int j = 0; j < V; j++) {
                float p = expf((last_logits[j] - mx) * inv_temp);
                probs[j] = p;
                sum += p;
            }
            if (!(sum > 0.0f) || !std::isfinite(sum)) {
                next_tok = 0;
            } else {
                std::uniform_real_distribution<float> dist(0.0f, sum);
                float r = dist(sample_rng);
                float cdf = 0.0f;
                int chosen = V - 1;
                for (int j = 0; j < V; j++) {
                    cdf += probs[j];
                    if (r <= cdf) {
                        chosen = j;
                        break;
                    }
                }
                next_tok = chosen;
            }
        }
        tokens[ctx_len + step] = next_tok;
    }

    cudaFree(d_tok);

    // ---- print token IDs (vocab decode happens outside C++) ----
    printf("  [sample] tokens: ");
    for (int i = 0; i < ctx_len + gen_len && i < 40; i++) {
        printf("%d ", tokens[i]);
    }
    if (ctx_len + gen_len > 40) printf("...");
    printf("\n");
}

// =================== Logging =================

void Trainer::log_step(int step, float loss, float val_loss, float lr, float gnorm,
                       double elapsed_s, double tok_per_sec, float gpu_temp_c,
                       float cur_loss_scale, bool overflow) {
    float ppl = expf(loss);
    float vram_mb = static_cast<float>(persistent.used() + scratch.peak()) / (1024.0f * 1024.0f);

    // ---- real VRAM via driver query (includes WDDM overhead, fragmentation, etc.)
    if (config.benchmark) {
        size_t free_bytes, total_bytes;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        last_vram_real_mb = static_cast<float>(total_bytes - free_bytes) / (1024.0f * 1024.0f);
    }

    if (std::isfinite(gpu_temp_c)) {
        printf("Step %d/%d | loss %.4f | ppl %.1f | lr %.1e | gnorm %.2f | %.0f tok/s | %.1f C | %.0f MB | ls %.0f%s\n",
               step, config.max_steps, loss, ppl, lr, gnorm, tok_per_sec, gpu_temp_c, vram_mb,
               cur_loss_scale, overflow ? " | overflow" : "");
    } else {
        printf("Step %d/%d | loss %.4f | ppl %.1f | lr %.1e | gnorm %.2f | %.0f tok/s | n/a C | %.0f MB | ls %.0f%s\n",
               step, config.max_steps, loss, ppl, lr, gnorm, tok_per_sec, vram_mb,
               cur_loss_scale, overflow ? " | overflow" : "");
    }

    if (log_file) {
        char temp_buf[32];
        if (std::isfinite(gpu_temp_c)) {
            snprintf(temp_buf, sizeof(temp_buf), "%.1f", gpu_temp_c);
        } else {
            temp_buf[0] = '\0';
        }
        // ---- per-kernel timing suffix (same for both branches) ----
        char kernel_buf[512];
        snprintf(kernel_buf, sizeof(kernel_buf),
                 ",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d",
                 g_profiler.attn_qkv_ms, g_profiler.attn_scores_ms,
                 g_profiler.attn_softmax_ms, g_profiler.attn_context_ms, g_profiler.attn_proj_ms,
                 g_profiler.ffn_up_ms, g_profiler.ffn_act_ms, g_profiler.ffn_down_ms,
                 g_profiler.ln_ms, g_profiler.residual_ms, g_profiler.embed_ms, g_profiler.ce_ms,
                 g_profiler.launch_count);

        if (std::isfinite(val_loss)) {
            float val_ppl = expf(val_loss);
            fprintf(log_file, "%d,%.6f,%.6f,%.6f,%.6e,%.4f,%.1f,%s,%.1f,%.1f,%.1f,%d,%.2f,%.2f,%.2f,%.1f%s\n",
                    step, loss, val_loss, val_ppl, lr, gnorm, tok_per_sec, temp_buf, vram_mb, elapsed_s,
                    cur_loss_scale, overflow ? 1 : 0,
                    last_fwd_ms, last_bwd_ms, last_optim_ms, last_vram_real_mb, kernel_buf);
        } else {
            fprintf(log_file, "%d,%.6f,,,%.6e,%.4f,%.1f,%s,%.1f,%.1f,%.1f,%d,%.2f,%.2f,%.2f,%.1f%s\n",
                    step, loss, lr, gnorm, tok_per_sec, temp_buf, vram_mb, elapsed_s,
                    cur_loss_scale, overflow ? 1 : 0,
                    last_fwd_ms, last_bwd_ms, last_optim_ms, last_vram_real_mb, kernel_buf);
        }
        fflush(log_file);
    }
}

// =================== Checkpointing =================

void Trainer::save_checkpoint() {
    char path[512];
    snprintf(path, sizeof(path), "%s/ckpt_step_%07d.bin",
             config.checkpoint_dir, current_step);

    Checkpoint::save(path, model, optimizer, current_step,
                     last_train_loss);

    Checkpoint::prune(config.checkpoint_dir, "ckpt_step_", 3);
}

}} // namespace ct::training
