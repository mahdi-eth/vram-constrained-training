#include "ct/training/trainer.cuh"
#include "ct/training/checkpoint.cuh"
#include "ct/fusion_config.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

using namespace ct;
using namespace ct::training;
using namespace ct::layers;

// =================== Default Configuration =================

static TrainConfig default_config() {
    TrainConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    // ---- model (Medium: ~55M params) ----
    cfg.model.vocab_size  = 8192;
    cfg.model.max_seq_len = 256;
    cfg.model.d_model     = 640;
    cfg.model.n_layers    = 10;
    cfg.model.n_heads     = 10;
    cfg.model.ffn_dim     = 2560;

    // ---- optimizer ----
    cfg.optim.beta1        = 0.9f;
    cfg.optim.beta2        = 0.95f;
    cfg.optim.eps          = 1e-8f;
    cfg.optim.weight_decay = 0.01f;
    cfg.optim.grad_clip    = 1.0f;

    // ---- training ----
    cfg.seq_len             = 256;
    cfg.grad_accum_steps    = 32;
    cfg.max_steps           = 50000;
    cfg.eval_interval       = 500;
    cfg.checkpoint_interval = 2000;
    cfg.log_interval        = 10;
    cfg.sample_interval     = 1000;
    cfg.sample_len          = 64;
    cfg.sample_temperature  = 0.0f;
    cfg.gradient_checkpointing = false;
    cfg.checkpoint_layers = "all";

    // ---- lr schedule ----
    cfg.peak_lr      = 3e-4f;
    cfg.min_lr       = 1e-5f;
    cfg.warmup_steps = 500;
    cfg.mixed_precision   = false;
    cfg.loss_scale_init   = 65536.0f;
    cfg.loss_scale_growth = 2.0f;
    cfg.loss_scale_backoff = 0.5f;
    cfg.loss_scale_interval = 2000;

    cfg.diag_enabled = false;
    cfg.diag_tier2_interval = 100;
    cfg.diag_tier3_interval = 1000;
    cfg.diag_tier4_interval = 1000;
    cfg.diag_hist_bins = 128;
    cfg.diag_attn_layers = 3;
    cfg.diag_attn_heads = 2;
    cfg.diag_attn_queries = 4;
    cfg.diag_async_io = true;
    cfg.diag_log_dir = "logs/diag";

    cfg.benchmark = false;

    // ---- paths ----
    cfg.train_path     = "data/tokenized/train.bin";
    cfg.val_path       = "data/tokenized/val.bin";
    cfg.checkpoint_dir = "checkpoints";
    cfg.log_path       = "train_log.csv";

    // ---- memory pools ----
    cfg.persistent_pool_mb = 1200;
    cfg.scratch_pool_mb    = 300;

    cfg.seed = 42;

    return cfg;
}

// =================== CLI Argument Parsing =================

static void parse_args(int argc, char** argv, TrainConfig& cfg) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            cfg.train_path = argv[++i];
        } else if (strcmp(argv[i], "--val") == 0 && i + 1 < argc) {
            cfg.val_path = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            cfg.max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            cfg.peak_lr = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--accum") == 0 && i + 1 < argc) {
            cfg.grad_accum_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            cfg.seq_len = atoi(argv[++i]);
            cfg.model.max_seq_len = cfg.seq_len;
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            cfg.model.n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            cfg.model.d_model = atoi(argv[++i]);
            cfg.model.ffn_dim = cfg.model.d_model * 4;
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            cfg.model.n_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            cfg.model.vocab_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ckpt-dir") == 0 && i + 1 < argc) {
            cfg.checkpoint_dir = argv[++i];
        } else if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            cfg.log_path = argv[++i];
        } else if (strcmp(argv[i], "--eval-interval") == 0 && i + 1 < argc) {
            cfg.eval_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--checkpoint-interval") == 0 && i + 1 < argc) {
            cfg.checkpoint_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--log-interval") == 0 && i + 1 < argc) {
            cfg.log_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sample-interval") == 0 && i + 1 < argc) {
            cfg.sample_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sample-len") == 0 && i + 1 < argc) {
            cfg.sample_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sample-temp") == 0 && i + 1 < argc) {
            cfg.sample_temperature = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--grad-checkpoint") == 0 && i + 1 < argc) {
            cfg.gradient_checkpointing = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--grad-checkpoint-layers") == 0 && i + 1 < argc) {
            cfg.checkpoint_layers = argv[++i];
        } else if (strcmp(argv[i], "--warmup-steps") == 0 && i + 1 < argc) {
            cfg.warmup_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-lr") == 0 && i + 1 < argc) {
            cfg.min_lr = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--mixed-precision") == 0 && i + 1 < argc) {
            cfg.mixed_precision = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--loss-scale-init") == 0 && i + 1 < argc) {
            cfg.loss_scale_init = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--loss-scale-growth") == 0 && i + 1 < argc) {
            cfg.loss_scale_growth = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--loss-scale-backoff") == 0 && i + 1 < argc) {
            cfg.loss_scale_backoff = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--loss-scale-interval") == 0 && i + 1 < argc) {
            cfg.loss_scale_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--diag") == 0 && i + 1 < argc) {
            cfg.diag_enabled = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--diag-log-dir") == 0 && i + 1 < argc) {
            cfg.diag_log_dir = argv[++i];
        } else if (strcmp(argv[i], "--diag-tier2") == 0 && i + 1 < argc) {
            cfg.diag_tier2_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--diag-tier3") == 0 && i + 1 < argc) {
            cfg.diag_tier3_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--diag-tier4") == 0 && i + 1 < argc) {
            cfg.diag_tier4_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--diag-hist-bins") == 0 && i + 1 < argc) {
            cfg.diag_hist_bins = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--diag-async-io") == 0 && i + 1 < argc) {
            cfg.diag_async_io = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
            cfg.benchmark = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--fusion") == 0 && i + 1 < argc) {
            const char* preset = argv[++i];
            if (strcmp(preset, "all") == 0)          g_fusion = FusionConfig::all_fused();
            else if (strcmp(preset, "none") == 0)     g_fusion = FusionConfig::all_unfused();
            else if (strcmp(preset, "baseline") == 0) g_fusion = FusionConfig::current_baseline();
            else {
                fprintf(stderr, "[CT] Unknown --fusion preset: %s (use all|none|baseline)\n", preset);
                exit(1);
            }
        } else if (strcmp(argv[i], "--fuse-bias-gelu") == 0 && i + 1 < argc) {
            g_fusion.bias_gelu = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--fuse-residual-ln") == 0 && i + 1 < argc) {
            g_fusion.residual_ln = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--fuse-causal-softmax") == 0 && i + 1 < argc) {
            g_fusion.causal_softmax = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--fuse-matmul-bias") == 0 && i + 1 < argc) {
            g_fusion.matmul_bias = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--fuse-cross-entropy") == 0 && i + 1 < argc) {
            g_fusion.cross_entropy = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--fuse-scale-softmax") == 0 && i + 1 < argc) {
            g_fusion.scale_softmax = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--fuse-qkv-proj") == 0 && i + 1 < argc) {
            g_fusion.qkv_proj = (atoi(argv[++i]) != 0);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = static_cast<uint64_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            (void)argv[++i]; // handled after trainer creation
        } else if (strcmp(argv[i], "--resume") == 0) {
            fprintf(stderr, "[CT] --resume requires a path\n");
            exit(1);
        } else if (strcmp(argv[i], "--persistent-mb") == 0 && i + 1 < argc) {
            cfg.persistent_pool_mb = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--scratch-mb") == 0 && i + 1 < argc) {
            cfg.scratch_pool_mb = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: ct_train [options]\n");
            printf("  --train PATH        Training data (CTRF binary)\n");
            printf("  --val PATH          Validation data (CTRF binary)\n");
            printf("  --steps N           Total optimizer steps\n");
            printf("  --lr FLOAT          Peak learning rate\n");
            printf("  --accum N           Gradient accumulation steps\n");
            printf("  --seq N             Sequence length\n");
            printf("  --dim N             Model dimension\n");
            printf("  --layers N          Number of transformer layers\n");
            printf("  --heads N           Number of attention heads\n");
            printf("  --vocab N           Vocabulary size\n");
            printf("  --ckpt-dir PATH     Checkpoint directory\n");
            printf("  --log PATH          CSV log path\n");
            printf("  --eval-interval N   Validation interval in steps (0 disables)\n");
            printf("  --checkpoint-interval N  Checkpoint interval in steps (0 disables)\n");
            printf("  --log-interval N    Logging interval in steps\n");
            printf("  --sample-interval N Text sample interval in steps (0 disables)\n");
            printf("  --sample-len N      Number of tokens to sample\n");
            printf("  --sample-temp FLOAT Sampling temperature (0 = greedy)\n");
            printf("  --grad-checkpoint 0|1 Enable gradient checkpointing\n");
            printf("  --grad-checkpoint-layers SPEC all|none|0,2,4\n");
            printf("  --warmup-steps N    LR warmup steps\n");
            printf("  --min-lr FLOAT      Minimum learning rate\n");
            printf("  --mixed-precision 0|1   Enable mixed-precision controls\n");
            printf("  --loss-scale-init FLOAT Initial loss scale\n");
            printf("  --loss-scale-growth FLOAT Loss scale growth factor\n");
            printf("  --loss-scale-backoff FLOAT Loss scale backoff factor\n");
            printf("  --loss-scale-interval N Steps between loss scale growth\n");
            printf("  --diag 0|1          Enable diagnostics logging\n");
            printf("  --diag-log-dir PATH Diagnostics output directory\n");
            printf("  --diag-tier2 N      Tier-2 interval\n");
            printf("  --diag-tier3 N      Tier-3 interval\n");
            printf("  --diag-tier4 N      Tier-4 interval\n");
            printf("  --diag-hist-bins N  Histogram bins\n");
            printf("  --diag-async-io 0|1 Async diagnostics writer\n");
            printf("  --benchmark 0|1     Enable CUDA event timing + real VRAM measurement\n");
            printf("  --fusion PRESET     Fusion preset: all|none|baseline\n");
            printf("  --fuse-bias-gelu 0|1       Toggle bias+GELU fusion\n");
            printf("  --fuse-residual-ln 0|1     Toggle residual+layernorm fusion\n");
            printf("  --fuse-causal-softmax 0|1  Toggle causal mask+softmax fusion\n");
            printf("  --fuse-matmul-bias 0|1     Toggle matmul+bias fusion\n");
            printf("  --fuse-cross-entropy 0|1   Toggle log-softmax+NLL fusion\n");
            printf("  --fuse-scale-softmax 0|1   Toggle scale+causal softmax fusion\n");
            printf("  --fuse-qkv-proj 0|1        Toggle merged QKV projection\n");
            printf("  --seed N            Random seed\n");
            printf("  --resume PATH       Resume from checkpoint\n");
            printf("  --persistent-mb N   Persistent pool size (MB)\n");
            printf("  --scratch-mb N      Scratch pool size (MB)\n");
            exit(0);
        } else {
            fprintf(stderr, "[CT] Unknown argument: %s\n", argv[i]);
            exit(1);
        }
    }
}

static const char* find_arg(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return nullptr;
}

// =================== Main =================

int main(int argc, char** argv) {
    printf("\n");
    printf("  Cuda-Transformer Training\n");
    printf("  -------------------------\n\n");

    // ---- GPU info ----
    cudaDeviceProp prop;
    CT_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("[CT] GPU: %s (%.0f MB, SM %d.%d)\n",
           prop.name,
           prop.totalGlobalMem / (1024.0 * 1024.0),
           prop.major, prop.minor);

    // ---- config ----
    TrainConfig cfg = default_config();
    parse_args(argc, argv, cfg);

    printf("[CT] Config: D=%d, L=%d, H=%d, FFN=%d, V=%d, seq=%d\n",
           cfg.model.d_model, cfg.model.n_layers, cfg.model.n_heads,
           cfg.model.ffn_dim, cfg.model.vocab_size, cfg.seq_len);
    printf("[CT] Training: %d steps, accum=%d, lr=%.1e, wd=%.3f, clip=%.1f, mp=%s, temp=%.2f, ckpt=%s, bench=%s\n",
           cfg.max_steps, cfg.grad_accum_steps, cfg.peak_lr,
           cfg.optim.weight_decay, cfg.optim.grad_clip,
           cfg.mixed_precision ? "on" : "off", cfg.sample_temperature,
           cfg.gradient_checkpointing ? "on" : "off",
           cfg.benchmark ? "on" : "off");

    // ---- create trainer ----
    Trainer trainer;
    trainer.create(cfg);

    // ---- resume from checkpoint if requested ----
    const char* resume_path = find_arg(argc, argv, "--resume");
    if (resume_path) {
        int step = 0;
        float loss = 0.0f;
        if (Checkpoint::load(resume_path, trainer.model, trainer.optimizer, step, loss)) {
            trainer.current_step = step;
            if (cfg.mixed_precision) {
                trainer.model.sync_model_weights_from_master();
            }
            printf("[CT] Resuming from step %d\n", step);
        } else {
            fprintf(stderr, "[CT] Failed to load checkpoint: %s\n", resume_path);
            exit(1);
        }
    }

    // ---- train ----
    trainer.train();
    trainer.destroy();

    return 0;
}
