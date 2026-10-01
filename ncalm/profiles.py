"""Training profile definitions.

V3 profiles: smoke (quick validation) and overnight (full run).
V4 profiles: smoke_v4 and overnight_v4 for the diffusion NCA model.
Only scale parameters differ — LR, architecture, etc. stay the same.
"""

SMOKE_PROFILE = {
    "description": "Smoke test — validate training works (200 steps)",
    # === Scale ===
    "max_steps": 20,
    "batch_size": 4,
    "seq_len": 256,
    # Token budget: 200 * 256 * 256 = 13,107,200 tokens
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 30,  # 15% of training
    # === Logging & Checkpointing ===
    "log_interval": 1,
    "eval_interval": 50,
    "eval_batches": 20,
    "generate_interval": 100,
    "save_interval": 200,
    "dataset": "tinystories",
}

OVERNIGHT_PROFILE = {
    "description": "Overnight — max quality (5000 steps)",
    # === Scale ===
    "max_steps": 5000,
    "batch_size": 24,
    "seq_len": 256,
    # Token budget: 5000 * 24 * 256 = 30,720,000 tokens (30.7M)
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.05,  # More aggressive than smoke
    "grad_clip": 1.0,
    "warmup_steps": 500,  # 10% of training
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 250,
    "eval_batches": 50,
    "generate_interval": 500,
    "save_interval": 1000,
    "dataset": "tinystories",
}

SMOKE_V4 = {
    "description": "Smoke test — validate V4 training works (50 steps)",
    # === Scale ===
    "max_steps": 50,
    "batch_size": 32,
    "seq_len": 256,
    # Token budget: 50 * 32 * 256 = 409,600 tokens
    # === V4 Architecture Overrides ===
    "K": 24,
    "checkpoint_interval": 4,
    "M": 8,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 5,
    # === Logging & Checkpointing ===
    "log_interval": 1,
    "eval_interval": 25,
    "eval_batches": 20,
    "generate_interval": 50,
    "save_interval": 100,
    "dataset": "tinystories",
}

OVERNIGHT_V4 = {
    "description": "6-hour V4 training run (40000 steps, ~328M tokens)",
    # === Scale ===
    "max_steps": 40000,
    "batch_size": 32,
    "seq_len": 256,
    # Token budget: 40000 * 32 * 256 = 327,680,000 tokens (328M)
    # Estimated time: ~6 hours on RTX 4080 Super (~0.49s/step at K=32)
    # === V4 Architecture Overrides ===
    "K": 32,
    "checkpoint_interval": 4,
    "M": 8,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.05,
    "grad_clip": 1.0,
    "warmup_steps": 4000,  # 10% of training
    # === Logging & Checkpointing ===
    "log_interval": 50,
    "eval_interval": 1000,
    "eval_batches": 50,
    "generate_interval": 2000,
    "save_interval": 5000,
    "dataset": "tinystories",
}

SMOKE_V41 = {
    "description": "Smoke test — validate V4.1 training works (50 steps)",
    # === Scale ===
    "max_steps": 50,
    "batch_size": 32,
    "seq_len": 256,
    # Token budget: 50 * 32 * 256 = 409,600 tokens
    # === V4.1 Architecture Overrides (§11.1) ===
    "d_z": 192,
    "d_h": 192,
    "kernel_radius": 2,        # §1, Eq 1.2: wider perception
    "d_hidden": 768,           # §4.2, Eq 4.8: SwiGLU kernel hidden
    "n_trunk_layers": 4,       # §4.2, Eq 4.10: 4x SwiGLU blocks
    "d_ff": 512,               # §4.2, Eq 4.11: SwiGLU intermediate
    "alpha_res": 0.5,          # §4.2, Eq 4.14: residual scaling
    "n_sensor_heads": 4,       # §7, Eq 7.1: 4 sensor heads
    "K": 24,
    "M": 8,
    "checkpoint_interval": 4,
    # === V4.1 Masking Rebalance (§6.1, Eq 6.2) ===
    "p_random": 0.3,
    "p_causal": 0.5,
    "p_block": 0.2,
    # === V4.1 New Loss Weights (§9.1, Eq 9.12) ===
    "lambda_pred": 0.1,        # §9.1b: prediction error loss
    "lambda_div": 0.01,        # §9.1f: sensor diversity loss
    "lambda_prec": 0.001,      # §9.1g: precision prior loss
    "tau_div": 0.3,            # §7.5: diversity cosine threshold
    # === V4.1 Stochastic Dynamics (§8, Eq 8.2) ===
    "sigma_noise_init": 0.01,
    # === V4.1 Gradient Stabilization (§4.1, Eq 4.2-4.3) ===
    "lambda_agc": 0.02,
    # === V4.1 Curriculum (§6.3, Eq 6.11-6.13) ===
    "curriculum_easy_steps": 5000,
    "curriculum_mid_steps": 20000,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 5,
    # === Logging & Checkpointing ===
    "log_interval": 1,
    "eval_interval": 25,
    "eval_batches": 20,
    "generate_interval": 50,
    "save_interval": 100,
    "dataset": "tinystories",
}

OVERNIGHT_V41 = {
    "description": "V4.1 overnight — 30.7M tokens (matches V10.1 benchmark budget)",
    # === Scale ===
    # Token budget: 3000 * 40 * 256 = 30,720,000 tokens (30.7M)
    # Matches V10.1 baseline exactly for fair PPL comparison.
    # Estimated time: ~45-60 min on RTX 4080 Super (~11K tok/s)
    "max_steps": 3000,
    "batch_size": 40,          # B=40 leaves VRAM headroom for GradNorm
    "seq_len": 256,
    # === V4.1 Architecture Overrides (§11.1) ===
    "d_z": 192,
    "d_h": 192,
    "kernel_radius": 2,        # §1, Eq 1.2: wider perception
    "d_hidden": 768,           # §4.2, Eq 4.8: SwiGLU kernel hidden
    "n_trunk_layers": 4,       # §4.2, Eq 4.10: 4x SwiGLU blocks
    "d_ff": 512,               # §4.2, Eq 4.11: SwiGLU intermediate
    "alpha_res": 0.5,          # §4.2, Eq 4.14: residual scaling
    "n_sensor_heads": 4,       # §7, Eq 7.1: 4 sensor heads
    "K": 32,                   # More NCA steps → better substrate refinement
    "M": 8,
    "checkpoint_interval": 4,
    # === V4.1 Masking Rebalance (§6.1, Eq 6.2) ===
    "p_random": 0.3,
    "p_causal": 0.5,
    "p_block": 0.2,
    # === V4.1 New Loss Weights (§9.1, Eq 9.12) ===
    "lambda_pred": 0.1,        # §9.1b: prediction error loss
    "lambda_div": 0.01,        # §9.1f: sensor diversity loss
    "lambda_prec": 0.001,      # §9.1g: precision prior loss
    "tau_div": 0.3,            # §7.5: diversity cosine threshold
    # === V4.1 Stochastic Dynamics (§8, Eq 8.2) ===
    "sigma_noise_init": 0.01,
    # === V4.1 Gradient Stabilization (§4.1, Eq 4.2-4.3) ===
    "lambda_agc": 0.02,
    "gradnorm_interval": 1000,  # GradNorm every 1000 steps (VRAM-safe; 3 updates total)
    # === V4.1 Curriculum (§6.3, Eq 6.11-6.13) ===
    # Scaled proportionally: easy=20%, mid=50% of total steps
    "curriculum_easy_steps": 600,
    "curriculum_mid_steps": 1500,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.05,      # Stronger regularization for production
    "grad_clip": 1.0,
    "warmup_steps": 300,       # 10% of training
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 150,      # Eval every 5% of training
    "eval_batches": 50,        # 50 val batches for stable estimates
    "generate_interval": 600,  # Text samples every 20%
    "save_interval": 600,      # Checkpoint every 20%
    "dataset": "tinystories",
}

# ═══════════════════════════════════════════════════════════════════════════
# V5 — Precision-Routed NCA Language Model (Autoregressive)
# ═══════════════════════════════════════════════════════════════════════════

SMOKE_V5 = {
    "description": "Smoke test — validate V5 training works (1000 steps, ~9.4M params)",
    # === Scale ===
    "max_steps": 1000,
    "batch_size": 32,
    "seq_len": 256,
    # Token budget: 1000 * 32 * 256 = 8,192,000 tokens
    # === V5 Architecture (Smoke = reduced dimensions) ===
    "d_model": 256,             # Cell state dim (reduced from 512)
    "d_msg": 32,                # PRMP message dim
    "d_ff": 512,                # Kernel feed-forward dim
    "d_pe": 64,                 # Sinusoidal PE dim for kernel input
    "n_heads": 4,               # PRMP attention heads
    "n_trunk": 2,               # SwiGLU blocks per kernel
    "kernel_radius": 1,         # Causal conv radius
    # NCA steps: K=12 (4+6+2)
    "k_phase1": 4,              # Propagation steps
    "k_phase2": 6,              # Integration steps
    "k_phase3": 2,              # Refinement steps
    # Readout
    "mos_components": 1,        # Single softmax for smoke
    "tie_embeddings": True,
    # Loss
    "lambda_reg": 0.01,
    "lambda_aux": 0.1,
    # Gradient
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 100,
    "eval_batches": 20,
    "generate_interval": 250,
    "save_interval": 500,
    "dataset": "tinystories",
}

OVERNIGHT_V5 = {
    "description": "V5.1 overnight — ~91.5M params, ~30M tokens, 3 conservative improvements",
    # === Scale ===
    # Token budget: 2440 * 48 * 256 = 29,982,720 tokens (~30M)
    # Estimated time: ~20-30 min on RTX 4080 Super
    # Peak VRAM: ~9.5 GB estimated (slightly more than V5.0 due to wider conv)
    "max_steps": 2440,
    "batch_size": 48,
    "seq_len": 256,
    # === V5 Architecture (XLarge config, ~91.5M params) ===
    "d_model": 768,             # Cell state dim — max capacity for this budget
    "d_msg": 96,                # PRMP message dim (H=4, d_head=24)
    "d_ff": 1536,               # Kernel feed-forward dim (2× d_model)
    "d_pe": 64,                 # Sinusoidal PE dim for kernel input
    "n_heads": 4,               # PRMP attention heads
    "n_trunk": 3,               # SwiGLU blocks per kernel — deeper trunk for larger model
    "kernel_radius": 2,         # V5.1: wider causal conv (was 1) — sees 3 positions instead of 2
    # NCA steps: K=16 (4+8+4) — 3 phases
    "k_phase1": 4,              # Propagation: establish local context
    "k_phase2": 8,              # Integration: global message passing refines state
    "k_phase3": 4,              # Refinement: polish for readout
    # Readout
    "mos_components": 1,        # Single softmax (MoS not needed — d_model=768 is expressive)
    "tie_embeddings": True,     # Save params, regularize
    # Loss
    "lambda_reg": 0.01,         # State drift regularization
    "lambda_aux": 0.1,          # Auxiliary early-exit loss at phase boundaries
    # Gradient
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,   # Gradient checkpoint every 4 NCA steps
    "dropout": 0.05,            # V5.1: mild dropout on PRMP output (was 0.0)
    # === Optimizer ===
    "learning_rate": 3e-4,      # Validated in smoke
    "weight_decay": 0.05,       # Stronger regularization for larger model
    "beta2": 0.99,              # V5.1: smoother second moment (was 0.95)
    "grad_clip": 1.0,
    "warmup_steps": 244,        # 10% of training
    # === Logging & Checkpointing ===
    "log_interval": 20,         # Log every 20 steps
    "eval_interval": 200,       # Eval ~12 times during training
    "eval_batches": 50,         # 50 val batches for stable PPL estimates
    "generate_interval": 500,   # Text samples ~5 times during training
    "save_interval": 610,       # Checkpoint every 25% (4 checkpoints)
    "dataset": "tinystories",
}

SMOLTALK2_5B_V5 = {
    "description": "V5.1 long-run on SmolTalk2 — 5B tokens with professional checkpointing",
    # === Scale ===
    # Token budget: 76294 * 8 * 1024 * 8 = 4,999,839,744 tokens (~5.0B)
    # Microbatch-based for 16GB-class GPUs with gradient accumulation.
    "max_steps": 76294,
    "batch_size": 8,
    "seq_len": 1024,
    "grad_accum_steps": 8,
    # === Data ===
    "dataset": "smoltalk2",
    "dataset_config": "SFT",
    "dataset_splits": [
        "smoltalk_smollm3_everyday_conversations_no_think",
        "smoltalk_smollm3_systemchats_30k_no_think",
        "smoltalk_smollm3_smol_magpie_ultra_no_think",
        "smoltalk_smollm3_explore_instruct_rewriting_no_think",
        "smoltalk_smollm3_smol_rewrite_no_think",
        "smoltalk_smollm3_smol_summarize_no_think",
        "OpenHermes_2.5_no_think",
        "OpenThoughts3_1.2M_no_think_no_think",
        "tulu_3_sft_personas_instruction_following_no_think",
    ],
    "dataset_hf_name": "HuggingFaceTB/smoltalk2",
    "dataset_cache_name": "smoltalk2_sft_tokens_5b.bin",
    "max_dataset_tokens": 5_000_000_000,
    # === Tokenizer ===
    "tokenizer_name": "pretrained_hf",
    "tokenizer_pretrained_name": "mistralai/Mistral-7B-v0.1",
    "tokenizer_vocab_size": 32000,
    # === Architecture ===
    "d_model": 768,
    "d_msg": 96,
    "d_ff": 1536,
    "d_pe": 64,
    "max_seq_len": 1024,
    "n_heads": 4,
    "n_trunk": 3,
    "kernel_radius": 2,
    "k_phase1": 4,
    "k_phase2": 8,
    "k_phase3": 4,
    # === Readout ===
    "mos_components": 1,
    "tie_embeddings": True,
    # === Loss / Stability ===
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.05,
    # === Optimizer ===
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "beta2": 0.99,
    "grad_clip": 1.0,
    "warmup_steps": 3000,
    # === Logging & Checkpointing ===
    "log_interval": 20,
    "eval_interval": 1000,
    "eval_batches": 50,
    "generate_interval": 2000,
    "save_interval": 500,
    "keep_last_n_checkpoints": 3,
    "backup_policy": "critical_only",
    "min_free_disk_gb": 20,
}

SMOLTALK2_5B_V5_MIXED = {
    "description": "V5.1 long-run on broader SmolTalk2 mix, redesigned toward a true multi-billion-token corpus",
    # === Scale ===
    "max_steps": 76294,
    "batch_size": 8,
    "seq_len": 1024,
    "grad_accum_steps": 8,
    # === Data ===
    "dataset": "smoltalk2",
    "dataset_hf_name": "HuggingFaceTB/smoltalk2",
    "dataset_cache_name": "smoltalk2_mixed_tokens_5b.bin",
    "max_dataset_tokens": 5_000_000_000,
    "dataset_sources": [
        {
            "config": "Mid",
            "splits": [
                "Llama_Nemotron_Post_Training_Dataset_reasoning_r1",
                "OpenThoughts3_1.2M",
            ],
        },
        {
            "config": "SFT",
            "splits": [
                "OpenThoughts3_1.2M_think",
                "smoltalk_multilingual8_Qwen3_32B_think",
                "smolagents_toolcalling_traces_think",
                "multi_turn_reasoning_if_think",
                "LongAlign_64k_Qwen3_32B_yarn_131k_think",
                "OpenHermes_2.5_no_think",
                "OpenThoughts3_1.2M_no_think_no_think",
                "smoltalk_smollm3_smol_magpie_ultra_no_think",
                "smoltalk_multilingual_8languages_lang_5_no_think",
                "smoltalk_smollm3_smol_summarize_no_think",
                "tulu_3_sft_personas_instruction_following_no_think",
                "xlam_traces_no_think",
            ],
        },
    ],
    # === Tokenizer ===
    "tokenizer_name": "pretrained_hf",
    "tokenizer_pretrained_name": "mistralai/Mistral-7B-v0.1",
    "tokenizer_vocab_size": 32000,
    # === Architecture ===
    "d_model": 768,
    "d_msg": 96,
    "d_ff": 1536,
    "d_pe": 64,
    "max_seq_len": 1024,
    "n_heads": 4,
    "n_trunk": 3,
    "kernel_radius": 2,
    "k_phase1": 4,
    "k_phase2": 8,
    "k_phase3": 4,
    # === Readout ===
    "mos_components": 1,
    "tie_embeddings": True,
    # === Loss / Stability ===
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.05,
    # === Optimizer ===
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "beta2": 0.99,
    "grad_clip": 1.0,
    "warmup_steps": 3000,
    # === Logging & Checkpointing ===
    "log_interval": 20,
    "eval_interval": 1000,
    "eval_batches": 50,
    "generate_interval": 2000,
    "save_interval": 500,
    "keep_last_n_checkpoints": 3,
    "backup_policy": "critical_only",
    "min_free_disk_gb": 20,
}

SMOLTALK2_5B_V5_PROBE = {
    "description": "Probe run for SmolTalk2 5B setup on local hardware",
    # === Scale ===
    # Keeps the exact long-run memory regime but only runs briefly.
    "max_steps": 40,
    "batch_size": 8,
    "seq_len": 1024,
    "grad_accum_steps": 8,
    # === Data ===
    "dataset": "smoltalk2",
    "dataset_config": "SFT",
    "dataset_splits": [
        "smoltalk_smollm3_everyday_conversations_no_think",
        "smoltalk_smollm3_systemchats_30k_no_think",
        "smoltalk_smollm3_smol_magpie_ultra_no_think",
        "OpenHermes_2.5_no_think",
    ],
    "dataset_hf_name": "HuggingFaceTB/smoltalk2",
    "dataset_cache_name": "smoltalk2_probe_tokens.bin",
    "max_dataset_tokens": 25_000_000,
    # === Tokenizer ===
    "tokenizer_name": "pretrained_hf",
    "tokenizer_pretrained_name": "mistralai/Mistral-7B-v0.1",
    "tokenizer_vocab_size": 32000,
    # === Architecture ===
    "d_model": 768,
    "d_msg": 96,
    "d_ff": 1536,
    "d_pe": 64,
    "max_seq_len": 1024,
    "n_heads": 4,
    "n_trunk": 3,
    "kernel_radius": 2,
    "k_phase1": 4,
    "k_phase2": 8,
    "k_phase3": 4,
    # === Readout ===
    "mos_components": 1,
    "tie_embeddings": True,
    # === Loss / Stability ===
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.05,
    # === Optimizer ===
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "beta2": 0.99,
    "grad_clip": 1.0,
    "warmup_steps": 4,
    # === Logging & Checkpointing ===
    "log_interval": 1,
    "eval_interval": 10,
    "eval_batches": 10,
    "generate_interval": 20,
    "save_interval": 10,
    "keep_last_n_checkpoints": 3,
    "backup_policy": "critical_only",
    "min_free_disk_gb": 20,
}

# ═══════════════════════════════════════════════════════════════════════════
# Academic Comparison: AENCA v5.1 vs GPT-2/NeoX Baseline on The Pile
# ═══════════════════════════════════════════════════════════════════════════
# Both models share identical: tokenizer (GPT-NeoX 50304), data (The Pile),
# optimizer settings, batch size, sequence length, and token budget.
# Differences: architecture only (AENCA NCA vs standard transformer).
# External reference: Pythia-160M at matched token count (~2B).

# ── Shared settings for Pile 2B comparison runs ─────────────────────────
_PILE_2B_COMMON = {
    # === Scale ===
    # 2B tokens / (8 × 1024 × 8) = 30517.6 → 30518 steps
    "max_steps": 30518,
    "batch_size": 8,
    "seq_len": 1024,
    "grad_accum_steps": 8,
    # === Data ===
    "dataset": "the_pile",
    "dataset_hf_name": "monology/pile-uncopyrighted",
    "dataset_cache_name": "pile_neox50k_tokens_2b.bin",
    "max_dataset_tokens": 2_000_000_000,
    # === Tokenizer (GPT-NeoX — same as Pythia) ===
    "tokenizer_name": "pretrained_hf",
    "tokenizer_pretrained_name": "EleutherAI/gpt-neox-20b",
    "tokenizer_vocab_size": 50304,
    # === Optimizer (matched for both architectures) ===
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "beta2": 0.99,
    "grad_clip": 1.0,
    "warmup_steps": 1500,
    # === Logging & Checkpointing ===
    "log_interval": 20,
    "eval_interval": 1000,
    "eval_batches": 50,
    "generate_interval": 5000,
    "save_interval": 5000,
    "keep_last_n_checkpoints": 3,
    "backup_policy": "critical_only",
    "min_free_disk_gb": 20,
}

AENCA_V5_PILE_2B = {
    **_PILE_2B_COMMON,
    "description": "AENCA v5.1 on The Pile — 2B tokens, ~161M params, academic comparison",
    # === Architecture (same body as 5B SmolTalk2 run, untied for ~161M) ===
    "d_model": 768,
    "d_msg": 96,
    "d_ff": 1536,
    "d_pe": 64,
    "max_seq_len": 1024,
    "n_heads": 4,
    "n_trunk": 3,
    "kernel_radius": 2,
    "k_phase1": 4,
    "k_phase2": 8,
    "k_phase3": 4,
    "mos_components": 1,
    "tie_embeddings": False,
    # === AENCA-specific losses ===
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,             # Zero dropout for fair comparison (matches GPT-2 & Pythia)
}

GPT2_PILE_2B = {
    **_PILE_2B_COMMON,
    "description": "GPT-2/NeoX baseline on The Pile — 2B tokens, ~162M params, academic comparison",
    # === Architecture (matches Pythia-160M) ===
    "d_model": 768,
    "n_layer": 12,
    "n_head": 12,
    "d_ff": 3072,
    "max_seq_len": 2048,
    "tie_embeddings": False,
    "use_parallel_residual": True,
    "rotary_pct": 0.25,
    # === No NCA auxiliary losses ===
    "lambda_reg": 0.0,
    "lambda_aux": 0.0,
    "lambda_agc": 0.0,
    "checkpoint_interval": 4,
    "dropout": 0.0,
}

# ── Smoke tests for VRAM & correctness validation ──────────────────────
_PILE_SMOKE_COMMON = {
    # Same memory regime as production, fewer steps + smaller cache
    "max_steps": 50,
    "batch_size": 8,
    "seq_len": 1024,
    "grad_accum_steps": 8,
    # === Data ===
    "dataset": "the_pile",
    "dataset_hf_name": "monology/pile-uncopyrighted",
    "dataset_cache_name": "pile_neox50k_tokens_smoke.bin",
    "max_dataset_tokens": 10_000_000,
    # === Tokenizer ===
    "tokenizer_name": "pretrained_hf",
    "tokenizer_pretrained_name": "EleutherAI/gpt-neox-20b",
    "tokenizer_vocab_size": 50304,
    # === Optimizer ===
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "beta2": 0.99,
    "grad_clip": 1.0,
    "warmup_steps": 5,
    # === Logging ===
    "log_interval": 1,
    "eval_interval": 10,
    "eval_batches": 10,
    "generate_interval": 25,
    "save_interval": 50,
}

AENCA_V5_PILE_SMOKE = {
    **_PILE_SMOKE_COMMON,
    "description": "AENCA v5.1 Pile smoke test — VRAM & correctness validation",
    "d_model": 768,
    "d_msg": 96,
    "d_ff": 1536,
    "d_pe": 64,
    "max_seq_len": 1024,
    "n_heads": 4,
    "n_trunk": 3,
    "kernel_radius": 2,
    "k_phase1": 4,
    "k_phase2": 8,
    "k_phase3": 4,
    "mos_components": 1,
    "tie_embeddings": False,
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.05,
}

GPT2_PILE_SMOKE = {
    **_PILE_SMOKE_COMMON,
    "description": "GPT-2/NeoX Pile smoke test — VRAM & correctness validation",
    "d_model": 768,
    "n_layer": 12,
    "n_head": 12,
    "d_ff": 3072,
    "max_seq_len": 2048,
    "tie_embeddings": False,
    "use_parallel_residual": True,
    "rotary_pct": 0.25,
    "lambda_reg": 0.0,
    "lambda_aux": 0.0,
    "lambda_agc": 0.0,
    "checkpoint_interval": 4,
    "dropout": 0.0,
}

# ═══════════════════════════════════════════════════════════════════════════
# V6 — Oscillator-Controlled Adaptive NCA
# ═══════════════════════════════════════════════════════════════════════════
# V6 replaces V5's flat 3-phase (4+8+4 = 16 steps, 3 kernels) with
# T_max NCA steps controlled by per-cell coupled Kuramoto oscillators.
# No attention by default; oscillators provide global coordination.

SMOKE_V6 = {
    "description": "Smoke test — validate V6 oscillator NCA training (1000 steps, ~9M params)",
    # === Scale ===
    "max_steps": 1000,
    "batch_size": 32,
    "seq_len": 256,
    # Token budget: 1000 * 32 * 256 = 8,192,000 tokens
    # === V6 Architecture (Smoke = reduced dimensions) ===
    "d_model": 256,             # Cell state dim
    "d_msg": 32,                # Slow updater message dim (dormant)
    "d_ff": 512,                # Fast kernel feed-forward dim
    "d_pe": 64,                 # Sinusoidal PE dim
    "d_slow_ff": 512,           # Slow updater MLP dim (dormant)
    "n_heads": 4,               # Slow updater heads (dormant)
    "n_trunk": 2,               # SwiGLU blocks in fast kernel
    "kernel_radius": 1,         # Causal conv radius
    # Oscillator dynamics (reduced for smoke)
    "n_osc_channels": 4,        # K: oscillator channels per cell
    "t_max": 16,                # Max NCA steps
    "checkpoint_segment_size": 4,  # Grad checkpoint segment
    # Readout
    "mos_components": 1,
    "tie_embeddings": True,
    # Loss
    "lambda_reg": 0.01,
    "lambda_aux": 0.1,
    "lambda_freq_reg": 0.001,
    # Gradient
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 100,
    "eval_batches": 20,
    "generate_interval": 250,
    "save_interval": 500,
    "dataset": "tinystories",
}

AENCA_V6_PILE_SMOKE = {
    **_PILE_SMOKE_COMMON,
    "description": "AENCA v6 Pile smoke test — VRAM & correctness validation (~164M params)",
    # VRAM override: smaller micro-batch, more accum (same effective batch)
    "batch_size": 2,
    "grad_accum_steps": 32,
    # Architecture: d_ff=2752 to match GPT-2 ~162M param budget
    "d_model": 768,
    "d_msg": 96,
    "d_ff": 2752,
    "d_pe": 64,
    "d_slow_ff": 2752,
    "max_seq_len": 1024,
    "n_heads": 4,
    "n_trunk": 3,
    "kernel_radius": 2,
    # Oscillator dynamics
    "n_osc_channels": 8,
    "t_max": 32,
    "checkpoint_segment_size": 8,
    # Readout
    "mos_components": 1,
    "tie_embeddings": False,
    # AENCA losses
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_freq_reg": 0.001,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
}

# ═══════════════════════════════════════════════════════════════════════════
# 250M Quick Comparison — AENCA v5.1 vs GPT-2 vs AENCA v6
# ═══════════════════════════════════════════════════════════════════════════
# 250M tokens = 2B / 8. Same data pipeline & optimizer, scaled-down schedule.
# Steps: 250M / (8 × 1024 × 8) = 3815

_PILE_250M_COMMON = {
    # === Scale ===
    "max_steps": 3815,
    "batch_size": 8,
    "seq_len": 1024,
    "grad_accum_steps": 8,
    # === Data ===
    "dataset": "the_pile",
    "dataset_hf_name": "monology/pile-uncopyrighted",
    "dataset_cache_name": "pile_neox50k_tokens_250m.bin",
    "max_dataset_tokens": 250_000_000,
    # === Tokenizer (GPT-NeoX — same as Pythia) ===
    "tokenizer_name": "pretrained_hf",
    "tokenizer_pretrained_name": "EleutherAI/gpt-neox-20b",
    "tokenizer_vocab_size": 50304,
    # === Optimizer (same as 2B — matched for all architectures) ===
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "beta2": 0.99,
    "grad_clip": 1.0,
    "warmup_steps": 188,
    # === Logging & Checkpointing (scaled for ~4K steps) ===
    "log_interval": 10,
    "eval_interval": 200,
    "eval_batches": 50,
    "generate_interval": 750,
    "save_interval": 750,
    "keep_last_n_checkpoints": 3,
    "backup_policy": "critical_only",
    "min_free_disk_gb": 20,
}

GPT2_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": "GPT-2/NeoX baseline on The Pile — 250M tokens, ~162M params, quick comparison",
    "d_model": 768,
    "n_layer": 12,
    "n_head": 12,
    "d_ff": 3072,
    "max_seq_len": 2048,
    "tie_embeddings": False,
    "use_parallel_residual": True,
    "rotary_pct": 0.25,
    "lambda_reg": 0.0,
    "lambda_aux": 0.0,
    "lambda_agc": 0.0,
    "checkpoint_interval": 4,
    "dropout": 0.0,
}

AENCA_V5_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": "AENCA v5.1 on The Pile — 250M tokens, ~161M params, quick comparison",
    "d_model": 768,
    "d_msg": 96,
    "d_ff": 1536,
    "d_pe": 64,
    "max_seq_len": 1024,
    "n_heads": 4,
    "n_trunk": 3,
    "kernel_radius": 2,
    "k_phase1": 4,
    "k_phase2": 8,
    "k_phase3": 4,
    "mos_components": 1,
    "tie_embeddings": False,
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
}

AENCA_V6_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": "AENCA v6 on The Pile — 250M tokens, ~162M params, K=16 oscillator, param-matched",
    # Batch: 4 × 1024 × 16 = 65536 tokens/step (matches GPT-2)
    "batch_size": 4,
    "grad_accum_steps": 16,
    # Architecture: lean oscillator (~1.3% of params, 2.2M),
    # kernel gets 92% of GPT-2's d_ff capacity (2830 vs 3072)
    "d_model": 768,
    "d_msg": 96,
    "d_ff": 2830,
    "d_pe": 64,
    "d_slow_ff": 2830,
    "max_seq_len": 1024,
    "n_heads": 4,
    "n_trunk": 3,
    "kernel_radius": 2,
    # Oscillator dynamics — K=16 channels, 2-layer MLPs
    # Lean oscillator: just enough to control temporal dynamics
    # d_osc_hidden=384 → ~2.2M oscillator params (1.3% of 162M)
    # t_max=16 NCA iterations (comparable to GPT-2's 12 unique layers)
    "n_osc_channels": 16,
    "d_osc_hidden": 384,
    "osc_freq_spread": 2.5,
    "t_max": 16,
    "t_min": 1,
    "checkpoint_segment_size": 4,
    # Deep supervision at power-of-2 boundaries within t_max
    "deep_supervision_steps": (2, 4, 8, 16),
    # Readout
    "mos_components": 1,
    "tie_embeddings": False,
    # Losses
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_freq_reg": 0.001,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
}

# FLOP-matched V6: smaller model but same compute budget as GPT-2
# The fair "compute-matched" comparison
AENCA_V6_PILE_250M_FLOPS = {
    **_PILE_250M_COMMON,
    "description": "AENCA v6 on The Pile — 250M tokens, FLOP-matched to GPT-2",
    # batch=4, accum=16 → 65536 tok/step (exact match with GPT-2)
    # peak VRAM ~9 GB, safe on 16 GB card
    "batch_size": 4,
    "grad_accum_steps": 16,
    # Architecture: d_model=512, d_ff=512 (smaller model, more steps)
    "d_model": 512,
    "d_msg": 64,
    "d_ff": 512,
    "d_pe": 64,
    "d_slow_ff": 1536,
    "max_seq_len": 1024,
    "n_heads": 4,
    "n_trunk": 3,
    "kernel_radius": 2,
    # Oscillator dynamics
    "n_osc_channels": 8,
    "t_max": 64,
    "checkpoint_segment_size": 8,
    # Readout
    "mos_components": 1,
    "tie_embeddings": False,
    # AENCA losses
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_freq_reg": 0.001,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
}

AENCA_V6_PILE_2B = {
    **_PILE_2B_COMMON,
    "description": "AENCA v6 on The Pile — 2B tokens, ~162M params, K=64 oscillator, academic comparison",
    # VRAM override: smaller micro-batch, more accum (same effective batch)
    # 2 × 1024 × 32 = 65536 tokens/step = same as GPT-2's 8 × 1024 × 8
    "batch_size": 2,
    "grad_accum_steps": 32,
    # Architecture: same as pile_250m_v6 but with t_max=256
    "d_model": 768,
    "d_msg": 96,
    "d_ff": 2384,
    "d_pe": 64,
    "d_slow_ff": 2384,
    "max_seq_len": 1024,
    "n_heads": 4,
    "n_trunk": 3,
    "kernel_radius": 2,
    # Oscillator dynamics (full-scale, freq_spread=2.5 keeps ω < π)
    "n_osc_channels": 64,
    "d_osc_hidden": 2560,
    "osc_freq_spread": 2.5,
    "t_max": 256,
    "checkpoint_segment_size": 8,
    # Readout
    "mos_components": 1,
    "tie_embeddings": False,
    # AENCA losses
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_freq_reg": 0.001,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
}

# ═══════════════════════════════════════════════════════════════════════════
# V7 — Stream-Hierarchical NCA
# ═══════════════════════════════════════════════════════════════════════════
# Two-timescale NCA:
#   θ level: unique attention + ultra-wide SwiGLU FFN (like transformer layers)
#   γ level: shared kernel (CausalConv → CSS → SlimKernel → gated residual)
# Causal State Streams (CSS) for O(L) global communication.
# Per-position adaptive halting at theta boundaries.

SMOKE_V7 = {
    "description": "Smoke test — validate V7.1 Hebbian NCA (1000 steps, ~4M params)",
    # === Scale ===
    "max_steps": 1000,
    "batch_size": 32,
    "seq_len": 256,
    # Token budget: 1000 * 32 * 256 = 8,192,000 tokens
    # === V7.1 Architecture (Smoke = reduced dimensions) ===
    "d_model": 256,
    "d_concept": 512,           # HebbianKernel concept space dim (reduced for smoke)
    # Hierarchy
    "n_theta": 2,               # 2 theta cycles
    "t_gamma": 2,               # 2 gamma steps per cycle → 4 kernel steps total
    # Theta attention
    "n_heads": 4,
    "d_head": 64,
    "d_ff_theta": 512,          # Explicit (not auto-sized) for smoke
    # Halting
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    # Conv
    "kernel_radius": 1,
    # Readout
    "mos_components": 1,
    "tie_embeddings": True,
    # Loss
    "lambda_reg": 0.01,
    "lambda_aux": 0.05,
    # Gradient
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 100,
    "eval_batches": 20,
    "generate_interval": 250,
    "save_interval": 500,
    "dataset": "tinystories",
}

AENCA_V7_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": "AENCA v7.1 on The Pile — 250M tokens, ~162M params, BDH-inspired Hebbian NCA",
    # Batch: 4 × 1024 × 16 = 65536 tokens/step (matches GPT-2)
    "batch_size": 4,
    "grad_accum_steps": 16,
    # === V7.1 Architecture — FLOP & param matched to GPT-2 ===
    # BDH-inspired: HebbianKernel + HebbianFFN replace SlimKernel + CSS + SwiGLU
    # FLOP analysis: 4θ×2γ at 162M params ≈ 1.00× GPT-2
    # Effective depth: 4 × (2 kernel + 1 attn+FFN) = 12 ops (= GPT-2's 12 layers)
    "d_model": 768,
    "d_concept": 2048,          # HebbianKernel concept space dim
    "max_seq_len": 1024,
    # Hierarchy: 4θ × 2γ = 8 kernel steps + 4 attention layers
    "n_theta": 4,
    "t_gamma": 2,
    # Theta attention: 12 heads × 64 dim = 768
    "n_heads": 12,
    "d_head": 64,
    "d_ff_theta": 0,            # Auto-size to fill 162M param budget
    "target_params": 162_000_000,
    # Conv
    "kernel_radius": 2,
    # Halting (adaptive compute)
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    # Readout
    "mos_components": 1,
    "tie_embeddings": False,    # Untied for fair comparison with GPT-2
    # Loss (validated stable in smoke test)
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
}

AENCA_V7_PILE_2B = {
    **_PILE_2B_COMMON,
    "description": "AENCA v7.1 on The Pile — 2B tokens, ~162M params, BDH-inspired Hebbian NCA",
    # Batch: 4 × 1024 × 16 = 65536 tokens/step → 2B / 65536 = 30517 steps
    "batch_size": 4,
    "grad_accum_steps": 16,
    "max_steps": 30517,
    # === V7.1 Architecture — identical to 250M (FLOP & param matched to GPT-2) ===
    "d_model": 768,
    "d_concept": 2048,
    "max_seq_len": 1024,
    # Hierarchy: 4θ × 2γ = 8 kernel steps + 4 attention layers (≈1.00× GPT-2 FLOPs)
    "n_theta": 4,
    "t_gamma": 2,
    # Theta attention: 12 heads × 64 dim = 768
    "n_heads": 12,
    "d_head": 64,
    "d_ff_theta": 0,               # Auto-size to fill 162M param budget → 7536
    "target_params": 162_000_000,
    # Conv
    "kernel_radius": 2,
    # Halting (PonderNet adaptive compute)
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    # Readout
    "mos_components": 1,
    "tie_embeddings": False,        # Untied for fair comparison with GPT-2
    # Loss (same as 250M — validated stable)
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
    # Disk safety: lower threshold (17.9 GB available, checkpoints ~0.6 GB each)
    "min_free_disk_gb": 5,
}

# ═══════════════════════════════════════════════════════════════════════
# V8: Growing Cellular Automata (Spec: AENCA_V8_GROWING_CELLULAR_AUTOMATA.md)
# ═══════════════════════════════════════════════════════════════════════

SMOKE_V8 = {
    "description": "Smoke test — validate V8 Growing NCA training (500 steps, CPU-safe)",
    # === Scale (minimal — designed to run on CPU alongside active GPU training) ===
    "max_steps": 500,
    "batch_size": 2,
    "seq_len": 128,
    # Token budget: 500 * 2 * 128 = 128,000 tokens
    # === V8 Architecture (Smoke = reduced dimensions for <250K params) ===
    "d_model": 64,
    "d_concept": 128,           # HebbianKernel concept space dim (reduced)
    # Hierarchy (Spec Sec 13: two-timescale loop)
    "n_theta": 2,               # 2 theta cycles
    "t_gamma": 2,               # 2 gamma steps per cycle → 4 kernel steps total
    # Theta attention
    "n_heads": 2,
    "d_head": 32,
    "d_ff_theta": 128,          # Explicit (not auto-sized) for smoke
    # Alpha channel & alive masking (Spec Sec 5-6)
    "tau_alpha": 0.1,
    # Settlement (Spec Sec 10)
    "settlement_beta": 0.9,
    "tau_s": 0.05,
    "settlement_eta": 0.1,
    "tau_c": 0.5,
    "tau_w": 0.5,
    # Stochastic masking (Spec Sec 9)
    "p_update": 0.5,
    # Conv
    "kernel_radius": 1,
    # Halting (Spec Sec 16.3)
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    # Readout
    "mos_components": 1,
    "tie_embeddings": False,    # Separate readout for tiny model
    # === V8 Loss (Spec Sec 16 — all 6 terms) ===
    # Permanence loss (Spec Sec 16.4)
    "lambda_perm": 0.1,
    # Edge-of-chaos (Spec Sec 16.6)
    "enable_edge_chaos": True,
    "lambda_edge": 0.1,
    "edge_lambda_plus": 0.05,
    "edge_lambda_minus": -0.05,
    "lyapunov_eps": 1e-4,
    "edge_batch_fraction": 0.5,   # 50% of tiny batch = 1 example
    # Spectral entropy (Spec Sec 19)
    "enable_spectral_entropy": True,
    "alpha_sv": 0.01,
    # Regeneration (Spec Sec 16.5, 18 — all 3 protocols)
    "enable_regen": True,
    "p_regen": 1.0,              # Always test regen in smoke
    "sigma_corrupt": 0.1,
    "T_r": 2,
    # Deep supervision (Spec Sec 16.7)
    "lambda_aux": 0.5,
    # State regularization (replaced by permanence)
    "lambda_reg": 0.0,
    # Pool-based training (Spec Sec 17 — disabled for smoke, needs more steps)
    "enable_pool": False,
    # Gradient
    "lambda_agc": 0.02,
    "checkpoint_interval": 4,
    "dropout": 0.0,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 25,
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 50,
    "eval_batches": 5,
    "generate_interval": 100,
    "save_interval": 100,
    "max_seq_len": 128,
    "dataset": "tinystories",
}

# ── Strategy F: Balanced Growth — Mini Smoke (2000 steps, CPU-safe) ──────
# Preserves Strategy F hierarchy (θ=6, γ=3) and proportions (dc/d≈3.33)
# at mini scale (d=128) for architecture validation.
#
# Strategy F rationale (see _v8_scaling_analysis.py):
#   θ=6 cycles × γ=3 steps = 18 shared kernel apps → eff depth 24
#   Enough gamma for settlement trajectories to develop
#   6 halting checkpoints for meaningful adaptive compute
#   Wider concept space (dc/d=3.38) → richer Class IV dynamics
#
# Param budget: ~4.4M (28.8% embed, 6.3% kernel, 35.4% theta, 0.8% conv)
# Token budget: 2000 × 2 × 256 = 1,024,000 tokens
# FLOPs/fwd:    ~3.5 GFLOPs (74% gamma, 26% theta)
SMOKE_V8_F = {
    "description": "Strategy F smoke — balanced growth V8 (2000 steps, CPU-safe)",
    # === Scale ===
    "max_steps": 2000,
    "batch_size": 2,
    "seq_len": 256,
    # === V8 Architecture (Strategy F proportions at mini scale) ===
    "d_model": 128,
    "d_concept": 432,           # dc/d = 3.38 (prod: 3.33)
    # Hierarchy: Strategy F's defining feature — θ=6, γ=3
    "n_theta": 6,               # 6 theta cycles (6 global integration points)
    "t_gamma": 3,               # 3 gamma steps per cycle → 18 kernel apps total
    # Theta attention
    "n_heads": 4,               # d_head=32 (prod: d_head=64, n_heads=12)
    "d_head": 32,
    "d_ff_theta": 512,          # 4× d_model (prod: 9.19× from budget fill)
    # Alpha channel & alive masking (Spec Sec 5-6)
    "tau_alpha": 0.1,
    # Settlement (Spec Sec 10) — same thresholds as production
    "settlement_beta": 0.9,
    "tau_s": 0.05,
    "settlement_eta": 0.1,
    "tau_c": 0.5,
    "tau_w": 0.5,
    # Stochastic masking (Spec Sec 9)
    "p_update": 0.5,
    # Conv
    "kernel_radius": 1,
    # Halting — 6 theta cycles gives meaningful adaptive compute
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    # Readout
    "mos_components": 1,
    "tie_embeddings": False,
    # === V8 Loss (all 6 terms) ===
    "lambda_perm": 0.1,
    "enable_edge_chaos": True,
    "lambda_edge": 0.1,
    "edge_lambda_plus": 0.05,
    "edge_lambda_minus": -0.05,
    "lyapunov_eps": 1e-4,
    "edge_batch_fraction": 0.5,
    "enable_spectral_entropy": True,
    "alpha_sv": 0.01,
    "enable_regen": True,
    "p_regen": 0.3,             # 30% regen chance (more realistic than 100%)
    "sigma_corrupt": 0.1,
    "T_r": 3,                   # Match t_gamma for recovery
    "lambda_aux": 0.5,
    "lambda_reg": 0.0,
    # Pool-based training — enabled! 2000 steps is enough
    "enable_pool": True,
    "pool_size": 64,            # Small pool for smoke
    # Gradient
    "lambda_agc": 0.02,
    "checkpoint_interval": 6,   # Match n_theta for gradient checkpointing
    "dropout": 0.0,
    # === Optimizer ===
    "learning_rate": 2e-4,      # Scaled down from 3e-4 (larger model)
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 50,         # 2.5% of training
    # === Logging & Checkpointing ===
    "log_interval": 20,
    "eval_interval": 100,
    "eval_batches": 5,
    "generate_interval": 500,
    "save_interval": 500,
    "max_seq_len": 256,
    "dataset": "tinystories",
}

# ── Strategy F: Production 160M (for reference / future use) ─────────
# d=768, dc=2560, θ=6, γ=3, d_ff_theta=0 (auto → 7058), V=50304, L=1024
# Total ~162M params, eff depth 24, 18 kernel apps, 18432× reuse
PRODUCTION_V8_F = {
    "description": "Strategy F production — 160M balanced growth V8",
    "max_steps": 50_000,
    "batch_size": 32,
    "seq_len": 1024,
    "grad_accum_steps": 4,
    "d_model": 768,
    "d_concept": 2560,
    "n_theta": 6,
    "t_gamma": 3,
    "n_heads": 12,
    "d_head": 64,
    "d_ff_theta": 0,            # Auto-size to fill 162M budget
    "target_params": 162_000_000,
    "tau_alpha": 0.1,
    "settlement_beta": 0.9,
    "tau_s": 0.05,
    "settlement_eta": 0.1,
    "tau_c": 0.5,
    "tau_w": 0.5,
    "p_update": 0.5,
    "kernel_radius": 2,
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    "mos_components": 1,
    "tie_embeddings": True,
    "vocab_size": 50304,
    "max_seq_len": 1024,
    "lambda_perm": 0.1,
    "enable_edge_chaos": True,
    "lambda_edge": 0.1,
    "edge_lambda_plus": 0.05,
    "edge_lambda_minus": -0.05,
    "lyapunov_eps": 1e-4,
    "edge_batch_fraction": 0.25,
    "enable_spectral_entropy": True,
    "alpha_sv": 0.01,
    "enable_regen": True,
    "p_regen": 0.1,
    "sigma_corrupt": 0.1,
    "T_r": 3,
    "lambda_aux": 0.5,
    "lambda_reg": 0.0,
    "enable_pool": True,
    "pool_size": 1024,
    "lambda_agc": 0.02,
    "checkpoint_interval": 6,
    "dropout": 0.0,
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 2000,
    "beta2": 0.95,
    "log_interval": 10,
    "eval_interval": 500,
    "eval_batches": 20,
    "generate_interval": 2500,
    "save_interval": 5000,
    "dataset": "the_pile",
    "dataset_hf_name": "monology/pile-uncopyrighted",
    "dataset_cache_name": "pile_neox50k_tokens_2b.bin",
    "max_dataset_tokens": 2_000_000_000,
    "tokenizer_name": "pretrained",
    "tokenizer_pretrained_name": "EleutherAI/gpt-neox-20b",
    "tokenizer_vocab_size": 50304,
}

# ═══════════════════════════════════════════════════════════════════════════
# ── Morphogenesis: CA-Native 250B-Scalable Architecture ─────────────────
# ═══════════════════════════════════════════════════════════════════════════
#
# Design philosophy: CELLULAR AUTOMATA FIRST.
#
# Standard transformers repeat identical layers (attention + FFN).
# Morphogenesis inverts the ratio: a massive SHARED Hebbian kernel
# (the CA's "genome") is reused millions of times per forward pass,
# while theta blocks provide periodic global coordination.
#
# At 250B scale with L=8192, n_θ=64, t_γ=4:
#   - 256 gamma steps × 8192 positions = 2,097,152 kernel applications/sample
#   - A 5.4B shared kernel amortized over 2.1M uses = 2.6K FLOPs per param
#   - Settlement (Theorem 22.1): total gamma FLOPs bounded INDEPENDENT of K
#   - Inference: settled tokens need 0 recomputation → O(L·d) vs O(L·N·d) KV
#
# Scaling recipe preserving CA ratios across 3 scales:
#
#   Scale     d_model  d_concept  n_θ   t_γ   Params   Kernel Reuse
#   ─────────────────────────────────────────────────────────────────
#   Smoke       128      512      8     4    ~4.5M      8,192×
#   Pile-250M   768     3072     12     4    ~162M    393,216×
#   250B      16384    65536     64     4    ~250B  2,097,152×
#
# Key ratios (preserved across scales):
#   d_concept / d_model  = 4.0  (richer kernel than Strategy F's 3.3)
#   d_ff_θ / d_model     ≈ 3.4  (modern transformer sweet spot)
#   t_gamma              = 4    (more growth time per theta → deeper CA dynamics)
#   kernel_radius        = 2/2/4 (wider perception at scale)
#   pool as % of seq     = 25%  (cross-batch working memory)
#
# The 250B model has 320 effective sequential steps (vs GPT-3's 96 layers),
# but settlement sparsity means average active compute ≈ 55% of dense cost,
# and inference memory is ~128× smaller than equivalent transformer KV cache.
# ═══════════════════════════════════════════════════════════════════════════

# ── Morphogenesis: Smoke Test (CPU, ~4.5M params, 2000 steps) ────────────
SMOKE_MORPHO = {
    "description": "Morphogenesis smoke — CA-native 250B-scalable architecture (CPU-safe)",
    # === Scale ===
    "max_steps": 2000,
    "batch_size": 2,
    "seq_len": 256,
    # === Architecture (250B ratios at mini scale) ===
    "d_model": 128,
    "d_concept": 512,           # 4:1 ratio (250B target: 65536/16384 = 4:1)
    "n_theta": 8,               # 8 theta cycles (250B: 64 — 8× deeper)
    "t_gamma": 4,               # 4 gamma/cycle → 32 kernel apps (250B: 256)
    "n_heads": 4,
    "d_head": 32,               # 4×32 = 128 = d_model
    "d_ff_theta": 432,          # 3.375× d_model (250B target: ~3.35×)
    # === Settlement / Growth Dynamics ===
    "tau_alpha": 0.1,
    "settlement_beta": 0.9,
    "tau_s": 0.05,
    "settlement_eta": 0.1,
    "tau_c": 0.5,
    "tau_w": 0.5,
    "p_update": 0.5,
    "kernel_radius": 2,
    # === Halting ===
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    # === Readout ===
    "mos_components": 1,
    "tie_embeddings": False,
    # === 6-Term Loss ===
    "lambda_perm": 0.1,
    "enable_edge_chaos": True,
    "lambda_edge": 0.1,
    "edge_lambda_plus": 0.05,
    "edge_lambda_minus": -0.05,
    "lyapunov_eps": 1e-4,
    "edge_batch_fraction": 0.5,
    "enable_spectral_entropy": True,
    "alpha_sv": 0.01,
    "enable_regen": True,
    "p_regen": 0.3,
    "sigma_corrupt": 0.1,
    "T_r": 4,                   # Match t_gamma for recovery horizon
    "lambda_aux": 0.5,
    "lambda_reg": 0.0,
    # === Pool ===
    "enable_pool": True,
    "pool_size": 64,
    # === Gradient ===
    "lambda_agc": 0.02,
    "checkpoint_interval": 8,   # Match n_theta
    "dropout": 0.0,
    # === Optimizer ===
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging ===
    "log_interval": 20,
    "eval_interval": 100,
    "eval_batches": 5,
    "generate_interval": 500,
    "save_interval": 500,
    "max_seq_len": 256,
    "dataset": "tinystories",
}

# ── Morphogenesis: Pile 250M Benchmark (~162M params, head-to-head) ──────
# Direct comparison vs GPT-2 162M on 250M Pile tokens.
# Same data pipeline, tokenizer, total tokens, effective batch size.
#
# Parameter budget (auto d_ff_theta to fill 162M):
#   Embed (tied):     50304 × 768               = 38.6M
#   CausalConv:       769² × 3                  =  1.8M
#   Hebbian kernel:   5 × 3072 × 769            = 11.8M
#   Attention (×12):  4 × 768² × 12             = 28.3M
#   FFN (×12):        3 × 768 × d_ff_θ × 12     ≈ 80.5M  (d_ff_θ ≈ 2,912)
#   Halt + norm:                                 ≈  1.0M
#   Total:                                       ≈ 162.0M
#
# Effective depth: 12 × (4 + 1) = 60 (vs GPT-2's 12)
# Kernel reuse: 48 gamma × 1024 pos × 8 batch = 393,216× per step
# Settlement advantage: ~1.4× FLOP reduction in late training
MORPHO_V8_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": "Morphogenesis V8 on The Pile — 250M tok, ~162M params, CA-native",
    # Batch: 4 × 1024 × 16 = 65536 tokens/step (matches GPT-2 baseline)
    "batch_size": 4,
    "grad_accum_steps": 16,
    # === Architecture ===
    "d_model": 768,
    "d_concept": 3072,          # 4:1 — richer CA kernel than Strategy F's 3.3
    "n_theta": 12,              # 12 theta cycles (2× GPT-2 layer count in attention)
    "t_gamma": 4,               # 4 gamma/cycle → 48 shared kernel apps
    "n_heads": 12,
    "d_head": 64,
    "d_ff_theta": 0,            # Auto-size to fill 162M budget
    "target_params": 162_000_000,
    "kernel_radius": 2,
    "max_seq_len": 1024,
    # === Settlement / Growth ===
    "tau_alpha": 0.1,
    "settlement_beta": 0.9,
    "tau_s": 0.05,
    "settlement_eta": 0.1,
    "tau_c": 0.5,
    "tau_w": 0.5,
    "p_update": 0.5,
    # === Halting (12 checkpoints for fine-grained adaptive depth) ===
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    # === Readout ===
    "mos_components": 1,
    "tie_embeddings": True,
    "vocab_size": 50304,
    # === 6-Term Loss ===
    "lambda_perm": 0.1,
    "enable_edge_chaos": True,
    "lambda_edge": 0.1,
    "edge_lambda_plus": 0.05,
    "edge_lambda_minus": -0.05,
    "lyapunov_eps": 1e-4,
    "edge_batch_fraction": 0.25,    # 25% of batch for Lyapunov
    "enable_spectral_entropy": True,
    "alpha_sv": 0.01,
    "enable_regen": True,
    "p_regen": 0.1,
    "sigma_corrupt": 0.1,
    "T_r": 4,
    "lambda_aux": 0.5,
    "lambda_reg": 0.0,
    # === Pool (cross-batch working memory) ===
    "enable_pool": True,
    "pool_size": 2048,              # 2× seq_len worth of persistent state
    # === Gradient ===
    "lambda_agc": 0.02,
    "checkpoint_interval": 12,      # Match n_theta
    "dropout": 0.0,
    # === Optimizer (tuned for V8's deeper effective depth) ===
    "learning_rate": 3e-4,          # Higher than GPT-2's 2e-4 — settlement stabilizes
    "beta2": 0.95,                  # More aggressive for large eff. depth
}

# ── Morphogenesis: 250B Full-Scale Target (Multi-Node Cluster) ───────────
# The definitive CA-native language model at frontier scale.
#
# Architecture rationale:
#   d_model = 16384     Same width as Llama 405B, rich cell state
#   d_concept = 65536   4:1 kernel — the CA genome, used 2.1M times/sample
#   n_theta = 64        64 global coordination checkpoints
#   t_gamma = 4         4 local growth steps between each checkpoint
#   n_heads = 128       128 attention heads × d_head=128 = d_model
#   kernel_radius = 4   9-position causal perception field
#   max_seq_len = 8192  Long-context with settlement-based caching
#
# Parameter decomposition:
#   Embedding (tied):   128256 × 16384            =  2.10B  (0.84%)
#   CausalConv:         16385² × 5                =  1.34B  (0.54%)
#   Hebbian kernel:     5 × 65536 × 16385         =  5.37B  (2.15%) ← THE GENOME
#   Attention (×64):    4 × 16384² × 64           = 68.72B (27.49%)
#   FFN (×64):          3 × 16384 × d_ff × 64     ≈172.47B (68.99%)
#   d_ff_θ auto-sizes to:                         ≈ 54,826  (3.35:1)
#   Total:                                        ≈250.00B
#
# CA advantages vs equivalent dense transformer:
#   Kernel reuse:    5.37B params × 2,097,152 uses = 11.26T effective FLOPs from 2% of params
#   Settlement:      Theorem 22.1 → gamma FLOPs bounded O(L·C_Φ/ρ) regardless of K
#   Inference mem:   O(L·d) settled state vs O(L·N·d) KV cache → 128× reduction
#   Sparse compute:  50% avg frontier → ~1.5× gamma FLOP savings
#   Gradient flow:   Settled cells = identity Jacobians → perfect gradient highways
#
# Training compute (Chinchilla-optimal):
#   Tokens: 20 × 250B = 5T tokens
#   Effective batch: 4M tokens/step → 1.25M steps
#   Hardware: ~4096 H100 GPUs for ~30 days
MORPHOGENESIS_250B = {
    "description": "Morphogenesis 250B — frontier CA-native language model",
    # === Scale ===
    "max_steps": 1_250_000,
    "batch_size": 64,
    "seq_len": 8192,
    "grad_accum_steps": 8,          # Effective batch: 64×8192×8 = 4.19M tok/step
    # === Architecture ===
    "d_model": 16384,
    "d_concept": 65536,             # 4:1 ratio — massive shared kernel
    "n_theta": 64,                  # 64 theta cycles
    "t_gamma": 4,                   # 4 gamma/cycle → 256 kernel apps total
    "n_heads": 128,
    "d_head": 128,                  # 128×128 = 16384 = d_model
    "d_ff_theta": 0,                # Auto → ~54,826 (3.35:1 ratio)
    "target_params": 250_000_000_000,
    "kernel_radius": 4,             # Wide 9-position causal perception
    "max_seq_len": 8192,
    # === Settlement / Growth ===
    "tau_alpha": 0.1,
    "settlement_beta": 0.9,
    "tau_s": 0.05,
    "settlement_eta": 0.1,
    "tau_c": 0.5,
    "tau_w": 0.5,
    "p_update": 0.5,
    # === Halting ===
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.005,           # Gentler halting pressure at scale
    # === Readout ===
    "mos_components": 1,
    "tie_embeddings": True,
    "vocab_size": 128256,           # Modern tokenizer (Llama 3 class)
    # === 6-Term Loss ===
    "lambda_perm": 0.05,            # Lighter permanence — settlement emerges naturally at scale
    "enable_edge_chaos": True,
    "lambda_edge": 0.05,
    "edge_lambda_plus": 0.05,
    "edge_lambda_minus": -0.05,
    "lyapunov_eps": 1e-4,
    "edge_batch_fraction": 0.125,   # 12.5% — amortize over huge batch
    "enable_spectral_entropy": True,
    "alpha_sv": 0.005,
    "enable_regen": True,
    "p_regen": 0.05,                # 5% regen frequency at scale
    "sigma_corrupt": 0.1,
    "T_r": 4,
    "lambda_aux": 0.3,              # Lighter deep supervision at 64 layers
    "lambda_reg": 0.0,
    # === Pool (massive cross-batch state memory) ===
    "enable_pool": True,
    "pool_size": 16384,             # 2× seq_len — 16K persistent growth states
    # === Gradient ===
    "lambda_agc": 0.01,
    "checkpoint_interval": 4,       # Checkpoint every 4 theta cycles (16 groups)
    "dropout": 0.0,
    # === Optimizer (frontier-scale schedule) ===
    "learning_rate": 1.5e-4,        # Scales as ~1/√d_model from smaller configs
    "weight_decay": 0.1,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 2000,
    # === Logging ===
    "log_interval": 10,
    "eval_interval": 1000,
    "eval_batches": 50,
    "generate_interval": 10000,
    "save_interval": 10000,
    # === Data (Chinchilla-optimal: 20× params) ===
    "dataset": "the_pile",
    "dataset_hf_name": "monology/pile-uncopyrighted",
    "tokenizer_name": "pretrained",
    "tokenizer_pretrained_name": "EleutherAI/gpt-neox-20b",
    "tokenizer_vocab_size": 128256,
}

# ══════════════════════════════════════════════════════════════════════════
# V9 — Wolfram Cellular Automata (True Parallel CA, ~45K tok/s target)
# ══════════════════════════════════════════════════════════════════════════

SMOKE_V9 = {
    "description": "Smoke test — V9 Wolfram CA (500 steps, CPU-safe, tiny dims)",
    # === Scale ===
    "max_steps": 500,
    "batch_size": 2,
    "seq_len": 128,
    # === V9 Architecture (tiny for smoke) ===
    "d_model": 256,
    "d_concept": 1024,              # Explicit (not auto-sized) for smoke
    "n_steps": 3,
    "conv_kernel_size": 3,
    "max_seq_len": 128,
    # === Vocab (TinyStories) ===
    "vocab_size": 10000,
    "tie_embeddings": True,
    # === Loss ===
    "lambda_perm": 0.01,
    "lambda_spectral": 0.001,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging ===
    "log_interval": 10,
    "eval_interval": 100,
    "eval_batches": 5,
    "generate_interval": 250,
    "save_interval": 250,
    # === Data ===
    "dataset": "tinystories",
    "dropout": 0.0,
    "checkpoint_interval": 0,       # No gradient checkpointing for tiny model
}

V9_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": "V9 Wolfram CA on The Pile — 250M tokens, ~162M params, GPU benchmark",
    # === V9 Architecture (full scale) ===
    "d_model": 2048,
    "d_concept": 0,                 # Auto-size from target_params
    "n_steps": 3,
    "conv_kernel_size": 3,
    "max_seq_len": 1024,
    "target_params": 162_000_000,
    # === Vocab ===
    "tie_embeddings": True,
    # === Loss ===
    "lambda_perm": 0.01,
    "lambda_spectral": 0.001,
    # === Training ===
    "dropout": 0.0,
    "checkpoint_interval": 0,       # No checkpointing — only 9 GEMMs
}

# ══════════════════════════════════════════════════════════════════════
# V10 — Persistent Growth NCA (True CA: tiny rule, 64 steps, settlement)
# ══════════════════════════════════════════════════════════════════════

SMOKE_V10 = {
    "description": "Smoke test — V10 Persistent Growth NCA (500 steps, CPU-safe)",
    # === Scale ===
    "max_steps": 500,
    "batch_size": 2,
    "seq_len": 128,
    # === V10 Architecture (tiny for smoke) ===
    "d_model": 256,
    "d_rule": 128,
    "n_steps": 16,                  # Fewer steps for CPU smoke
    "n_scales": 4,                  # 4 dilation scales
    "conv_kernel_size": 3,
    "max_seq_len": 128,
    # === Settlement ===
    "settlement_beta": 0.9,
    "settlement_tau": 0.05,
    "settlement_eta": 0.1,
    # === Vocab (TinyStories) ===
    "vocab_size": 10000,
    "tie_embeddings": True,
    # === Loss ===
    "lambda_perm": 0.01,
    "lambda_spectral": 0.001,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging ===
    "log_interval": 10,
    "eval_interval": 100,
    "eval_batches": 5,
    "generate_interval": 250,
    "save_interval": 250,
    # === Data ===
    "dataset": "tinystories",
    "dropout": 0.0,
    "checkpoint_interval": 0,       # No checkpointing for tiny model
}

V10_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": "V10 Persistent Growth NCA on The Pile — 250M tokens, ~110M params, 64 steps",
    # === V10 needs smaller micro-batches (64 deep CA steps eat VRAM) ===
    "batch_size": 2,                # Down from 8 — 64 CA steps are deep
    "grad_accum_steps": 32,         # Up from 8 — same effective batch (65536 tok/step)
    # === V10 Architecture (full scale) ===
    "d_model": 2048,
    "d_rule": 800,                  # Small rule: 4.9M params (Wolfram: simple!)
    "n_steps": 64,                  # Many iterations (21× more than V9)
    "n_scales": 8,                  # Cyclic [1,2,4,8,16,32,64,128]
    "conv_kernel_size": 3,
    "max_seq_len": 1024,
    # === Settlement (PERMANENCE — PRIMARY) ===
    "settlement_beta": 0.9,
    "settlement_tau": 0.05,
    "settlement_eta": 0.1,
    # === Vocab ===
    "tie_embeddings": True,
    # === Loss ===
    "lambda_perm": 0.01,
    "lambda_spectral": 0.001,
    # === Training ===
    "dropout": 0.0,
    "checkpoint_interval": 4,       # Frequent checkpointing — 64 deep steps
}


# ═══════════════════════════════════════════════════════════════════════════════
# V11: Morphogenetic Field NCA
# Two-field coupled CA: content z + morphogen h.
# Morphogen gates content perception → emergent routing from local-only dynamics.
# ═══════════════════════════════════════════════════════════════════════════════

SMOKE_V11 = {
    "description": "Smoke test — V11 Morphogenetic Field NCA (500 steps, CPU-safe)",
    # === Scale ===
    "max_steps": 500,
    "batch_size": 2,
    "seq_len": 128,
    # === V11 Architecture (tiny for smoke) ===
    "d_model": 256,
    "d_rule": 128,
    "d_h": 32,                      # Morphogen dimension (tiny for smoke)
    "n_steps": 16,                  # Fewer steps for CPU smoke
    "n_scales": 4,                  # 4 dilation scales
    "conv_kernel_size": 3,
    "max_seq_len": 128,
    # === Settlement ===
    "settlement_beta": 0.9,
    "settlement_tau": 0.05,
    "settlement_eta": 0.1,
    # === Vocab (TinyStories) ===
    "vocab_size": 10000,
    "tie_embeddings": True,
    # === Loss ===
    "lambda_perm": 0.01,
    "lambda_spectral": 0.001,
    # === Persistent Morphogen Scaffold (tiny for smoke) ===
    "n_scaffold_templates": 8,       # K=8 (tiny for CPU smoke)
    "scaffold_alpha": 0.3,
    "scaffold_temperature": 1.0,
    "scaffold_ema_decay": 0.99,      # Faster decay for short runs
    "scaffold_warmup_steps": 50,     # 10% of 500 steps
    "scaffold_maintenance_interval": 100,
    "scaffold_min_norm": 0.01,
    "scaffold_merge_threshold": 0.95,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging ===
    "log_interval": 10,
    "eval_interval": 100,
    "eval_batches": 5,
    "generate_interval": 250,
    "save_interval": 250,
    # === Data ===
    "dataset": "tinystories",
    "dropout": 0.0,
    "checkpoint_interval": 0,       # No checkpointing for tiny model
}

V11_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": "V11 Morphogenetic Field NCA on The Pile — 250M tokens, ~162M params, bilinear gate + scaffold",
    # === V11 needs smaller micro-batches (64 deep CA steps eat VRAM) ===
    "batch_size": 2,                # Down from 8 — 64 CA steps are deep
    "grad_accum_steps": 32,         # Up from 8 — same effective batch (65536 tok/step)
    # === V11 Architecture (162M param-matched to GPT-2) ===
    # Compact morphogen dynamics in d_h=256 (cheap MLP: ~197K).
    # BILINEAR GATE: gate = σ(SiLU(h·W_a) ⊙ h·W_b + bias)
    #   Two (256→2048) projections create quadratic forms with
    #   d_h(d_h+1)/2 = 32,896 independent patterns >> d_model=2048.
    #   Effective FULL RANK gating from compact morphogen. 13× faster
    #   morphogen path vs d_h=2048 approach.
    # Content rule 2048→8968→2048 (4.38× ratio) — massive content budget.
    # Morphogen ~1.8M (1.1%), Content rule ~55.1M (34%), Embed ~103M (64%).
    "d_model": 2048,
    "d_rule": 8968,                 # Content rule: 55.1M params, iterated 64×
    "d_h": 256,                     # Compact morphogen dynamics → bilinear lift
    "n_steps": 64,                  # 64 growth steps — full developmental depth
    "n_scales": 8,                  # Cyclic [1,2,4,8,16,32,64,128]
    "conv_kernel_size": 3,
    "max_seq_len": 1024,
    "vocab_size": 50304,            # Explicit — matches tokenizer_vocab_size
    # === Settlement (PERMANENCE — PRIMARY) ===
    "settlement_beta": 0.9,
    "settlement_tau": 0.05,
    "settlement_eta": 0.1,
    # === Vocab ===
    "tie_embeddings": True,
    # === Loss ===
    "lambda_perm": 0.01,
    "lambda_spectral": 0.001,
    # === Persistent Morphogen Scaffold (EPIGENETIC MEMORY) ===
    "n_scaffold_templates": 64,      # K=64 cell type prototypes (64KB at d_h=256)
    "scaffold_alpha": 0.3,           # h = h_fresh + 0.3 * h_scaffold
    "scaffold_temperature": 1.0,     # Softmax temperature for template selection
    "scaffold_ema_decay": 0.999,     # Slow crystallization over training
    "scaffold_warmup_steps": 200,    # ~5% of 3815 steps
    "scaffold_maintenance_interval": 500, # Prune/compress every 500 steps
    "scaffold_min_norm": 0.01,       # Dead template threshold
    "scaffold_merge_threshold": 0.95, # Cosine sim for merge
    # === Training ===
    "dropout": 0.0,
    "checkpoint_interval": 4,       # Frequent checkpointing — 64 deep steps
}

# Production-scale smoke: 162M model on Pile, 200 steps — verify VRAM/speed
SMOKE_V11_PROD = {
    **V11_PILE_250M,
    "description": "V11 production-scale smoke — 162M params, 200 steps on The Pile (CUDA)",
    "max_steps": 200,
    "eval_interval": 50,
    "eval_batches": 10,
    "generate_interval": 100,
    "save_interval": 100,
    "log_interval": 5,
    "scaffold_warmup_steps": 20,       # Earlier scaffold for short run
    "scaffold_maintenance_interval": 50, # See pruning/merge in 200 steps
}

# ── V12 MMHR: Morphogenetic Mixture of Hebbian Rules ─────────────────────────
# K=8 parallel Hebbian rules with morphogen-driven routing (MNCA §3.1-§3.2).
# Shared W_p (perception) + K independent (W_s_k, W_d_k) binding pairs.
# Morphogen expanded to d_h=512 → primary routing computation substrate.
# Diversity loss −λ_div·H(weights) prevents rule collapse.
#
# Total params at defaults: ~137M (103M embed + 34M non-embed)
#   d_rule=800: W_p(1.64M) + K=8×(W_s+W_d)(26.2M) + gate(2.1M) + morph(~4M) = ~34M
# VRAM: ~deepens V11 with K=8 rules, but einsum is GPU-batched (NOT K serial ops)
# Throughput estimate: ~14-16K tok/s on A100-80GB (vs V11 ~18K, V7.1 ~22.5K)

SMOKE_V12 = {
    "description": "Smoke test — V12 MMHR Mixture Hebbian Rules NCA (500 steps, CPU-safe)",
    # === Scale ===
    "max_steps": 500,
    "batch_size": 2,
    "seq_len": 128,
    # === V12 Architecture (tiny for smoke — CPU-runnable) ===
    "d_model": 256,
    "d_rule": 64,                   # Tiny concept space for smoke
    "d_h": 64,                      # Tiny morphogen for smoke
    "n_rules": 4,                   # Half the rules for CPU smoke
    "n_steps": 16,                  # Fewer steps for CPU
    "n_scales": 4,
    "conv_kernel_size": 3,
    "max_seq_len": 128,
    # === V12-specific ===
    "noise_sigma": 0.02,
    "route_temp_min": 0.1,
    "route_temp_max": 1.0,
    "lambda_div": 0.01,
    # === Settlement ===
    "settlement_beta": 0.9,
    "settlement_tau": 0.05,
    "settlement_eta": 0.1,
    # === Vocab (TinyStories) ===
    "vocab_size": 10000,
    "tie_embeddings": True,
    # === Loss ===
    "lambda_perm": 0.01,
    "lambda_spectral": 0.001,
    # === Scaffold (tiny for smoke) ===
    "n_scaffold_templates": 8,
    "scaffold_alpha": 0.3,
    "scaffold_temperature": 1.0,
    "scaffold_ema_decay": 0.99,
    "scaffold_warmup_steps": 50,
    "scaffold_maintenance_interval": 100,
    "scaffold_min_norm": 0.01,
    "scaffold_merge_threshold": 0.95,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging ===
    "log_interval": 10,
    "eval_interval": 100,
    "eval_batches": 5,
    "generate_interval": 250,
    "save_interval": 250,
    # === Data ===
    "dataset": "tinystories",
    "dropout": 0.0,
    "checkpoint_interval": 0,       # No checkpointing for tiny smoke
}

V12_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": (
        "V12 MMHR NCA on The Pile — 250M tokens, ~137M params, "
        "K=8 mixture Hebbian rules, d_h=512 morphogen routing, diversity loss"
    ),
    # === Micro-batch (same as V11 — deep CA with K=8 batched einsum) ===
    "batch_size": 2,
    "grad_accum_steps": 32,         # Same effective batch (65536 tok/step)
    # === V12 Architecture (~137M params) ===
    # Concept space d_rule=800: W_p(1.64M) shared + K=8×(W_s+W_d)(26.2M) binding.
    # Morphogen d_h=512: routing substrate with 131K independent bilinear patterns.
    # K=8 rules are GPU-batched via einsum (no Python loop over K=8 during forward).
    # Diversity loss prevents rule collapse; noise sigma=0.02 injects MNCA variability.
    "d_model": 2048,
    "d_rule": 800,                  # Concept space: ~137M total at K=8, d_h=512
    "d_h": 512,                     # Expanded morphogen routing substrate
    "n_rules": 8,                   # K=8 parallel Hebbian rules
    "n_steps": 64,
    "n_scales": 8,
    "conv_kernel_size": 3,
    "max_seq_len": 1024,
    "vocab_size": 50304,
    # === V12-specific training ===
    "noise_sigma": 0.02,            # MNCA §3.2 intrinsic noise per rule
    "route_temp_min": 0.1,          # Temperature floor (sharp routing at step 63)
    "route_temp_max": 1.0,          # Temperature ceiling (uniform at step 0)
    "lambda_div": 0.01,             # Diversity loss weight
    # === Settlement ===
    "settlement_beta": 0.9,
    "settlement_tau": 0.05,
    "settlement_eta": 0.1,
    # === Vocab ===
    "tie_embeddings": True,
    # === Loss ===
    "lambda_perm": 0.01,
    "lambda_spectral": 0.001,
    # === Scaffold (same as V11, auto-scales to d_h=512) ===
    "n_scaffold_templates": 64,     # K=64 morphogen prototypes (128KB at d_h=512)
    "scaffold_alpha": 0.3,
    "scaffold_temperature": 1.0,
    "scaffold_ema_decay": 0.999,
    "scaffold_warmup_steps": 200,
    "scaffold_maintenance_interval": 500,
    "scaffold_min_norm": 0.01,
    "scaffold_merge_threshold": 0.95,
    # === Training ===
    "dropout": 0.0,
    "checkpoint_interval": 0,       # No grad checkpointing — use VRAM (saves 2x backward)
}

# Production-scale smoke: ~137M V12 on Pile, 200 steps
SMOKE_V12_PROD = {
    **V12_PILE_250M,
    "description": "V12 MMHR production-scale smoke — ~137M params, 200 steps on The Pile (CUDA)",
    "max_steps": 200,
    "eval_interval": 50,
    "eval_batches": 10,
    "generate_interval": 100,
    "save_interval": 100,
    "log_interval": 5,
    "scaffold_warmup_steps": 20,
    "scaffold_maintenance_interval": 50,
}

# ── V12 Fast: max throughput variant ─────────────────────────────────────────
# Architectural changes vs pile_250m_v12:
#   n_steps  64 → 32   halves sequential depth (biggest single win)
#   n_rules   8 →  4   halves K-batched einsum cost (W_s + W_d)
#   d_rule  800 → 512  smaller per-rule binding matrices (-36% W_s/W_d size)
#   d_h     512 → 256  halves gate FLOPS (2×256×2048 vs 2×512×2048) and morph MLP
#
# d_model stays at 2048 — embeddings are 103M params but have ZERO per-step cost.
# VRAM drops: activation memory ∝ n_steps × B × L × d_model (halved by n_steps=32)
# Expected throughput: ~4-6× vs pile_250m_v12 (~3000-5000 tok/s on A100)
#
# Non-embed params: ~11M (vs ~32M), Grand total: ~114M
#   shared_wp:   512 × 2048  =  1.05M
#   binding K=4: 4×2×2048×512 =  8.39M  (vs 26.2M at K=8, d_rule=800)
#   gate (d_h=256): 2×256×2048 + 2048 =  1.05M
#   morph rule:  2×256×256+256+256²  =  0.20M
#   morph seed:  2048×256             =  0.52M
V12_PILE_250M_FAST = {
    **_PILE_250M_COMMON,
    "description": (
        "V12 MMHR FAST on The Pile — 250M tokens, ~114M params, "
        "n_steps=32, K=4 rules, d_rule=512, d_h=256 — max throughput"
    ),
    "batch_size": 2,
    "grad_accum_steps": 32,
    # === V12 Fast Architecture (~114M params) ===
    "d_model": 2048,
    "d_rule": 512,
    "d_h": 256,
    "n_rules": 4,
    "n_steps": 32,
    "n_scales": 8,
    "conv_kernel_size": 3,
    "max_seq_len": 1024,
    "vocab_size": 50304,
    # === V12-specific training ===
    "noise_sigma": 0.02,
    "route_temp_min": 0.1,
    "route_temp_max": 1.0,
    "lambda_div": 0.01,
    # === Settlement ===
    "settlement_beta": 0.9,
    "settlement_tau": 0.05,
    "settlement_eta": 0.1,
    # === Vocab ===
    "tie_embeddings": True,
    # === Loss ===
    "lambda_perm": 0.01,
    "lambda_spectral": 0.001,
    # === Scaffold (auto-scales to d_h=256) ===
    "n_scaffold_templates": 64,
    "scaffold_alpha": 0.3,
    "scaffold_temperature": 1.0,
    "scaffold_ema_decay": 0.999,
    "scaffold_warmup_steps": 200,
    "scaffold_maintenance_interval": 500,
    "scaffold_min_norm": 0.01,
    "scaffold_merge_threshold": 0.95,
    # === Training ===
    "dropout": 0.0,
    "checkpoint_interval": 0,
    "min_free_disk_gb": 5,
}

# ═══════════════════════════════════════════════════════════════════════════
# V13 — Morphogenetic Attention (morphogen field → Q/K gate steering)
# ═══════════════════════════════════════════════════════════════════════════

SMOKE_V13 = {
    "description": "Smoke test — V13 Morphogenetic Attention (~50k tokens, tiny dims)",
    # === Scale ===
    "max_steps": 1000,
    "batch_size": 32,
    "seq_len": 256,
    # === V13 Architecture (scaled down for smoke) ===
    "d_model": 256,
    "d_h": 32,
    "n_rules": 2,
    "n_morph_steps": 2,
    "n_theta": 2,
    "n_heads": 4,
    "d_head": 64,
    "d_ff": 512,
    "morph_dilations": (1, 2),
    "morph_gate_bias": 3.0,
    "morph_conv_kernel": 3,
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    "mos_components": 1,
    "tie_embeddings": True,
    "lambda_reg": 0.01,
    "lambda_aux": 0.05,
    "lambda_div": 0.01,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 100,
    "eval_batches": 20,
    "generate_interval": 250,
    "save_interval": 500,
    "dataset": "tinystories",
    "checkpoint_interval": 0,
    "min_free_disk_gb": 1,
}

V13_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": "V13 Morphogenetic Attention on The Pile — 250M tokens, ~162M params",
    "batch_size": 4,
    "grad_accum_steps": 16,
    # === V13 Architecture (full scale) ===
    "d_model": 768,
    "d_h": 128,
    "n_rules": 4,
    "n_morph_steps": 4,
    "n_theta": 4,
    "n_heads": 12,
    "d_head": 64,
    "d_ff": 0,  # auto-compute to fill 162M target
    "target_params": 162_000_000,
    "morph_dilations": (1, 2, 4, 8),
    "morph_gate_bias": 3.0,
    "morph_conv_kernel": 3,
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    "mos_components": 1,
    "tie_embeddings": False,
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_div": 0.01,
    "dropout": 0.0,
    "checkpoint_interval": 4,
    "min_free_disk_gb": 5,
}

V13_PILE_2B = {
    **_PILE_2B_COMMON,
    "description": "V13 Morphogenetic Attention on The Pile — 2B tokens, ~162M params (vs V7.1 PPL 15.68)",
    "batch_size": 4,
    "grad_accum_steps": 16,
    # === V13 Architecture (same as 250M) ===
    "d_model": 768,
    "d_h": 128,
    "n_rules": 4,
    "n_morph_steps": 4,
    "n_theta": 4,
    "n_heads": 12,
    "d_head": 64,
    "d_ff": 0,
    "target_params": 162_000_000,
    "morph_dilations": (1, 2, 4, 8),
    "morph_gate_bias": 3.0,
    "morph_conv_kernel": 3,
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    "mos_components": 1,
    "tie_embeddings": False,
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_div": 0.01,
    "dropout": 0.0,
    "checkpoint_interval": 4,
    "min_free_disk_gb": 5,
}


SMOKE_V14 = {
    "description": "Smoke test — V14 Emergent Dynamics (~50k tokens, tiny dims)",
    # === Scale ===
    "max_steps": 500,
    "batch_size": 32,
    "seq_len": 256,
    # === V14 Architecture (scaled down for smoke) ===
    "d_model": 128,
    "d_h": 64,
    "n_rules": 4,
    "morph_rule_hidden": 0,   # = d_h for smoke
    "n_morph_steps": 6,
    "n_theta": 2,
    "d_ff": 0,                # auto-compute
    "morph_dilations": (1, 2, 4, 8, 16, 32),
    "morph_gate_bias": 0.0,
    "morph_conv_kernel": 3,
    "step_embed_dim": 16,
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    "mos_components": 1,
    "tie_embeddings": True,
    "lambda_reg": 0.01,
    "lambda_aux": 0.05,
    "lambda_div": 0.1,
    "lambda_dyn": 0.01,
    "lambda_bal": 0.1,
    "lambda_step_div": 0.01,
    "target_params": 3_000_000,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 100,
    "eval_batches": 20,
    "generate_interval": 250,
    "save_interval": 500,
    "dataset": "tinystories",
    "checkpoint_interval": 0,
    "min_free_disk_gb": 0.1,
}

SMOKE_V14_20MIN = {
    "description": "V14 Emergent Dynamics — ~20min TinyStories, full-scale dims",
    # === Scale (~20 min on single GPU) ===
    "max_steps": 2000,
    "batch_size": 16,
    "seq_len": 512,
    # === V14 Architecture (full-scale dims, same as pile_250m) ===
    "d_model": 768,
    "d_h": 256,
    "n_rules": 8,
    "morph_rule_hidden": 0,   # = d_h (ratio 1.0); set 512 for dynamic ratio 2.0
    "n_morph_steps": 12,
    "n_theta": 4,
    "d_ff": 0,                # auto-compute to ~162M
    "target_params": 162_000_000,
    "morph_dilations": (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    "morph_gate_bias": 0.0,
    "morph_conv_kernel": 3,
    "step_embed_dim": 16,
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    "mos_components": 1,
    "tie_embeddings": False,
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_div": 0.1,
    "lambda_dyn": 0.01,
    "lambda_bal": 0.1,
    "lambda_step_div": 0.01,
    "dropout": 0.0,
    "checkpoint_interval": 4,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 100,
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 200,
    "eval_batches": 30,
    "generate_interval": 500,
    "save_interval": 500,
    "dataset": "tinystories",
    "min_free_disk_gb": 0.1,
}


SMOKE_V15 = {
    "description": "Smoke test — V15 2D RD-NCA (~50k tokens, tiny dims)",
    # === Scale ===
    "max_steps": 500,
    "batch_size": 32,
    "seq_len": 64,             # Small seq → 8×8 grid
    # === V15 Architecture (scaled down for smoke) ===
    "d_model": 128,
    "d_a": 32,
    "d_b": 16,
    "d_c": 32,
    "d_pos": 16,
    "d_rule": 96,              # ρ = 96/48 = 2.0
    "n_rules": 4,
    "n_morph_steps": 4,
    "n_theta": 2,
    "d_ff": 0,                 # auto-compute
    "step_embed_dim": 8,
    "k_a": 3,
    "k_b": 5,
    "p_async": 0.5,
    "max_seq_len": 64,
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    "mos_components": 1,
    "tie_embeddings": True,
    "lambda_reg": 0.01,
    "lambda_aux": 0.05,
    "lambda_div": 0.1,
    "lambda_bal": 0.1,
    "lambda_dyn": 0.01,
    "lambda_spatial": 0.1,
    "lambda_temporal": 0.05,
    "target_params": 3_000_000,
    "n_levels": 1,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_steps": 50,
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 100,
    "eval_batches": 20,
    "generate_interval": 250,
    "save_interval": 500,
    "dataset": "tinystories",
    "checkpoint_interval": 0,
    "min_free_disk_gb": 0.1,
}

SMOKE_V15_20MIN = {
    "description": "V15 2D RD-NCA — ~20min TinyStories, 160M params",
    # === Scale (~20 min on single GPU) ===
    "max_steps": 2000,
    "batch_size": 12,
    "seq_len": 512,             # 23×23 grid (TinyStories is short)
    # === V15 Architecture (full-scale dims) ===
    "d_model": 768,
    "d_a": 128,
    "d_b": 64,
    "d_c": 128,
    "d_pos": 32,
    "d_rule": 384,             # ρ = 384/192 = 2.0
    "n_rules": 8,
    "n_morph_steps": 8,
    "n_theta": 4,
    "d_ff": 0,                 # auto-compute to ~160M
    "step_embed_dim": 16,
    "k_a": 3,
    "k_b": 5,
    "p_async": 0.5,
    "max_seq_len": 512,
    "target_params": 160_000_000,
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    "mos_components": 1,
    "tie_embeddings": False,
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_div": 0.1,
    "lambda_bal": 0.1,
    "lambda_dyn": 0.01,
    "lambda_spatial": 0.1,
    "lambda_temporal": 0.05,
    "dropout": 0.0,
    "checkpoint_interval": 4,
    "n_levels": 1,
    # === Optimizer ===
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 100,
    # === Logging & Checkpointing ===
    "log_interval": 10,
    "eval_interval": 200,
    "eval_batches": 30,
    "generate_interval": 500,
    "save_interval": 500,
    "dataset": "tinystories",
    "min_free_disk_gb": 0.1,
}

V15_PILE_250M = {
    **_PILE_250M_COMMON,
    "description": "V15 2D RD-NCA on The Pile — 250M tokens, ~160M params",
    "batch_size": 4,
    "grad_accum_steps": 16,
    # === V15 Architecture (full scale) ===
    "d_model": 768,
    "d_a": 128,
    "d_b": 64,
    "d_c": 128,
    "d_pos": 32,
    "d_rule": 384,
    "n_rules": 8,
    "n_morph_steps": 8,
    "n_theta": 4,
    "d_ff": 0,
    "step_embed_dim": 16,
    "k_a": 3,
    "k_b": 5,
    "p_async": 0.5,
    "max_seq_len": 1024,
    "target_params": 160_000_000,
    "enable_halting": True,
    "halt_eps": 0.01,
    "lambda_halt": 0.01,
    "mos_components": 1,
    "tie_embeddings": False,
    "lambda_reg": 0.005,
    "lambda_aux": 0.05,
    "lambda_div": 0.1,
    "lambda_bal": 0.1,
    "lambda_dyn": 0.01,
    "lambda_spatial": 0.1,
    "lambda_temporal": 0.05,
    "dropout": 0.0,
    "checkpoint_interval": 4,
    "n_levels": 1,
    "min_free_disk_gb": 5,
}


PROFILES = {
    "smoke": SMOKE_PROFILE,
    "overnight": OVERNIGHT_PROFILE,
    "smoke_v4": SMOKE_V4,
    "overnight_v4": OVERNIGHT_V4,
    "smoke_v41": SMOKE_V41,
    "overnight_v41": OVERNIGHT_V41,
    "smoke_v5": SMOKE_V5,
    "overnight_v5": OVERNIGHT_V5,
    "smoltalk2_5b_v5": SMOLTALK2_5B_V5,
    "smoltalk2_5b_v5_mixed": SMOLTALK2_5B_V5_MIXED,
    "smoltalk2_5b_v5_probe": SMOLTALK2_5B_V5_PROBE,
    # ── Academic Comparison: The Pile 2B ──
    "pile_2b_aenca": AENCA_V5_PILE_2B,
    "pile_2b_gpt2": GPT2_PILE_2B,
    "pile_smoke_aenca": AENCA_V5_PILE_SMOKE,
    "pile_smoke_gpt2": GPT2_PILE_SMOKE,
    # ── 250M Quick Comparison ──
    "pile_250m_gpt2": GPT2_PILE_250M,
    "pile_250m_aenca": AENCA_V5_PILE_250M,
    "pile_250m_v6": AENCA_V6_PILE_250M,
    "pile_250m_v6_flops": AENCA_V6_PILE_250M_FLOPS,
    # ── V6 Oscillator-Controlled NCA ──
    "smoke_v6": SMOKE_V6,
    "pile_smoke_v6": AENCA_V6_PILE_SMOKE,
    "pile_2b_v6": AENCA_V6_PILE_2B,
    # ── V7 Stream-Hierarchical NCA ──
    "smoke_v7": SMOKE_V7,
    "pile_250m_v7": AENCA_V7_PILE_250M,
    "pile_2b_v7": AENCA_V7_PILE_2B,
    # ── V8 Growing Cellular Automata ──
    "smoke_v8": SMOKE_V8,
    "smoke_v8_f": SMOKE_V8_F,
    "prod_v8_f": PRODUCTION_V8_F,
    # ── V8 Morphogenesis (250B-scalable CA-Native) ──
    "smoke_morpho": SMOKE_MORPHO,
    "pile_250m_morpho": MORPHO_V8_PILE_250M,
    "morpho_250b": MORPHOGENESIS_250B,
    # ── V9 Wolfram CA (True Parallel, ~45K tok/s) ──
    "smoke_v9": SMOKE_V9,
    "pile_250m_v9": V9_PILE_250M,
    # ── V10 Persistent Growth NCA (64 steps, tiny rule, settlement) ──
    "smoke_v10": SMOKE_V10,
    "pile_250m_v10": V10_PILE_250M,
    # ── V11 Morphogenetic Field NCA (two-field coupled CA, emergent routing) ──
    "smoke_v11": SMOKE_V11,
    "smoke_v11_prod": SMOKE_V11_PROD,
    "pile_250m_v11": V11_PILE_250M,
    # ── V12 MMHR: Morphogenetic Mixture of Hebbian Rules (K=8 parallel rules) ──
    "smoke_v12": SMOKE_V12,
    "smoke_v12_50k": {**SMOKE_V12, "max_steps": 195, "max_dataset_tokens": 50_000, "description": "V12 MMHR 50k token quick run"},
    "smoke_v12_prod": SMOKE_V12_PROD,
    "pile_250m_v12": V12_PILE_250M,
    "pile_250m_v12_fast": V12_PILE_250M_FAST,
    # ── V13 Morphogenetic Attention (morphogen → Q/K gating) ──
    "smoke_v13": SMOKE_V13,
    "pile_250m_v13": V13_PILE_250M,
    "pile_2b_v13": V13_PILE_2B,
    # ── V14 Emergent Dynamics (content-enriched NCA communication, no attention) ──
    "smoke_v14": SMOKE_V14,
    "smoke_v14_20min": SMOKE_V14_20MIN,
    # ── V15 2D Latent Reaction-Diffusion NCA ──
    "smoke_v15": SMOKE_V15,
    "smoke_v15_20min": SMOKE_V15_20MIN,
    "pile_250m_v15": V15_PILE_250M,
}
