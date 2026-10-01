# NCALM Comparative Training Protocol

**Document version:** 1.0
**Date:** 2026-03-20 (the original file carried a typo'd year)
**Purpose:** Everything an independent researcher needs to train a novel ~162M-parameter language model and produce academically rigorous, apples-to-apples comparisons against our GPT-2 baseline.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Experimental Design](#2-experimental-design)
3. [Data Specification](#3-data-specification)
4. [Tokenizer](#4-tokenizer)
5. [GPT-2 Baseline Architecture](#5-gpt-2-baseline-architecture)
6. [Optimizer & Schedule](#6-optimizer--schedule)
7. [Training Tiers](#7-training-tiers)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Metrics & Logging Format](#9-metrics--logging-format)
10. [Generation Protocol](#10-generation-protocol)
11. [Reproducibility](#11-reproducibility)
12. [GPT-2 Baseline Results](#12-gpt-2-baseline-results)
13. [What You Must Match](#13-what-you-must-match)
14. [What You May Change](#14-what-you-may-change)
15. [File & Directory Layout](#15-file--directory-layout)
16. [Statistical Reporting Checklist](#16-statistical-reporting-checklist)

---

> **Read this first.** This protocol was written in March 2026 and governed every run in `runs/`.
> Two defects in the *implementation* of §8 were found afterwards and are **not** fixed in the recorded
> runs: (1) validation windows are drawn at random on every evaluation (`ncalm/data`), so there is no
> fixed held-out set and the minimum over evaluations is an optimistic statistic; (2) the validation
> loader inherits the training micro-batch size, so the GPT-2 baseline was scored on roughly twice the
> validation tokens per evaluation that the challengers were. Both effects favour the challengers.
> Treat differences under ~15% as within evaluation noise. The §12 baseline tables below are the
> numbers as logged.

## 1. Overview

We are comparing novel LLM architectures against a well-characterised GPT-2 baseline at 162M parameters. The baseline has **already been trained** — you do **not** need to retrain it. Your task is to train your architecture under the same controlled conditions so we can make a fair scientific comparison on a single shared axis: **validation perplexity as a function of tokens seen**.

The protocol has three tiers, each one a gate to the next:

| Tier | Token budget | Purpose | Wall-clock (RTX 4070 Ti SUPER) |
|------|-------------|---------|-------------------------------|
| **Smoke** | ~3.3M | Sanity check: loss decreases, no NaN, VRAM fits | ~5 min |
| **250M** | 250M | Quick comparison: meaningful perplexity gap visible | ~2 h |
| **2B** | 2B | Academic-grade comparison: publishable result | ~16 h |

All tiers use the same data source, tokenizer, optimizer, LR schedule shape, and evaluation protocol. Only the token budget and derived schedule parameters differ.

---

## 2. Experimental Design

### Independent Variable
Model architecture (your model vs. the GPT-2 baseline).

### Controlled Variables
Everything else:
- Dataset (source, split, ordering)
- Tokenizer (pretrained, frozen, identical vocab)
- Total parameter count (~162M ± 0.5%)
- Token budget per tier
- Optimizer (algorithm, hyperparameters)
- LR schedule (shape, warmup fraction, min LR ratio)
- Batch configuration (effective tokens per step)
- Evaluation protocol (number of batches, metric computation)
- Random seed
- Mixed precision (AMP + GradScaler on CUDA)

### Dependent Variable
Validation perplexity (PPL), measured at fixed step intervals.

### Parameter Budget
Your model's **total trainable parameter count** must be within ±0.5% of the baseline:

$$
\text{Target: } 162{,}322{,}944 \pm 811{,}615 \text{ parameters}
$$

Report the exact count in your `config.json` and `analysis.json`.

---

## 3. Data Specification

### Source
- **Dataset:** The Pile (uncopyrighted subset)
- **HuggingFace identifier:** `monology/pile-uncopyrighted`
- **Split used:** `train` only (we never touch the official test/validation splits)
- **Streaming:** Loaded via HuggingFace `datasets` with `streaming=True`

### Tokenization & Caching
The raw text is tokenized into a flat `int32` array and cached to disk as a memory-mapped binary file (`.bin`). Each document is appended with an EOS token (`<|endoftext|>`, token ID 0). Documents are concatenated sequentially in their natural dataset order.

### Train/Validation Split
The flat token array is split **positionally** (not by document boundaries):

| Split | Range | Fraction |
|-------|-------|----------|
| Train | `tokens[0 : 0.995 × N]` | 99.5% |
| Val   | `tokens[0.995 × N : N]` | 0.5% |

where `N` = total cached tokens (capped by `max_dataset_tokens` for the tier).

### Batch Construction
Each training batch is constructed by **uniform random sampling** of start positions within the train split:

1. Sample `batch_size` random start indices from `[0, N_train - seq_len - 1)`
2. Extract windows of length `seq_len` (1024 tokens)
3. No epoch boundary, no sequential scan — purely random access

This means the data loader is infinite (never exhausts); the training loop runs for exactly `max_steps` steps.

### Cache Files Per Tier

| Tier | Cache file | `max_dataset_tokens` |
|------|-----------|---------------------|
| Smoke | `pile_neox50k_tokens_smoke.bin` | 10,000,000 |
| 250M | `pile_neox50k_tokens_250m.bin` | 250,000,000 |
| 2B | `pile_neox50k_tokens_2b.bin` | 2,000,000,000 |

The cache is deterministic: same tokenizer + same dataset + same `max_dataset_tokens` ⇒ identical token sequence.

---

## 4. Tokenizer

| Property | Value |
|----------|-------|
| Type | Pretrained HuggingFace fast tokenizer |
| Model | `EleutherAI/gpt-neox-20b` (GPT-NeoX tokenizer) |
| Vocabulary size | **50,304** (50,257 base + 47 padding tokens for GPU alignment to ×128) |
| EOS token | `<|endoftext|>` (ID 0) |
| Byte-fallback | Yes (BPE with byte-level fallback) |

**Critical:** Your model's embedding table must have exactly 50,304 rows. This is required for:
1. Identical token-to-ID mapping (fair comparison)
2. GPU-friendly tensor sizes (divisible by 128)

Load with:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True)
```

The actual vocabulary has 50,257 entries. We pad `vocab_size` to 50,304 in the embedding/output layers.

---

## 5. GPT-2 Baseline Architecture

The baseline is a Pythia-160M–style GPT-2 variant. Architecture details for reference:

| Hyperparameter | Value |
|---------------|-------|
| `d_model` | 768 |
| `n_layer` | 12 |
| `n_head` | 12 |
| `d_ff` | 3072 (4× d_model) |
| `max_seq_len` | 2048 |
| `vocab_size` | 50,304 |
| `tie_embeddings` | False |
| `use_parallel_residual` | True |
| `rotary_pct` | 0.25 (RoPE on 25% of head dim) |
| `dropout` | 0.0 |
| **Total parameters** | **162,322,944** |

Key design choices:
- **Parallel residual connections** (GPT-J style): attention and FFN outputs are added in parallel rather than sequentially
- **Rotary Position Embeddings (RoPE):** Applied to 25% of the head dimension (following Pythia convention)
- **No weight tying:** Separate input embedding and output projection matrices
- **No dropout:** Regularisation comes solely from weight decay

---

## 6. Optimizer & Schedule

### Optimizer: AdamW

| Parameter | Value |
|-----------|-------|
| Algorithm | AdamW (decoupled weight decay) |
| Learning rate | 2 × 10⁻⁴ |
| β₁ | 0.9 |
| β₂ | 0.99 |
| ε | 1 × 10⁻⁸ |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 (global L2 norm) |

### LR Schedule: Cosine Annealing with Linear Warmup

```
LR(step) = 
  ┌ base_lr × (step / warmup_steps)                    if step < warmup_steps
  │
  └ base_lr × [min_ratio + (1 − min_ratio) × 
      0.5 × (1 + cos(π × progress))]                   if step ≥ warmup_steps
```

where:
- `progress = (step − warmup_steps) / (max_steps − warmup_steps)`, clamped to [0, 1]
- `min_ratio = 0.1` (decays to 10% of peak LR)
- `base_lr = 2 × 10⁻⁴`

At the end of training: LR = 2 × 10⁻⁵.

### Warmup Steps Per Tier

| Tier | Warmup steps | ≈ % of total |
|------|-------------|-------------|
| Smoke | 5 | 10% |
| 250M | 188 | ~5% |
| 2B | 1,500 | ~5% |

### Mixed Precision
- **AMP** (Automatic Mixed Precision) enabled when device is CUDA
- **GradScaler** used alongside AMP for stable FP16 training
- No BF16 override — uses PyTorch's default AMP dtype selection

---

## 7. Training Tiers

### Effective Batch Size
All tiers use the same effective batch size:

$$
\text{tokens/step} = \text{batch\_size} \times \text{seq\_len} \times \text{grad\_accum\_steps} = 65{,}536
$$

The micro-batch size (`batch_size`) and gradient accumulation steps (`grad_accum_steps`) may vary by architecture to fit in VRAM, but their product must yield 65,536 tokens per optimizer step.

### Tier 1: Smoke Test

| Parameter | Value |
|-----------|-------|
| `max_steps` | 50 |
| `batch_size` | 8 |
| `seq_len` | 1024 |
| `grad_accum_steps` | 8 |
| `max_dataset_tokens` | 10,000,000 |
| `warmup_steps` | 5 |
| `eval_interval` | 10 |
| `eval_batches` | 10 |
| `generate_interval` | 25 |
| `save_interval` | 50 |
| `log_interval` | 1 |

**Token budget:** 50 × 65,536 = 3,276,800

**Pass criteria:**
- No NaN in loss or gradients
- Loss monotonically decreasing (roughly)
- Peak VRAM < 16 GB (on RTX 4070 Ti SUPER; adjust threshold to your hardware)
- Training completes without error

### Tier 2: 250M Quick Comparison

| Parameter | Value |
|-----------|-------|
| `max_steps` | 3,815 |
| `batch_size` | 8 |
| `seq_len` | 1024 |
| `grad_accum_steps` | 8 |
| `max_dataset_tokens` | 250,000,000 |
| `warmup_steps` | 188 |
| `eval_interval` | 200 |
| `eval_batches` | 50 |
| `generate_interval` | 750 |
| `save_interval` | 750 |
| `log_interval` | 10 |

**Token budget:** 3,815 × 65,536 = 250,019,840 ≈ 250M

**Note on `batch_size` / `grad_accum_steps`:** If your model requires smaller micro-batches for VRAM reasons, adjust these two values keeping their product at 64 (i.e., `4 × 16`, `2 × 32`, etc.). The effective tokens per step must remain 65,536.

### Tier 3: 2B Academic Comparison

| Parameter | Value |
|-----------|-------|
| `max_steps` | 30,518 |
| `batch_size` | 8 |
| `seq_len` | 1024 |
| `grad_accum_steps` | 8 |
| `max_dataset_tokens` | 2,000,000,000 |
| `warmup_steps` | 1,500 |
| `eval_interval` | 1,000 |
| `eval_batches` | 50 |
| `generate_interval` | 5,000 |
| `save_interval` | 5,000 |
| `log_interval` | 20 |

**Token budget:** 30,518 × 65,536 = 2,000,076,288 ≈ 2B

---

## 8. Evaluation Protocol

### Validation Loss Computation

1. Set model to eval mode
2. For each of `eval_batches` batches (drawn from the val split):
   a. Forward pass through the model
   b. Compute cross-entropy loss on **T − 1 predicted positions** (standard causal LM: predict token `t+1` from context `[0..t]`)
   c. Accumulate: `total_ce += ce_per_token × B × (T − 1)`
   d. Accumulate: `total_tokens += B × (T − 1)`
3. Average CE: `val_ce = total_ce / total_tokens`
4. Perplexity: `val_ppl = exp(min(val_ce, 20))`

The `min(val_ce, 20)` cap prevents numerical overflow early in training (exp(20) ≈ 4.85 × 10⁸).

### Evaluation Schedule

| Tier | Eval every N steps | Eval batches | Total eval tokens per checkpoint |
|------|-------------------|-------------|--------------------------------|
| Smoke | 10 | 10 | 10 × 8 × 1023 = 81,840 |
| 250M | 200 | 50 | 50 × 8 × 1023 = 409,200 |
| 2B | 1,000 | 50 | 50 × 8 × 1023 = 409,200 |

### What to Report

For each evaluation checkpoint, record:
- `step`: Training step number
- `val_ce`: Average cross-entropy (nats, not bits)
- `val_ppl`: exp(val_ce)

---

## 9. Metrics & Logging Format

### File: `metrics.jsonl`

One JSON object per line. Two types of records:

#### Training step record (every step)

```json
{
  "step": 100,
  "timestamp": 1773919250.486,
  "loss": 7.234,
  "ce": 7.234,
  "ppl": 1383.7,
  "lr": 0.000106,
  "grad_norm": 3.45,
  "tokens_per_sec": 48000.0,
  "step_time": 1.365,
  "gpu_mem_gb": 2.45,
  "gpu_peak_gb": 6.68,
  "tokens_processed": 6553600,
  "grad_accum_steps": 8
}
```

#### Evaluation record (every `eval_interval` steps)

```json
{
  "step": 200,
  "type": "eval",
  "val_ce": 6.1772,
  "val_ppl": 481.6
}
```

### Required Fields for Comparison

The following fields are **mandatory** for scientific comparison. You may add architecture-specific fields, but these must be present:

**Training records:**
| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Global optimizer step (1-indexed) |
| `timestamp` | float | Unix epoch seconds |
| `loss` | float | Total training loss (may include aux losses) |
| `ce` | float | Cross-entropy component only (for PPL) |
| `ppl` | float | exp(min(ce, 20)) |
| `lr` | float | Current learning rate |
| `grad_norm` | float | Global gradient L2 norm (before clipping) |
| `tokens_per_sec` | float | Training throughput |
| `step_time` | float | Wall-clock seconds for this step |
| `gpu_mem_gb` | float | Current GPU memory allocated |
| `gpu_peak_gb` | float | Peak GPU memory allocated |
| `tokens_processed` | int | Cumulative tokens seen |
| `grad_accum_steps` | int | Gradient accumulation steps |

**Evaluation records:**
| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Step at which eval was performed |
| `type` | str | Must be `"eval"` |
| `val_ce` | float | Validation cross-entropy |
| `val_ppl` | float | exp(min(val_ce, 20)) |

### File: `config.json`

Saved at experiment start. Must include all training hyperparameters and architecture parameters. This is the single source of truth for reproducing the run.

### File: `training.log`

Human-readable log file. No strict format required, but should contain:
- Training banner (model name, param count, config summary)
- Periodic console-style progress lines
- Eval results
- Generation samples
- Checkpoint save events
- Final summary

### File: `analysis.json`

Written at training completion. Summary of the run including:
- Final validation PPL and CE
- Best validation PPL
- Total training time
- Hardware info
- Complete config

---

## 10. Generation Protocol

At periodic intervals during training (`generate_interval` steps), generate a text sample for qualitative monitoring:

| Parameter | Value |
|-----------|-------|
| Prompt | `"Once upon a time"` |
| Max tokens | 100 |
| Temperature | 0.8 |
| Top-k | 50 |
| Context window | Last 512 tokens (if generated sequence exceeds 512) |

Generation uses greedy top-k sampling (no beam search). Stop on EOS token or max_tokens.

At training completion, generate three samples with prompts:
- `"Once upon a time"`
- `"The little cat"`
- `"One day, a"`

---

## 11. Reproducibility

### Random Seed
All runs use **seed 42**:
```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
```

**Note:** `torch.backends.cudnn.deterministic` is intentionally **not** set because it significantly degrades performance. The seed controls initialisation and data sampling order, which is sufficient for reproducibility of aggregate metrics.

### Deterministic Data Order
Because the data loader uses random start positions (seeded by PyTorch's RNG), and the tokenized cache is deterministic, the same seed produces the same training data sequence.

### Config Recording
The complete config dict is serialised to `config.json` at experiment start. This includes the seed, all hyperparameters, and all architecture parameters. Any run should be reproducible from this file.

---

## 12. GPT-2 Baseline Results

**These results are final. Do NOT retrain the GPT-2 baseline.** Compare your model against these numbers directly.

### Tier 1: Smoke Test

| Step | val_ce | val_ppl |
|-----:|-------:|--------:|
| 10 | 9.1242 | 9,174.5 |
| 20 | 8.2006 | 3,643.1 |
| 30 | 7.7884 | 2,412.4 |
| 40 | 7.7080 | 2,226.0 |
| 50 | 7.6292 | 2,057.4 |

Final train CE: 7.450 | Final train PPL: 1,719.9

### Tier 2: 250M Quick Comparison

| Step | val_ce | val_ppl |
|-----:|-------:|--------:|
| 200 | 6.1772 | 481.6 |
| 400 | 5.4580 | 234.6 |
| 600 | 4.8665 | 129.9 |
| 800 | 4.3954 | 81.1 |
| 1,000 | 4.1971 | 66.5 |
| 1,200 | 3.9715 | 53.1 |
| 1,400 | 3.8017 | 44.8 |
| 1,600 | 3.7329 | 41.8 |
| 1,800 | 3.6696 | 39.2 |
| 2,000 | 3.6683 | 39.2 |
| 2,200 | 3.5931 | 36.3 |
| 2,400 | 3.4990 | 33.1 |
| 2,600 | 3.5052 | 33.3 |
| 2,800 | 3.5232 | 33.9 |
| 3,000 | 3.3660 | 29.0 |
| 3,200 | 3.3861 | 29.5 |
| 3,400 | 3.3166 | 27.6 |
| 3,600 | 3.4452 | 31.3 |
| 3,800 | 3.3198 | 27.7 |

**Best val_ppl: 27.6** (step 3,400) | Final val_ppl: 27.7 (step 3,800)
Final train CE: 3.173 | Final LR: 2 × 10⁻⁵

### Tier 3: 2B Academic Comparison

| Step | val_ce | val_ppl |
|-----:|-------:|--------:|
| 1,000 | 5.0397 | 154.4 |
| 2,000 | 3.7681 | 43.3 |
| 3,000 | 3.4331 | 31.0 |
| 4,000 | 3.3053 | 27.3 |
| 5,000 | 3.2443 | 25.6 |
| 6,000 | 3.1660 | 23.7 |
| 7,000 | 3.0827 | 21.8 |
| 8,000 | 3.0345 | 20.8 |
| 9,000 | 3.0295 | 20.7 |
| 10,000 | 2.9876 | 19.8 |
| 11,000 | 3.0143 | 20.4 |
| 12,000 | 2.9004 | 18.2 |
| 13,000 | 2.8482 | 17.3 |
| 14,000 | 2.8698 | 17.6 |
| 15,000 | 2.8600 | 17.5 |
| 16,000 | 2.7847 | 16.2 |
| 17,000 | 2.7722 | 16.0 |
| 18,000 | 2.8858 | 17.9 |
| 19,000 | 2.7980 | 16.4 |
| 20,000 | 2.7582 | 15.8 |
| 21,000 | 2.8165 | 16.7 |
| 22,000 | 2.7698 | 16.0 |
| 23,000 | 2.7423 | 15.5 |
| 24,000 | 2.7480 | 15.6 |
| 25,000 | 2.6926 | 14.8 |
| 26,000 | 2.7321 | 15.4 |
| 27,000 | 2.7103 | 15.0 |
| 28,000 | 2.7057 | 15.0 |
| 29,000 | 2.7454 | 15.6 |
| 30,000 | 2.6710 | 14.5 |

**Best val_ppl: 14.5** (step 30,000) | Final val_ppl: 14.5
Final train CE: 2.824 | Final LR: 2 × 10⁻⁵

---

## 13. What You Must Match

These are **hard constraints** — deviations invalidate the comparison:

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| Tokenizer | `EleutherAI/gpt-neox-20b` | Identical token-to-ID mapping |
| `vocab_size` | 50,304 | Same embedding table size |
| Dataset | `monology/pile-uncopyrighted`, train split | Same data distribution |
| `max_dataset_tokens` | Per tier (see §7) | Same token budget |
| `seq_len` | 1024 | Same context length |
| Effective tokens/step | 65,536 | Same optimiser exposure per step |
| `max_steps` | Per tier (see §7) | Same number of optimiser updates |
| Optimizer | AdamW | Same optimiser algorithm |
| `learning_rate` | 2 × 10⁻⁴ | Same peak learning rate |
| `weight_decay` | 0.1 | Same regularisation strength |
| β₁, β₂ | 0.9, 0.99 | Same momentum parameters |
| ε | 1 × 10⁻⁸ | Same numerical stability constant |
| `grad_clip` | 1.0 | Same gradient clipping threshold |
| LR schedule | Cosine w/ linear warmup, min_ratio=0.1 | Same schedule shape |
| `warmup_steps` | Per tier (see §7) | Same warmup duration |
| `eval_interval` | Per tier (see §7) | Same eval frequency |
| `eval_batches` | Per tier (see §7) | Same eval set size |
| Eval metric | PPL = exp(min(CE, 20)) on T−1 tokens | Same evaluation formula |
| Train/val split | 99.5% / 0.5% positional | Same data split |
| Seed | 42 | Same initialisation randomness |
| `dropout` | 0.0 | No dropout (fair comparison) |
| Total params | 162,322,944 ± 0.5% | Same model capacity |

---

## 14. What You May Change

The comparative AI model architecture

---

## 15. File & Directory Layout

Each experiment run should produce:

```
experiments/<YourArchName>/<run_name>_<timestamp>/
├── config.json           # Complete training config (written at start)
├── metrics.jsonl         # All training + eval metrics (one JSON per line)
├── training.log          # Human-readable log
├── analysis.json         # Summary (written at end)
├── best_model.pt         # Checkpoint with lowest val_ppl
├── latest.pt             # Most recent checkpoint
├── final_model.pt        # Checkpoint after last step
└── checkpoint_step*.pt   # Periodic checkpoints (keep last N)
```

The `<timestamp>` format is `YYYYMMDD_HHMMSS`.

### Checkpoint Format

Each `.pt` file contains:

```python
{
    "step": int,
    "tokens_processed": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "best_val_ppl": float,
    "config": dict,
    # Optional:
    "scaler_state_dict": dict,    # If using AMP
}
```

---

## 16. Statistical Reporting Checklist

When reporting results, include:

- [ ] **Parameter count** — exact total trainable parameters
- [ ] **val_ppl trajectory** — full table at every eval checkpoint (as in §12)
- [ ] **Best val_ppl** — lowest observed during training, with step number
- [ ] **Final val_ppl** — at the last eval checkpoint
- [ ] **Final train CE** — training cross-entropy at last step
- [ ] **Training throughput** — average tokens/sec
- [ ] **Peak VRAM** — maximum GPU memory (GB)
- [ ] **Total wall-clock time** — hours
- [ ] **Hardware** — GPU model, VRAM
- [ ] **config.json** — attached or transcribed in full
- [ ] **metrics.jsonl** — attached for full reproducibility
- [ ] **Relative PPL** — your model's best val_ppl / GPT-2 best val_ppl (< 1.0 means you beat GPT-2)

### How to Interpret Results

| Guideline | Interpretation |
|-----------|---------------|
| val_ppl ratio < 0.95 | Your model convincingly outperforms GPT-2 |
| val_ppl ratio ∈ [0.95, 1.05] | Roughly equivalent — differences may be noise |
| val_ppl ratio > 1.05 | GPT-2 outperforms your model |
| val_ppl ratio > 1.50 | Significant capacity or optimisation issue |
| val_ppl ratio > 2.00 | Fundamental architectural problem likely |

For the 2B tier, a ratio below 1.0 at this parameter scale would be a notable result worth investigating further (e.g., with ablations, different model sizes, longer training).

---

## Appendix A: Quick Reference — Tier Configs

```
                        Smoke       250M        2B
─────────────────────────────────────────────────────
max_steps               50          3,815       30,518
batch_size              8           8           8
seq_len                 1024        1024        1024
grad_accum_steps        8           8           8
tokens/step             65,536      65,536      65,536
total tokens            ~3.3M       ~250M       ~2B
max_dataset_tokens      10M         250M        2B
warmup_steps            5           188         1,500
eval_interval           10          200         1,000
eval_batches            10          50          50
generate_interval       25          750         5,000
save_interval           50          750         5,000
log_interval            1           10          20
cache file              *_smoke.bin *_250m.bin  *_2b.bin
GPT-2 best val_ppl      2,057.4     27.6        14.5
```

## Appendix B: Minimal Training Loop Pseudocode

```python
# Setup
seed_everything(42)
tokenizer = load_pretrained("EleutherAI/gpt-neox-20b")
tokens = tokenize_and_cache(dataset="monology/pile-uncopyrighted", 
                            split="train", max_tokens=MAX_DATASET_TOKENS)
train_tokens = tokens[:int(len(tokens) * 0.995)]
val_tokens   = tokens[int(len(tokens) * 0.995):]

model = YourModel(config)  # ~162M params
optimizer = AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.99), 
                  weight_decay=0.1, eps=1e-8)
scheduler = CosineAnnealingWithWarmup(optimizer, warmup=WARMUP, 
                                       max_steps=MAX_STEPS, min_ratio=0.1)
scaler = GradScaler()

for step in range(1, MAX_STEPS + 1):
    optimizer.zero_grad()
    
    for _ in range(GRAD_ACCUM):
        batch = random_windows(train_tokens, batch_size=B, seq_len=1024)
        with autocast():
            loss, info = model.forward_train(batch)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
    log_metrics(step, info)
    
    if step % EVAL_INTERVAL == 0:
        val_ce = evaluate(model, val_tokens, n_batches=EVAL_BATCHES)
        val_ppl = math.exp(min(val_ce, 20))
        log_eval(step, val_ce, val_ppl)
```

## Appendix C: Cross-Entropy & Perplexity Calculation Detail

For a batch of shape `(B, T)`:
- Input: `tokens[:, :-1]` (positions 0 to T−2)
- Target: `tokens[:, 1:]` (positions 1 to T−1)
- CE is computed as mean per predicted token: `sum(CE) / (B × (T − 1))`
- During eval, CE is accumulated across all batches and re-averaged over total tokens
- PPL = exp(min(CE, 20)) — the min-cap prevents overflow, allowing early-training metrics to be finite

The CE is in **nats** (natural log base), not bits. To convert: bits = nats / ln(2).
