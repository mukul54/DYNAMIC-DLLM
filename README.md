# Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLMs

This repository is the official PyTorch implementation of [Dynamic-dLLM](https://openreview.net/pdf?id=SdnkB5pGbq) (ICLR 2026).

## Overview

![Motivation of Dynamic-dLLM](assets/motivation_1.png)

(a-b) Layer input similarity and attention output similarity across adjacent denoising steps. Brighter regions indicate higher similarity — most tokens are stable across steps. (c-d) The number of tokens requiring updates varies across steps and layers, motivating layer-specific cache budgets. (e) Existing parallel decoding methods may discard valid candidates due to fixed thresholds.

![Illustration of Dynamic-dLLM](assets/frame_work.jpg)

Dynamic-dLLM consists of two components: **Dynamic Cache Updating (DCU)** reallocates the cache update budget per-layer at each denoising step, and **Adaptive Parallel Decoding (APD)** dynamically adjusts decoding thresholds for all tokens.

## News

- **[2026.01.26]** Paper accepted at **ICLR 2026**.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers accelerate lm-eval datasets
```

### 2. Prepare Model

Download [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and note its local path.

### 3. Run Demo

Edit `demo.py` to set your model path, then run:

```bash
python demo.py
```

The demo supports two generation modes configured at the top of the file:
- `generate_mode = "default"` — fixed-schedule block-wise masked diffusion
- `generate_mode = "pd"` — Prediction Dynamics adaptive threshold

Cache parameters:
- `window_size` — sliding window size for forced cache refresh (default 32)
- `layer_budget` — per-layer token update budget (default 32)
- `select_from` — feature source for similarity computation: `"v"`

### Hyper-parameters from the paper

`window_size` and `layer_budget` default to 32, which is the paper's value at
generation length 256. For other lengths the paper gives an auto-tuning rule,
`B_layer = B_window = gen_length / 8` — so generation length 512 wants **64**.

Adaptive Parallel Decoding uses `pd_mode=2` (Eq. 11 adapts a threshold per
token), initial threshold `pd_threshold=0.9` (Fig. 6c), and
`pd_alpha=0.001` / `pd_beta=0.0008` (Table 7, and stated again in Appendix B.2).

Block length is 32 for every benchmark (Table 4), at both generation length
256 and 512:

| Dataset | Steps | Block length | Generation length |
| --- | --- | --- | --- |
| MMLU, ARC-C, GSM8K, Math | 256 | 32 | 256 |
| HumanEval | 512 | 32 | 512 |

GSM8K is evaluated 4-shot.

**`pd_alpha` and `pd_beta` must stay in the 1e-3 range.** They control how fast
the per-token threshold relaxes, and the paper's Table 7 measures the cost of
scaling them on GSM8K at generation length 256:

| `pd_alpha` | `pd_beta` | Accuracy | Inference steps |
| --- | --- | --- | --- |
| 0.001 | 0.0008 | 78.01 | 95 |
| 0.01 | 0.008 | 69.75 | 54 |
| 0.1 | 0.08 | 59.76 | 16 |

Larger values commit more tokens per step and lose accuracy fast. Small
variations *within* the 1e-3 range are safe (0.0005–0.0017 all score ~78.8).

### 4. Evaluation

```bash
python evaluation_script.py \
  --model LLaDA \
  --model_args "pretrained=/path/to/LLaDA-8B-Instruct,is_feature_cache=True,window_size=32,layer_budget=32,select_from=v,prompt_interval_steps=5000,gen_interval_steps=5000,generate_mode=default" \
  --include_path ./lm_eval_tasks \
  --tasks gsm8k_local \
  --num_fewshot 4 \
  --batch_size 1 \
  --output_path ./gsm8k_log/default_window_budget_v \
  --log_samples \
  --apply_chat_template \
  --fewshot_as_multiturn
```

See `scripts/run_LLaDA_gsm8k_Instruct.sh` for a complete example.

### 5. Benchmarking: accuracy, throughput, tokens/step

`scripts/run_LLaDA_gsm8k_512_benchmark.sh` runs GSM8K at generation length 512
and reports all three metrics for four configurations (no-cache greedy baseline,
DCU only, APD only, and full Dynamic-dLLM).

```bash
MODEL_PATH=/path/to/LLaDA-1.5 bash scripts/run_LLaDA_gsm8k_512_benchmark.sh
```

Every run writes a `speed.json` next to its lm-eval results. Collect them into a
single table with:

```bash
python metrics/summarize_runs.py ./gsm8k512_log
```

```
run                     acc %      tok/s   speedup  tok/step      NFE     time s
--------------------------------------------------------------------------------
nocache_greedy          81.35        2.6      1.0x      1.00    512.0     9999.0
dcu_apd                 81.20      117.2     45.1x      6.80     75.3      221.0
```

Metric definitions:

| Metric | Definition |
| --- | --- |
| **Accuracy** | lm-eval `exact_match` under the `flexible-extract` filter, 5-shot |
| **Throughput** | total generated tokens ÷ total generation wall-time, aggregated over the benchmark. Only `generate()` is timed (tokenization, filtering and scoring are excluded), with `torch.cuda.synchronize()` on both sides. Tokens are counted up to the first `<eos>` of each sequence; `throughput_tok_s_full` in `speed.json` counts every filled position instead. |
| **Tokens/step** | `gen_length / NFE`, where NFE is the number of denoising steps (model forward passes). Fixed-schedule greedy decoding is exactly `1.00`; parallel decoding is higher. |

Defaults follow the standard LLaDA-1.5 GSM8K-512 protocol: 5-shot, `gen_length=512`,
`block_length=64`, `batch_size=16`, single A100 80GB. Override any of them from the
environment, e.g. `BATCH_SIZE=1 BLOCK_LENGTH=32 bash scripts/run_LLaDA_gsm8k_512_benchmark.sh`.
Throughput is strongly batch-size dependent, so compare only runs that used the
same batch size.

#### Memory at large batch sizes

Adaptive parallel decoding keeps two `(batch, gen_length, vocab)` probability
tensors alive to measure the step-to-step distribution shift. With LLaDA's
126k vocabulary that is the dominant allocation:

| batch | `gen_length` | float64 | float32 |
| --- | --- | --- | --- |
| 1 | 512 | 1.0 GiB | 0.5 GiB |
| 8 | 512 | 7.7 GiB | 3.9 GiB |
| 16 | 512 | 15.4 GiB | 7.7 GiB |

If a run OOMs, lower `BATCH_SIZE` or set `PD_DTYPE=float32` (≈1e-7 relative
error on the threshold comparisons; `float64` is the default and reproduces the
original implementation exactly).

## Project Structure

```
dynamic_dllm/
├── demo.py                         # Interactive/scripted generation demo
├── evaluation_script.py            # lm-eval-harness entry point
├── dynamic_dllm_cache/             # Core caching package
│   ├── cache/
│   │   ├── Cache.py                # DynamicDLLMCache singleton
│   │   └── Config.py               # DynamicDLLMCacheConfig dataclass
│   └── hooks/
│       └── cache_hook_LLaDA.py     # Window-budget cache hooks for LLaDA
├── eval_model/
│   └── LLaDA.py                    # lm-eval model wrapper (registered as "LLaDA")
├── utils/
│   ├── generate_function.py        # generate() and generate_pd() routines
│   └── utils.py                    # set_seed() helper
├── data/                           # Dataset download/verification scripts
├── lm_eval_tasks/                  # Custom lm-eval task definitions
├── metrics/                        # Accuracy and pass@1 computation
│   ├── speed.py                    # Throughput / tokens-per-step accounting
│   └── summarize_runs.py           # Collect accuracy + speed into one table
├── scripts/                        # Evaluation run scripts
└── assets/                         # Images for documentation
```

## Citation

```bibtex
@inproceedings{dynamic-dllm2026,
  title={Dynamic dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion Large Language Models},
  author={...},
  booktitle={ICLR},
  year={2026}
}
```
