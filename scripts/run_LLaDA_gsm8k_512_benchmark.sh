#!/usr/bin/env bash
#
# GSM8K / generation length 512 benchmark on LLaDA-1.5.
#
# Reports, for every configuration, the three numbers used in the Flash-dLLM
# comparison table:
#   1. accuracy   -- 5-shot exact_match, flexible-extract filter
#   2. throughput -- decoding tokens/sec over the whole benchmark
#   3. tokens/step-- gen_length / NFE, i.e. tokens committed per denoising step
#
# Accuracy comes from lm-eval-harness (results_*.json); throughput and
# tokens/step come from the SpeedMeter written to speed.json in each run dir.
# Collect everything afterwards with:
#     python metrics/summarize_runs.py ./gsm8k512_log
#
# Protocol matches the Flash-dLLM paper's Table 1 / Table 5 setup for
# LLaDA-1.5 GSM8K-512: 5-shot, gen_length 512, block size 64, batch size 16,
# single A100 80GB.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# ---------------------------------------------------------------------------
# Knobs (override from the environment)
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/data1/wutianyi/models/LLaDA-1.5}"
# A stray space in a HF repo id turns into a confusing "repo not found".
# Also drop U+00A0, which copy-paste introduces and [:space:] does not match.
MODEL_PATH="$(printf '%s' "${MODEL_PATH}" \
  | LC_ALL=C tr -d '[:space:][:cntrl:]' \
  | sed $'s/\xc2\xa0//g')"
GEN_LENGTH="${GEN_LENGTH:-512}"
BLOCK_LENGTH="${BLOCK_LENGTH:-64}"   # beta = 64 in the Flash-dLLM paper
BATCH_SIZE="${BATCH_SIZE:-16}"       # Table 5: batch 16 for LLaDA-1.5 GSM8K
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"      # GSM8K is reported 5-shot
OUT_ROOT="${OUT_ROOT:-./gsm8k512_log}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Dynamic-dLLM cache (DCU) hyper-parameters, as in the README example.
WINDOW_SIZE="${WINDOW_SIZE:-32}"
LAYER_BUDGET="${LAYER_BUDGET:-32}"
SELECT_FROM="${SELECT_FROM:-v}"
PROMPT_INTERVAL="${PROMPT_INTERVAL:-5000}"
GEN_INTERVAL="${GEN_INTERVAL:-5000}"

# Adaptive Parallel Decoding (APD) hyper-parameters.
PD_MODE="${PD_MODE:-2}"              # 2 = per-token adaptive threshold
PD_THRESHOLD="${PD_THRESHOLD:-0.9}"
PD_ALPHA="${PD_ALPHA:-0.01}"
PD_BETA="${PD_BETA:-0.15}"
# The PD statistics hold two (batch, gen_length, vocab) probability tensors.
# At batch 16 / gen_length 512 that is 2 x 7.7 GiB in float64; pd_dtype=float32
# halves it if you are close to the memory limit.
PD_DTYPE="${PD_DTYPE:-float64}"

# Use the pre-populated local HF cache only if it actually exists, otherwise
# fall back to the default cache and let HF download. Forcing offline mode
# against a missing cache fails with an unhelpful "couldn't find" error.
# Override explicitly with USE_LOCAL_DATA_ENV=1 / 0.
# Note the directory must be non-empty: data/env.sh does `mkdir -p` on it, so
# merely sourcing that file once creates an empty tree that would otherwise
# look like a populated cache and wrongly flip us to offline mode.
LLADA_DATA_ROOT="${LLADA_DATA_ROOT:-/data1/wutianyi/data/llada}"
if [[ -z "${USE_LOCAL_DATA_ENV:-}" ]]; then
  if [[ -n "$(ls -A "${LLADA_DATA_ROOT}/hf_datasets" 2>/dev/null)" ]]; then
    USE_LOCAL_DATA_ENV=1
  else
    USE_LOCAL_DATA_ENV=0
  fi
fi

if [[ "${USE_LOCAL_DATA_ENV}" == "1" && -f "${REPO_DIR}/data/env.sh" ]]; then
  echo "[benchmark] using local HF cache at ${LLADA_DATA_ROOT} (offline mode)"
  source "${REPO_DIR}/data/env.sh"
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_OFFLINE=1
else
  echo "[benchmark] no local HF cache at ${LLADA_DATA_ROOT}; using the default HF cache (online)"
  export HF_DATASETS_TRUST_REMOTE_CODE=1
  export HF_ALLOW_CODE_EVAL=1
fi

GEN_KWARGS="block_length=${BLOCK_LENGTH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},cfg_scale=0.0"
COMMON_ARGS=(
  --model LLaDA
  --include_path ./lm_eval_tasks
  --tasks gsm8k_local
  --num_fewshot "${NUM_FEWSHOT}"
  --batch_size "${BATCH_SIZE}"
  --log_samples
  --apply_chat_template
  --fewshot_as_multiturn
  --gen_kwargs "${GEN_KWARGS}"
)

CACHE_ARGS="is_feature_cache=True,window_size=${WINDOW_SIZE},layer_budget=${LAYER_BUDGET},select_from=${SELECT_FROM},prompt_interval_steps=${PROMPT_INTERVAL},gen_interval_steps=${GEN_INTERVAL}"
APD_ARGS="generate_mode=pd,pd_mode=${PD_MODE},pd_threshold=${PD_THRESHOLD},pd_alpha=${PD_ALPHA},pd_beta=${PD_BETA},pd_dtype=${PD_DTYPE}"

# run <name> <extra model_args>
run () {
  local name="$1"; shift
  local extra="$1"; shift
  local out_dir="${OUT_ROOT}/${name}"

  echo ""
  echo "############################################################"
  echo "# ${name}"
  echo "#   gen_length=${GEN_LENGTH} block_length=${BLOCK_LENGTH} batch_size=${BATCH_SIZE} ${NUM_FEWSHOT}-shot"
  echo "#   model_args: pretrained=${MODEL_PATH},${extra}"
  echo "############################################################"

  mkdir -p "${out_dir}"
  DLLM_RUN_NAME="${name}" \
  DLLM_SPEED_LOG="${out_dir}/speed.json" \
  python evaluation_script.py \
    "${COMMON_ARGS[@]}" \
    --output_path "${out_dir}" \
    --model_args "pretrained=${MODEL_PATH},${extra},speed_log=${out_dir}/speed.json,run_name=${name}"
}

# --- 1. Baseline: no cache, fixed-schedule greedy decoding (1 token/step) ----
run "nocache_greedy" \
    "is_feature_cache=False,generate_mode=default"

# --- 2. DCU only: dynamic cache budget, greedy decoding ---------------------
run "dcu_greedy" \
    "${CACHE_ARGS},generate_mode=default"

# --- 3. APD only: adaptive parallel decoding, no cache ----------------------
run "apd_nocache" \
    "is_feature_cache=False,${APD_ARGS}"

# --- 4. Full Dynamic-dLLM: DCU + APD ----------------------------------------
run "dcu_apd" \
    "${CACHE_ARGS},${APD_ARGS}"

echo ""
echo "=== All runs finished. Summary: ==="
python metrics/summarize_runs.py "${OUT_ROOT}"
