#!/usr/bin/env bash
#
# Isolate which configuration knob costs accuracy on GSM8K.
#
# Each rung changes exactly ONE thing from a neighbouring rung, so the step
# where accuracy collapses identifies the cause. Runs on a small --limit so the
# whole ladder finishes in hours rather than days.
#
#   bash scripts/diagnose_gsm8k.sh                # every rung, cheapest first
#   bash scripts/diagnose_gsm8k.sh d1 d2          # only these rungs
#   LIMIT=100 bash scripts/diagnose_gsm8k.sh      # more samples per rung
#
# Then:  python metrics/summarize_runs.py ./gsm8k_diag
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-1.5}"
MODEL_PATH="$(printf '%s' "${MODEL_PATH}" \
  | LC_ALL=C tr -d '[:space:][:cntrl:]' | sed $'s/\xc2\xa0//g')"
LIMIT="${LIMIT:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
OUT_ROOT="${OUT_ROOT:-./gsm8k_diag}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Cache settings. "slow" = README/eval setting (partial updates only, no full
# refresh); "demo" = demo.py setting (periodic full refresh every 7 steps).
CACHE_SLOW="is_feature_cache=True,window_size=64,layer_budget=64,select_from=v,prompt_interval_steps=5000,gen_interval_steps=5000"
CACHE_DEMO="is_feature_cache=True,window_size=64,layer_budget=64,select_from=v,prompt_interval_steps=100,gen_interval_steps=7"
NO_CACHE="is_feature_cache=False"

# Paper values: Eq. 11 / Fig. 6c / Table 7.
APD_PAPER="generate_mode=pd,pd_mode=2,pd_threshold=0.9,pd_alpha=0.001,pd_beta=0.0008"
# What the repo's stale defaults produce, kept as the negative control.
APD_REPO="generate_mode=pd,pd_mode=2,pd_threshold=0.9,pd_alpha=0.01,pd_beta=0.15"

# name | gen_length | block_length | model_args | what it isolates
RUNGS=(
  # d1 is the headline check: paper alpha/beta at the benchmark's settings.
  # Table 4 fixes block length at 32 for every benchmark, so d1 is the paper's
  # configuration end to end and is the one to report.
  "d1_paper_dcu_apd_b32_g512|512|32|${CACHE_SLOW},${APD_PAPER}|paper config: alpha/beta 1e-3, block 32, budget 64"
  "d2_paper_apd_nocache_b32_g512|512|32|${NO_CACHE},${APD_PAPER}|same without the cache (isolates APD)"
  "d3_stale_ab_dcu_apd_b32_g512|512|32|${CACHE_SLOW},${APD_REPO}|repo's stale alpha/beta (negative control)"
  "d4_paper_dcu_apd_b64_g512|512|64|${CACHE_SLOW},${APD_PAPER}|block 64, to match a block-64 comparison"
  "d5_paper_fixed_thresh_b32_g512|512|32|${CACHE_SLOW},generate_mode=pd,pd_mode=0,pd_threshold=0.9|fixed threshold 0.9 (APD should match this accuracy with fewer steps)"
  "d6_paper_dcu_apd_b32_g256|256|32|${CACHE_SLOW},${APD_PAPER}|the paper's exact GSM8K row (gen 256)"
  "d7_nocache_greedy_b32_g256|256|32|${NO_CACHE},generate_mode=default|no-cache greedy, paper's gen 256 (ceiling)"
  "d8_nocache_greedy_b32_g512|512|32|${NO_CACHE},generate_mode=default|no-cache greedy at gen 512"
  "d9_dcu_greedy_b32_g512|512|32|${CACHE_SLOW},generate_mode=default|greedy + cache (isolates DCU alone)"
)

# Guard against running a stale checkout: the whole point of this ladder is the
# corrected alpha/beta, and a run with the old values looks superficially fine
# in the logs. Fail loudly rather than burn GPU hours on the wrong config.
if ! grep -q "pd_alpha: float = 0.001" "${REPO_DIR}/eval_model/LLaDA.py"; then
  echo "ERROR: eval_model/LLaDA.py still has the old pd_alpha default." >&2
  echo "       Run: git pull origin \$(git rev-parse --abbrev-ref HEAD)" >&2
  exit 1
fi

echo "model=${MODEL_PATH}  limit=${LIMIT}  batch=${BATCH_SIZE}  ${NUM_FEWSHOT}-shot"
echo "APD: ${APD_PAPER}"
echo "Rungs d1-d6 are cheap (parallel decoding); d7-d9 are slow (one token/step)."
echo ""

for entry in "${RUNGS[@]}"; do
  IFS='|' read -r name gen_length block_length extra note <<< "${entry}"

  if [[ $# -gt 0 ]]; then
    wanted=0
    for arg in "$@"; do [[ "${name}" == "${arg}"* ]] && wanted=1; done
    [[ ${wanted} -eq 1 ]] || continue
  fi

  out_dir="${OUT_ROOT}/${name}"
  echo "############################################################"
  echo "# ${name}"
  echo "#   ${note}"
  echo "#   gen_length=${gen_length} block_length=${block_length}"
  echo "############################################################"
  mkdir -p "${out_dir}"

  python evaluation_script.py \
    --model LLaDA \
    --model_args "pretrained=${MODEL_PATH},${extra},speed_log=${out_dir}/speed.json,run_name=${name}" \
    --gen_kwargs "block_length=${block_length},gen_length=${gen_length},steps=${gen_length},cfg_scale=0.0" \
    --include_path ./lm_eval_tasks --tasks gsm8k_local \
    --num_fewshot "${NUM_FEWSHOT}" --batch_size "${BATCH_SIZE}" --limit "${LIMIT}" \
    --apply_chat_template --fewshot_as_multiturn --log_samples \
    --output_path "${out_dir}"
done

echo ""
echo "=== Diagnostic summary ==="
python metrics/summarize_runs.py "${OUT_ROOT}"
