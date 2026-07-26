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
CACHE_SLOW="is_feature_cache=True,window_size=32,layer_budget=32,select_from=v,prompt_interval_steps=5000,gen_interval_steps=5000"
CACHE_DEMO="is_feature_cache=True,window_size=32,layer_budget=32,select_from=v,prompt_interval_steps=100,gen_interval_steps=7"
NO_CACHE="is_feature_cache=False"

# name | gen_length | block_length | model_args | what it isolates
RUNGS=(
  # The only PD configuration the repo actually ships end-to-end: demo.py's
  # settings verbatim (pd_mode=0, threshold 0.9, cache refresh every 7 steps,
  # block 8, gen 256). If this does not score well, the problem is upstream of
  # any knob I picked.
  "d0_demo_exact_b8_g256|256|8|${CACHE_DEMO},generate_mode=pd,pd_mode=0,pd_threshold=0.9|demo.py config verbatim (reference)"
  "d1_apd2_nocache_b64_g512|512|64|${NO_CACHE},generate_mode=pd,pd_mode=2,pd_threshold=0.9|adaptive per-token threshold, no cache"
  "d2_apd0_nocache_b64_g512|512|64|${NO_CACHE},generate_mode=pd,pd_mode=0,pd_threshold=0.9|fixed threshold 0.9, no cache"
  "d3_apd1_nocache_b64_g512|512|64|${NO_CACHE},generate_mode=pd,pd_mode=1,pd_threshold=0.9|adaptive global threshold, no cache"
  "d4_dcu_apd2_b64_g512|512|64|${CACHE_SLOW},generate_mode=pd,pd_mode=2,pd_threshold=0.9|the failing config (cache + per-token)"
  "d5_dcu_apd2_demoint_b64_g512|512|64|${CACHE_DEMO},generate_mode=pd,pd_mode=2,pd_threshold=0.9|same but demo.py cache intervals"
  "d6_apd2_nocache_b8_g512|512|8|${NO_CACHE},generate_mode=pd,pd_mode=2,pd_threshold=0.9|block_length 8 instead of 64"
  "d7_nocache_greedy_b8_g256|256|8|${NO_CACHE},generate_mode=default|repo's own validated setting (ceiling)"
  "d8_nocache_greedy_b64_g512|512|64|${NO_CACHE},generate_mode=default|greedy at the benchmark's length/block"
  "d9_dcu_greedy_b64_g512|512|64|${CACHE_SLOW},generate_mode=default|greedy + cache (isolates DCU alone)"
)

echo "model=${MODEL_PATH}  limit=${LIMIT}  batch=${BATCH_SIZE}  ${NUM_FEWSHOT}-shot"
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
