#!/bin/bash
# Run a judge over whatever checkpoint generations exist, repeatedly, until the RLHF runs
# have finished and nothing is left to judge. The GPUs are shared with other jobs, so each
# pass waits until the chosen GPU has enough free memory for the judge weights.
# usage: judge_loop.sh <judge> <min_free_MiB> <main_steps or ""> <ablation_steps> <shard> <n_shards> <gpu> <batch>
J=$1; NEED=$2; MAIN_STEPS=$3; ABL_STEPS=$4; SH=${5:-0}; NSH=${6:-1}; G=$7; BS=${8:-32}
MAIN=seed0_beta0.05,seed1_beta0.05,seed2_beta0.05; ABL=seed0_beta0.5
cd "$(dirname "$0")"
while true; do
  running=$(pgrep -u "$USER" -f "python rlhf.py --seed" | wc -l)
  FREE=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$FREE" -lt "$NEED" ]; then sleep 30; continue; fi
  echo "[loop] $(date +%T) judge=$J shard=$SH/$NSH gpu=$G free=$FREE rlhf_running=$running"
  for spec in "$MAIN|$MAIN_STEPS" "$ABL|$ABL_STEPS"; do
    tags=${spec%%|*}; steps=${spec##*|}
    CUDA_VISIBLE_DEVICES=$G python judge.py --judge $J --tags $tags ${steps:+--steps $steps} --bs $BS \
      --shard $SH --n_shards $NSH --out ../results/judge/${J}_shard$SH.jsonl 2>&1 | grep -E "items to judge|done|Traceback|Error"
  done
  if [ "$running" -eq 0 ]; then echo "[loop] all runs were finished before this pass; exiting"; break; fi
  sleep 45
done
