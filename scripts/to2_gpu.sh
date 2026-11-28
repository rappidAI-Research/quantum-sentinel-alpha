#!/usr/bin/env bash
# Training Optimization v2 on the GPU machine: docs/GPU_RUNBOOK.md § 17, docs/TRAINING_OPTIMIZATION_V2.md.
# NOT APPROVED YET: run nothing on a GPU before the owner approves the stage. Each stage ends at a decision point.
#
#   scripts/to2_gpu.sh f0                      # no training: v1 artifacts re-analysed, E2 on the training precision,
#                                              # exact averages of E2 checkpoints, selection (decision point D0)
#   scripts/to2_gpu.sh train <pair_rank | lr5e-5 | pair_rank_lr5e-5>     # F1, F2, F3 (F3 only after D1 and D2)
#   scripts/to2_gpu.sh evaluate <pair_rank | lr5e-5 | pair_rank_lr5e-5>  # scores, calibration, screening, generation
#   scripts/to2_gpu.sh select                  # rule v2 over V6, v1, v2 runs and the checkpoint averages
#   scripts/to2_gpu.sh final                   # the final test, once (refused if v1 or v2 already ran it)
#   DRY_RUN=1 scripts/to2_gpu.sh f0            # print the commands, run nothing
#
# The helpers come from scripts/to1_gpu.sh (sourced with TAG=to2): finished outputs are skipped on a rerun, an
# interrupted training run resumes from its last checkpoint, and final_test / the reserved holdout are read only
# by `final`. Backup afterwards: docs/GPU_RUNBOOK.md § 17 step 6.
set -euo pipefail
TAG=to2
EXPERIMENTS=(pair_rank lr5e-5 pair_rank_lr5e-5)  # the last one only if F1 and F2 both pass (D3)
# shellcheck source=scripts/to1_gpu.sh
. "$(dirname "$0")/to1_gpu.sh"

V1=runs/eval/to1
E2RUN=runs/train/qsa-qlora-ornith15-9b-v6-to1-decision_w10
E2LABEL=qsa-qlora-ornith15-9b-v6-to1-decision_w10__step-000200
AVERAGE_ENDS=(200 250 300 346)  # pre-registered: exact averages of 5 checkpoints (50 steps) ending at these steps
NF4_STEPS=(160 200 240 330)     # E2 checkpoints scored on the training precision (NF4, double quantization)

# files matching a pattern; in DRY_RUN the pattern itself, so the printed commands stay readable
matching() {
  local found=()
  # shellcheck disable=SC2206
  found=($1)
  if [ ${#found[@]} -eq 0 ] && dry; then found=("$1"); fi
  MATCHED=("${found[@]}")
}

analyze_into() {  # analyze_into OUTPUT CASES PATTERN
  matching "$3"
  if [ ${#MATCHED[@]} -eq 0 ]; then
    echo "nothing matches $3: $1 skipped" >&2
    return 0
  fi
  run_once "$1" python -m sentinel.eval.decision_scores analyze "${MATCHED[@]}" --cases "$2" --output "$1"
}

f0() {
  local O=$TO1/f0 E end s
  run mkdir -p "$O/avg"
  # 1. what the labels teach and where the sets differ (CPU)
  run_once "$O/supervision_audit.json" python -m sentinel.train.supervision_audit --workdir "$G" --output "$O/supervision_audit.json"
  run_once "$O/calibration.eval_cases.jsonl" python -m sentinel.eval.decision_scores cases --records "$CAL" --output "$O/calibration.eval_cases.jsonl"
  # 2. the v1 artifacts with the v2 analysis: ROC-AUC, pairs, flips, sources, true-positive quality (CPU)
  for E in e0 decision_w10 detection_only; do
    analyze_into "$O/$E-dev-scores.json" "$DEV/dev.eval_cases.jsonl" "$V1/$E/dev-bf16/*.json"
    analyze_into "$O/$E-calibration-scores.json" "$O/calibration.eval_cases.jsonl" "$V1/$E/cal-bf16/*.json"
    analyze_into "$O/$E-dev-runs.json" "$DEV/dev.eval_cases.jsonl" "$V1/$E/runs/*.json"
  done
  # gradient norms and clipped steps of every step of E2 and E3 (CPU, from the training logs)
  for E in decision_w10 detection_only; do
    run_once "$O/$E-diagnostics.json" python -m sentinel.train.diagnostics --config "configs/train/qlora_ornith15_9b_v6_to1_$E.yaml" \
      --log "runs/train-to1-$E.log" --window 1-346 --output "$O/$E-diagnostics.json"
  done
  # 3. E2 step 200: do the greedy and the calibrated run decide every case alike? (CPU)
  if dry || [ -f "$V1/decision_w10/runs/$E2LABEL-greedy.json" ]; then
    run_once "$O/e2-step200-greedy-vs-calibrated.txt" sh -c "python -m sentinel.eval.compare '$V1/decision_w10/runs/$E2LABEL-greedy.json' '$V1/decision_w10/runs/$E2LABEL-calibrated.json' > '$O/e2-step200-greedy-vs-calibrated.txt' || true"
  else
    echo "no greedy run of E2 step 200 in $V1/decision_w10/runs: generate it with run_model (runbook § 17 step 1)" >&2
  fi
  # 4. E2 on the precision it was trained on (GPU, teacher-forced)
  local nf4=()
  for s in "${NF4_STEPS[@]}"; do nf4+=(--adapter "$(printf '%s/checkpoints/step-%06d/adapter' "$E2RUN" "$s")"); done
  score_dir "$O/e2-dev-nf4dq" --quantization bnb-4bit-dq "${nf4[@]}" --cases "$DEV/dev.eval_cases.jsonl"
  analyze_into "$O/e2-dev-nf4dq.json" "$DEV/dev.eval_cases.jsonl" "$O/e2-dev-nf4dq/*.json"
  # 5. exact averages of E2 checkpoints (CPU to build; GPU to score), then calibration, screening, generation
  local averaged=()
  for end in "${AVERAGE_ENDS[@]}"; do
    s=$(printf '%06d' "$end")
    run_once "$E2RUN/averaged/avg5-step-$s/adapter/adapter_model.safetensors" python -m sentinel.train.average --run "$E2RUN" --end-step "$end" --window 5
    averaged+=(--adapter "$E2RUN/averaged/avg5-step-$s/adapter")
  done
  score_dir "$O/avg/dev-bf16" "${averaged[@]}" --cases "$DEV/dev.eval_cases.jsonl"
  score_dir "$O/avg/cal-bf16" "${averaged[@]}" --records "$CAL"
  calibrate_all "$O/avg"
  analyze_into "$O/avg/analysis.json" "$DEV/dev.eval_cases.jsonl" "$O/avg/dev-bf16/*.json"
  shortlist "$O/avg"
  generate_shortlist "$O/avg"
  # 6. rule v2 over V6, v1 and the averages (decision point D0, docs/TRAINING_OPTIMIZATION_V2.md § 6)
  matching "runs/eval/dev-v6/runs/*.json"; local runs=("${MATCHED[@]}")
  matching "$V1/*/runs/*.json"; runs+=("${MATCHED[@]}")
  matching "$O/avg/runs/*.json"; runs+=("${MATCHED[@]}")
  select_into "$O/selection.json" "${runs[@]}"
}

select_all() {
  local runs=()
  matching "runs/eval/dev-v6/runs/*.json"; runs+=("${MATCHED[@]}")
  matching "$V1/*/runs/*.json"; runs+=("${MATCHED[@]}")
  matching "$TO1/f0/avg/runs/*.json"; runs+=("${MATCHED[@]}")
  matching "$TO1/*/runs/*.json"; runs+=("${MATCHED[@]}")
  select_into "$TO1/selection.json" "${runs[@]}"
}

one_experiment() {
  local E=${1:-}
  if [[ ! " ${EXPERIMENTS[*]} " == *" $E "* ]]; then
    echo "usage: $0 $2 <${EXPERIMENTS[*]}>" >&2
    exit 2
  fi
  EXPERIMENTS=("$E")
}

case "${1:-}" in
  f0) f0 ;;
  train)
    one_experiment "${2:-}" train
    train
    ;;
  evaluate)
    one_experiment "${2:-}" evaluate
    evaluate "$2"
    ;;
  select) select_all ;;
  final)
    if [ -e runs/eval/to1-final.done ]; then
      echo "the final test already ran under Training Optimization v1 (runs/eval/to1-final.done); it runs once" >&2
      exit 1
    fi
    final
    ;;
  *)
    sed -n '2,11p' "$0" >&2
    exit 2
    ;;
esac
