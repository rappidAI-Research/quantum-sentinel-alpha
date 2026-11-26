#!/usr/bin/env bash
# Training Optimization v1 on the GPU machine: docs/GPU_RUNBOOK.md § 16, approved by the owner on 2026-11-26.
#
#   scripts/to1_gpu.sh all                 # check, E0, E1-E4 training and evaluation, selection, final test (once)
#   scripts/to1_gpu.sh check | e0 | train | select | final
#   scripts/to1_gpu.sh evaluate <baseline_dense | decision_w10 | detection_only | lr5e-5>
#   DRY_RUN=1 scripts/to1_gpu.sh all       # print the commands, run nothing
#
# Run it under nohup or tmux; everything is logged to stderr. A rerun continues where it stopped: finished
# outputs are skipped, and an interrupted training run resumes from its last checkpoint. final_test and the
# reserved holdout are touched only by the `final` stage, once (runs/eval/to1-final.done); decision scores,
# calibration and screening never read them. Backup afterwards: docs/GPU_RUNBOOK.md § 16 step 9.
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."

REV=489cb97981b8654bcfcf30ce1f94ed1b62e07b53
M=ornith-ai/Ornith-1.5-9B
G=artifacts/garden/seed_v6
DEV=$G/development
CAL=$G/validation.jsonl
RUN=runs/train/qsa-qlora-ornith15-9b-v6
V0=runs/train/qsa-qlora-ornith15-9b-v0
TO1=runs/eval/to1
FINAL_DONE=runs/eval/to1-final.done
EXPERIMENTS=(baseline_dense decision_w10 detection_only lr5e-5)
# what the reported V6 run was trained on (docs/TRAINING_OPTIMIZATION_V1.md § 3)
V6_DATA_SHA256=51ab4c9d54ea8139311877533a9aadf8b831ce4e70337279effa16cef7a12e9a
V6_CONFIG_SHA256=f9a0edf6dc25743b981720392b7c6cfb59c48fcd2ab4aa92c0d312da167d27d4

dry() { [ -n "${DRY_RUN:-}" ]; }

run() {
  echo "+ $*" >&2
  dry || "$@"
}

# run_once OUTPUT CMD...: skipped when OUTPUT exists (every tool here writes its output file last)
run_once() {
  local out=$1
  shift
  if [ -e "$out" ]; then
    echo "= $out exists, skipped" >&2
    return 0
  fi
  run "$@"
}

# decision scores of several adapters into a directory, made atomically (a partial directory is never kept)
score_dir() {
  local out=$1
  shift
  if [ -d "$out" ]; then
    echo "= $out exists, skipped" >&2
    return 0
  fi
  dry || rm -rf "$out.tmp"
  run python -m sentinel.eval.decision_scores score --model "$M" --revision "$REV" "$@" --output "$out.tmp"
  run mv "$out.tmp" "$out"
}

# ADAPTER_ARGS: --adapter for every completed checkpoint of a run (an interrupted save, step-NNNNNN.tmp, is not one)
adapter_args() {
  ADAPTER_ARGS=()
  local c
  for c in "$1"/checkpoints/step-[0-9][0-9][0-9][0-9][0-9][0-9]; do
    ADAPTER_ARGS+=(--adapter "$c/adapter")
  done
  if [ ${#ADAPTER_ARGS[@]} -eq 0 ]; then
    if dry; then
      ADAPTER_ARGS=(--adapter "$1/checkpoints/step-NNNNNN/adapter")
    else
      echo "no checkpoints in $1" >&2
      exit 1
    fi
  fi
}

calibrate_all() {
  local O=$1 f
  dry && echo "+ decision_scores calibrate --max-fpr 0.15 for every file in $O/cal-bf16 -> $O/calibration" >&2
  run mkdir -p "$O/calibration"
  for f in "$O"/cal-bf16/*.json; do
    run_once "$O/calibration/$(basename "$f")" python -m sentinel.eval.decision_scores calibrate "$f" --max-fpr 0.15 \
      --output "$O/calibration/$(basename "$f")" > /dev/null || echo "no bias within FPR 0.15: $f" >&2
  done
}

shortlist() {
  local O=$1
  local cals=("$O"/calibration/*.json)
  echo "+ python -m sentinel.eval.decision_scores screen $O/dev-bf16/*.json --calibrations $O/calibration/*.json --output $O/screen.json > $O/shortlist.tsv" >&2
  dry && return 0
  python -m sentinel.eval.decision_scores screen "$O"/dev-bf16/*.json --calibrations "${cals[@]}" --output "$O/screen.json" > "$O/shortlist.tsv"
  cat "$O/shortlist.tsv" >&2
}

# generation runs (run_model) of every shortlisted candidate; "calibrated" skips the greedy ones
generate_shortlist() {
  local O=$1 only=${2:-}
  if dry; then
    echo "+ python -m sentinel.eval.run_model ... --adapter <shortlisted> [--calibration <its file>] for every line of $O/shortlist.tsv" >&2
    return 0
  fi
  mkdir -p "$O/runs"
  local mode adapter cal label out
  while IFS=$'\t' read -r mode adapter cal label; do
    if [ "$only" = calibrated ] && [ "$mode" != calibrated ]; then
      continue
    fi
    local extra=()
    if [ "$mode" = calibrated ]; then
      extra=(--calibration "$cal")
    fi
    out="$O/runs/${label//\//__}-$mode.json"
    run_once "$out" python -m sentinel.eval.run_model --model "$M" --revision "$REV" --cases "$DEV/dev.eval_cases.jsonl" \
      --adapter "$adapter" "${extra[@]}" --output "$out" < /dev/null
  done < "$O/shortlist.tsv"
}

# select_checkpoint exits 1 when nothing is eligible; that is a result, not a failure
select_into() {
  local out=$1 status=0
  shift
  run python -m sentinel.eval.select_checkpoint --rule 2 --devset "$DEV/manifest.json" "$@" --output "$out" || status=$?
  if [ "$status" -gt 1 ]; then
    exit "$status"
  fi
}

training_log() {
  if [ -f "$1" ]; then echo --log "$1"; else echo --run "$2/run.json"; fi
}

check() {
  run python -m sentinel.foundation
  if dry; then
    echo "+ (identity of $RUN/run.json checked against the V6 data and config hashes)" >&2
    return 0
  fi
  python - "$RUN/run.json" "$V6_DATA_SHA256" "$V6_CONFIG_SHA256" <<'EOF'
import json
import sys

run = json.load(open(sys.argv[1], encoding="utf-8"))
found = (run["data"]["sha256"], run["config"]["sha256"])
if found != (sys.argv[2], sys.argv[3]):
    sys.exit(f"V6 run.json has data {found[0]} and config {found[1]}; the analysis assumes {sys.argv[2]} / {sys.argv[3]}")
print("V6 run: data and config hashes as analysed")
EOF
}

e0() {
  local O=$TO1/e0
  run mkdir -p "$O"
  local tok="<pinned tokenizer files>"
  dry || tok=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$M', revision='$REV', allow_patterns=['*.json', '*.txt', '*.jinja']))")
  # shellcheck disable=SC2046
  run_once "$O/diagnostics.json" python -m sentinel.train.diagnostics --config configs/train/qlora_ornith15_9b_v6.yaml --tokenizer-dir "$tok" \
    $(training_log runs/train-v6.log "$RUN") --window 126-175 --output "$O/diagnostics.json"
  if ! dry; then  # the local data must be the data V6 was trained on
    python -c "import json, sys; d = json.load(open('$O/diagnostics.json'))['data_sha256']; sys.exit(0 if d == '$V6_DATA_SHA256' else f'local data {d} is not the V6 data')"
  fi

  adapter_args "$RUN"
  local adapters=("${ADAPTER_ARGS[@]}")
  if dry || [ -d "$V0/checkpoints/step-000200/adapter" ]; then
    adapters+=(--adapter "$V0/checkpoints/step-000200/adapter")
  fi
  score_dir "$O/dev-bf16" "${adapters[@]}" --cases "$DEV/dev.eval_cases.jsonl"
  run_once "$O/dev-bf16/foundation.json" python -m sentinel.eval.decision_scores score --model "$M" --revision "$REV" \
    --cases "$DEV/dev.eval_cases.jsonl" --output "$O/dev-bf16/foundation.json"
  score_dir "$O/cal-bf16" "${adapters[@]}" --records "$CAL"
  local nf4=() s
  for s in 125 150 175 200; do
    nf4+=(--adapter "$(printf '%s/checkpoints/step-%06d/adapter' "$RUN" "$s")")
  done
  score_dir "$O/dev-nf4" --quantization bnb-4bit-dq "${nf4[@]}" --cases "$DEV/dev.eval_cases.jsonl"

  local generated=(runs/eval/dev-v6/runs/*.json)
  dry && generated=("runs/eval/dev-v6/runs/*.json")
  run_once "$O/analysis-bf16.json" python -m sentinel.eval.decision_scores analyze "$O"/dev-bf16/*.json --cases "$DEV/dev.eval_cases.jsonl" --output "$O/analysis-bf16.json"
  run_once "$O/analysis-nf4.json" python -m sentinel.eval.decision_scores analyze "$O"/dev-nf4/*.json --cases "$DEV/dev.eval_cases.jsonl" --output "$O/analysis-nf4.json"
  if [ ${#generated[@]} -gt 0 ]; then
    run_once "$O/analysis-generated.json" python -m sentinel.eval.decision_scores analyze "${generated[@]}" --cases "$DEV/dev.eval_cases.jsonl" --output "$O/analysis-generated.json"
  else
    echo "no V6 run_model artifacts in runs/eval/dev-v6/runs (§ 15): analysis of generated decisions skipped" >&2
  fi

  calibrate_all "$O"
  shortlist "$O"
  generate_shortlist "$O" calibrated  # the greedy V6 runs exist already (§ 15)
  select_into "$O/selection.json" "${generated[@]}" "$O"/runs/*.json
}

train() {
  local E R resume
  for E in "${EXPERIMENTS[@]}"; do
    run python -m sentinel.train --config "configs/train/qlora_ornith15_9b_v6_to1_$E.yaml" --dry-run --check-tokens
  done
  for E in "${EXPERIMENTS[@]}"; do
    R=runs/train/qsa-qlora-ornith15-9b-v6-to1-$E
    if [ -f "$R/adapter/adapter_model.safetensors" ]; then
      echo "= $R finished, skipped" >&2
      continue
    fi
    resume=()
    if [ -n "$(echo "$R"/checkpoints/step-[0-9][0-9][0-9][0-9][0-9][0-9])" ]; then
      resume=(--resume)  # an interrupted run continues from its last verified checkpoint
    fi
    echo "+ python -m sentinel.train --config configs/train/qlora_ornith15_9b_v6_to1_$E.yaml ${resume[*]} >> runs/train-to1-$E.log" >&2
    dry || python -m sentinel.train --config "configs/train/qlora_ornith15_9b_v6_to1_$E.yaml" "${resume[@]}" >> "runs/train-to1-$E.log" 2>&1
  done
}

evaluate() {
  local E=$1
  local R=runs/train/qsa-qlora-ornith15-9b-v6-to1-$E O=$TO1/$1
  run mkdir -p "$O"
  adapter_args "$R"
  score_dir "$O/dev-bf16" "${ADAPTER_ARGS[@]}" --cases "$DEV/dev.eval_cases.jsonl"
  score_dir "$O/cal-bf16" "${ADAPTER_ARGS[@]}" --records "$CAL"
  calibrate_all "$O"
  run_once "$O/analysis.json" python -m sentinel.eval.decision_scores analyze "$O"/dev-bf16/*.json --cases "$DEV/dev.eval_cases.jsonl" --output "$O/analysis.json"
  shortlist "$O"
  generate_shortlist "$O"
  # shellcheck disable=SC2046
  run_once "$O/diagnostics.json" python -m sentinel.train.diagnostics --config "configs/train/qlora_ornith15_9b_v6_to1_$E.yaml" \
    $(training_log "runs/train-to1-$E.log" "$R") --window 150-200 --output "$O/diagnostics.json"
}

select_all() {
  local runs=(runs/eval/dev-v6/runs/*.json "$TO1"/*/runs/*.json)
  dry && runs=("runs/eval/dev-v6/runs/*.json" "$TO1/*/runs/*.json")
  select_into "$TO1/selection.json" "${runs[@]}"
}

final() {
  if [ -e "$FINAL_DONE" ]; then
    echo "the final test has already run ($FINAL_DONE); it runs once" >&2
    exit 1
  fi
  if dry; then
    echo "+ run_model of the foundation and of the selected checkpoint (with its calibration file) on final_test and holdout_reserved, once" >&2
    return 0
  fi
  local sel cal set
  sel=$(python -c "import json; print(json.load(open('$TO1/selection.json'))['selected_adapter'] or '')")
  cal=$(python -c "import json; print(json.load(open('$TO1/selection.json'))['selected_calibration'] or '')")
  if [ -z "$sel" ]; then
    echo "no checkpoint is eligible under rule v2: nothing is selected, no final test" >&2
    return 0
  fi
  for set in final_test holdout_reserved; do
    mkdir -p "runs/eval/$set-to1"
    run_once "runs/eval/$set-to1/foundation.json" python -m sentinel.eval.run_model --model "$M" --revision "$REV" \
      --cases "$G/$set.eval_cases.jsonl" --output "runs/eval/$set-to1/foundation.json"
    run_once "runs/eval/$set-to1/selected.json" python -m sentinel.eval.run_model --model "$M" --revision "$REV" \
      --cases "$G/$set.eval_cases.jsonl" --adapter "$sel" ${cal:+--calibration "$cal"} --output "runs/eval/$set-to1/selected.json"
    python -m sentinel.eval.compare "runs/eval/$set-to1/foundation.json" "runs/eval/$set-to1/selected.json" | tee "runs/eval/$set-to1/compare.txt"
  done
  echo "selected $sel ${cal:+with $cal}" > "$FINAL_DONE"
}

case "${1:-}" in
  check) check ;;
  e0) e0 ;;
  train) train ;;
  evaluate)
    if [ -z "${2:-}" ]; then
      echo "usage: $0 evaluate <${EXPERIMENTS[*]}>" >&2
      exit 2
    fi
    evaluate "$2"
    ;;
  select) select_all ;;
  final) final ;;
  all)
    check
    e0
    train
    for E in "${EXPERIMENTS[@]}"; do evaluate "$E"; done
    select_all
    final
    echo "done; back up the results before terminating the instance (docs/GPU_RUNBOOK.md § 16 step 9)" >&2
    ;;
  *)
    sed -n '2,7p' "$0" >&2
    exit 2
    ;;
esac
