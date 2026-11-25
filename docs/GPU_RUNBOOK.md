# GPU Runbook: first Quantum Sentinel Alpha training run

Target: AWS EC2 `g6e.xlarge` in `us-east-1` (1 × NVIDIA L40S 48 GB, 4 vCPUs, 32 GiB RAM), On-Demand.
Foundation: `ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53` (docs/FOUNDATION.md).
Data: Sentinel Garden v3, 1,871 training and 187 validation examples (detection + patch).

**Verification status of this runbook.** Every repository command below ran offline in CI or on a CPU machine as far as possible: config, governance and data checks, tokenization with the real Ornith tokenizer, and the training loop on a tiny model of the same architecture. Nothing here has run on a GPU yet. Steps that need the GPU are marked **GPU**. The AWS commands were not executed from this repository.

## 0. Instance

- AMI: *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*. It ships the NVIDIA driver, CUDA and Docker, and Ubuntu 24.04 provides Python 3.12 (the repository needs ≥ 3.11). Find the current image ID:

  ```bash
  aws ec2 describe-images --region us-east-1 --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" \
    --query 'sort_by(Images,&CreationDate)[-1].[ImageId,Name]' --output text
  ```

- Root volume: 150 GB gp3. The full checkpoint download is 19.3 GB, the CUDA Python environment about 10 GB, the Garden git cache a few GB, plus runs and checkpoints.
- Only port 22 is needed. The model is public; no Hugging Face token is required.

## 1. GPU driver and CUDA

```bash
nvidia-smi                                  # expect: NVIDIA L40S with about 46 GB, a driver and CUDA version
python3 --version                           # >= 3.11
docker info --format '{{.ServerVersion}}'   # optional: only needed for the coding-regression sandbox
```

## 2. Repository at a pinned revision

```bash
git clone https://github.com/rappidAI-Research/quantum-sentinel-alpha.git
cd quantum-sentinel-alpha
git checkout main && git rev-parse HEAD     # record this commit; run.json records it too
```

Train only from a clean tree: the trainer refuses uncommitted changes unless `--allow-dirty` is passed.

## 3. Python environment

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e '.[train,quant,dev]'         # torch (CUDA wheel from PyPI), transformers>=5.12,<6, peft, accelerate, bitsandbytes, safetensors
python -c "import torch, transformers, peft, bitsandbytes; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), transformers.__version__, peft.__version__, bitsandbytes.__version__)"
```

`torch.cuda.is_available()` must print `True`.

The optional Gated DeltaNet kernels (`flash-linear-attention`, `causal-conv1d`) are not installed by this runbook. Without them transformers uses its slower PyTorch path. If you add them, use the same environment for the baseline and the adapter run: throughput differs, and the run artifacts record whether the kernels are present.

## 4. Offline checks (no GPU work)

```bash
python -m sentinel.foundation               # schema, registry, approvals, foundation pin, baseline candidate
pytest -q                                   # full suite; container-sandbox tests skip without Docker
```

## 5. Garden v3 materialization

```bash
python -m sentinel.garden materialize --config configs/garden/seed_v3.yaml
```

This rebuilds only the 1,464 training-approved records (684 advisories in 533 repositories) from the pinned upstream commits and writes `train.jsonl`, `validation.jsonl`, `train.chat.jsonl`, `train.patch.chat.jsonl` and `SHA256SUMS` to `artifacts/garden/seed_v3/`. It refuses on any content or supervision hash that differs from the committed manifest. From a warm cache it took about 1.5 minutes; on a fresh instance it first fetches the commits over the network, which has not been timed. The exports contain upstream code: keep them on the instance and do not publish them.

## 6. Training dry run

```bash
python -m sentinel.train --dry-run --check-tokens
```

Expected (verified on CPU with the real tokenizer):

```text
foundation ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53; train 1871 / validation 187 approved examples from 533 repositories (tasks detection, patch: ...); data sha256 c47e36167f2d
tokens train: 1871 kept, 0 dropped at 4096; median 764, max 2918, total 1,732,959, supervised 157,384
tokens validation: 187 kept, 0 dropped at 4096; median 793, max 2481, total 174,148, supervised 17,674
dry run: configuration, governance and data checks passed; no model loaded
```

## 7. Ornith foundation baseline (**GPU**, bf16, about 18 GiB of weights)

With the coding-regression sandbox (Docker and the pinned image, docs/CODING_SANDBOX.md):

```bash
docker pull python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
python -m sentinel.eval.run_model --bakeoff configs/model/baseline_ornith15.yaml --candidate ornith15-9b \
    --output runs/eval/ornith15-9b-baseline.json
```

Without Docker, run security only, and then skip it in the adapter run too:

```bash
python -m sentinel.eval.run_model --bakeoff configs/model/baseline_ornith15.yaml --candidate ornith15-9b \
    --skip-coding-regression --output runs/eval/ornith15-9b-baseline.json
```

The first run downloads the full checkpoint (19.3 GB) into the Hugging Face cache. Settings: QuickEval v1 (86 cases), prompt `quickeval-review-v2`, greedy, `max_new_tokens 384`, non-thinking prompt, stop at `<|im_end|>`. After loading, the run hashes every loaded checkpoint file against the pinned digests (one extra read of the 19.3 GB; `runtime.revision_verification` in the artifact). Without that proof `resolved_revision` stays empty and the run is not reportable. Look at `invalid` predictions in the artifact. They show how well the untuned model follows the JSON contract, which has not been measured yet.

## 8. QLoRA training run (**GPU**, 4-bit NF4 base, bf16 compute)

Optional calibration first (20 optimizer steps; separate config outside the repository, so the tree stays clean, and a separate output directory):

```bash
sed -e 's/^run_name: .*/run_name: qsa-qlora-ornith15-9b-calibration/' \
    -e 's#^output_dir: .*#output_dir: runs/train/qsa-qlora-ornith15-9b-calibration#' \
    -e 's/^  max_steps: null/  max_steps: 20/' \
    configs/train/qlora_ornith15_9b.yaml > ~/calibration.yaml
python -m sentinel.train --config ~/calibration.yaml
python -c "import json; r=json.load(open('runs/train/qsa-qlora-ornith15-9b-calibration/run.json')); print(r['status'], r.get('peak_vram_gib'), r['state']['history'][-1])"   # status, peak VRAM, last step (loss, tokens/s)
```

The real run:

```bash
mkdir -p runs && python -m sentinel.train 2>&1 | tee runs/train-ornith15-9b-v0.log
```

Defaults: 2 epochs, micro-batch 1 × 16 accumulation (about 117 optimizer steps per epoch for 1,871 examples), lr 1e-4 cosine with 3 % warmup, LoRA rank 16 / alpha 32 / dropout 0.05 on 200 modules (40,108,032 trainable parameters), max 4,096 tokens, gradient checkpointing. Validation and checkpoints every 50 steps, the last 3 kept.

Memory, estimated and **not measured**: about 3.5 GB for the 4-bit layers, about 8 GB for the embedding and `lm_head` (kept unquantized and cast to fp32 for k-bit training), under 1 GB for the LoRA weights and AdamW states, and about 9 GB peak for logits and loss on the longest example (2,918 tokens × 248,320 vocabulary). That adds up to roughly 20–25 GB, plausible for 48 GB. Throughput, and therefore run time, is unknown until the calibration run.

Exit codes: 0 completed, 1 failed, 2 governance/data refusal, 3 CUDA out of memory, 4 interrupted. `run.json` is written in every outcome.

## 9. Checkpoint recovery

- Run training inside `tmux` so a lost SSH session does not stop it. After exit code 3 or 4: `python -m sentinel.train --resume`. It continues from the latest complete checkpoint and refuses if the data, config, base model or revision differ (`provenance.json`, `checksums.json`).
- On OOM: lower `micro_batch_size` (it is already 1) or `max_seq_len`; changing the config starts a new run, because resume requires identical inputs.
- Moving to a new instance: same commit, steps 3 and 5, restore `runs/train/qsa-qlora-ornith15-9b-v0/`, then `--resume`.

## 10. Adapter evaluation (**GPU**)

Same config, same candidate, same coding-regression choice as the baseline:

```bash
python -m sentinel.eval.run_model --bakeoff configs/model/baseline_ornith15.yaml --candidate ornith15-9b \
    --adapter runs/train/qsa-qlora-ornith15-9b-v0/adapter --output runs/eval/ornith15-9b-qlora-v0.json
python -m sentinel.eval.compare runs/eval/ornith15-9b-baseline.json runs/eval/ornith15-9b-qlora-v0.json
```

`--adapter` refuses, before anything is downloaded, an adapter trained for another model, revision or prompt. Add `--skip-coding-regression` to both runs or to neither; `compare` refuses mismatched contracts.

Optional patch check on the reserved holdout (offline scoring, nothing is executed), before and after. `materialize` writes only the training exports. The holdout patch cases come from a full reconstruction, which rebuilds all 7,751 candidates from the pinned upstream commits and takes much longer. It rewrites the training exports with identical bytes and checks every hash; run it after training, not during a run:

```bash
python -m sentinel.garden reconstruct --config configs/garden/seed_v3.yaml   # writes artifacts/garden/seed_v3/holdout_reserved.patch_cases.jsonl
python -m sentinel.eval.patch --model ornith-ai/Ornith-1.5-9B --revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53 \
    --cases artifacts/garden/seed_v3/holdout_reserved.patch_cases.jsonl --output runs/eval/ornith15-9b-patch-baseline.json
python -m sentinel.eval.patch --model ornith-ai/Ornith-1.5-9B --revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53 \
    --adapter runs/train/qsa-qlora-ornith15-9b-v0/adapter \
    --cases artifacts/garden/seed_v3/holdout_reserved.patch_cases.jsonl --output runs/eval/ornith15-9b-patch-qlora-v0.json
```

## 11. Save results before terminating the instance

The instance disk is lost on termination. Keep the adapter, run records and evaluation artifacts (not the Garden exports, which contain upstream code):

```bash
tar czf qsa-ornith15-9b-v0-results.tgz runs/train/qsa-qlora-ornith15-9b-v0/adapter runs/train/qsa-qlora-ornith15-9b-v0/run.json runs/eval runs/train-ornith15-9b-v0.log
sha256sum qsa-ornith15-9b-v0-results.tgz
# either a private S3 bucket (instance role with s3:PutObject) ...
aws s3 cp qsa-ornith15-9b-v0-results.tgz s3://<private-bucket>/quantum-sentinel-alpha/
# ... or copy to your own machine
scp ubuntu@<instance-ip>:~/quantum-sentinel-alpha/qsa-ornith15-9b-v0-results.tgz .
```

If the run is not finished, also copy `runs/train/qsa-qlora-ornith15-9b-v0/checkpoints/` so it can be resumed elsewhere. Verify the copy (checksum) before terminating, then terminate the instance so it stops billing.

## 12. Post-training evaluation: development set, checkpoint selection, final test (**GPU**)

Roles and the selection rule: `docs/EVALUATION.md`. Every run below loads the bf16 foundation (`run_model` direct mode, same generation settings as the baseline config). The git tree must stay clean for runs to be reportable.

Update the code and check that the training outputs are present (restore them from the backup if not):

```bash
cd ~/quantum-sentinel-alpha && git pull --ff-only && git log --oneline -1 && . .venv/bin/activate && pip install -e '.[train,quant,dev]'
python -m sentinel.foundation
ls runs/train/qsa-qlora-ornith15-9b-v0/adapter runs/train/qsa-qlora-ornith15-9b-v0/checkpoints/
```

Build the development set from the verified training exports (`materialize` first if `artifacts/garden/seed_v3/validation.jsonl` is missing). Expected: 133 detection cases (abstain 16, finding 63, no_finding 54), 54 patch cases, detection `cases_sha256` `397f5cb2fb6e…`:

```bash
python -m sentinel.eval.devset --config configs/garden/seed_v3.yaml --output-dir runs/devset
```

Foundation and every retained checkpoint on the development detection cases (`step-000234` holds the final weights):

```bash
REV=489cb97981b8654bcfcf30ce1f94ed1b62e07b53; RUN=runs/train/qsa-qlora-ornith15-9b-v0
python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases runs/devset/dev.eval_cases.jsonl --output runs/eval/dev/foundation.json
for ckpt in $RUN/checkpoints/step-*; do python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases runs/devset/dev.eval_cases.jsonl --adapter $ckpt/adapter --output runs/eval/dev/$(basename $ckpt).json; done
```

Patch generation on the development patch cases (information only, not part of the rule):

```bash
python -m sentinel.eval.patch --model ornith-ai/Ornith-1.5-9B --revision $REV --cases runs/devset/dev.patch_cases.jsonl --output runs/eval/dev-patch/foundation.json
for ckpt in $RUN/checkpoints/step-*; do python -m sentinel.eval.patch --model ornith-ai/Ornith-1.5-9B --revision $REV --cases runs/devset/dev.patch_cases.jsonl --adapter $ckpt/adapter --output runs/eval/dev-patch/$(basename $ckpt).json; done
```

Apply the pre-registered rule (exit code 0: a checkpoint was selected):

```bash
python -m sentinel.eval.select_checkpoint --devset runs/devset/manifest.json runs/eval/dev/*.json --patch runs/eval/dev-patch/*.json --output runs/eval/dev/selection.json
SEL=$(python -c "import json; print(json.load(open('runs/eval/dev/selection.json'))['selected_adapter'])"); echo $SEL
```

Regression check of the selected adapter on QuickEval v1 and `coding-v1`, the same contract as the baseline. Write to a new file: the earlier adapter artifact stays as recorded.

```bash
python -m sentinel.eval.run_model --bakeoff configs/model/baseline_ornith15.yaml --candidate ornith15-9b --adapter $SEL --output runs/eval/quickeval/selected.json
python -m sentinel.eval.compare runs/eval/ornith15-9b-baseline.json runs/eval/quickeval/selected.json
```

Only after the selection is fixed: the final independent test on `holdout_reserved`, once, for the foundation and the selected adapter. `reconstruct` rebuilds all candidates from upstream git to write the holdout cases; its run time has not been measured.

```bash
python -m sentinel.garden reconstruct --config configs/garden/seed_v3.yaml
python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases artifacts/garden/seed_v3/holdout_reserved.eval_cases.jsonl --output runs/eval/holdout/foundation.json
python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases artifacts/garden/seed_v3/holdout_reserved.eval_cases.jsonl --adapter $SEL --output runs/eval/holdout/selected.json
python -m sentinel.eval.compare runs/eval/holdout/foundation.json runs/eval/holdout/selected.json
```

Save `runs/eval/` and `runs/devset/manifest.json` together with the training outputs (§ 11). Do not save the case files: they contain upstream code.

## 13. Garden v4: diagnosis, training (A and B), selection, final test (**GPU**)

> **Superseded by § 14** (owner decision of 2026-11-25: fresh training on Garden v5 only; the continuation config was removed). Step 1 (the diagnosis of `step-000200`) remains useful and can be run on the v5 development set instead.

Data and roles: `docs/GARDEN_V4.md`. Approaches and settings: `docs/TRAINING_V4.md`. Selection: `docs/EVALUATION.md`. Keep the first run's outputs (`runs/train/qsa-qlora-ornith15-9b-v0/`, including `checkpoints/step-000200`) in place: approach A starts from them and refuses a modified checkpoint.

Update, install, offline checks:

```bash
cd ~/quantum-sentinel-alpha && git pull --ff-only && git log --oneline -1 && . .venv/bin/activate && pip install -e '.[train,quant,dev]'
python -m sentinel.foundation
```

Evaluation sets (development, final test, previous holdout) and training exports, rebuilt only for the records they need and verified against the manifest. The development set should have 176 detection cases (abstain 23, finding 87, no_finding 66):

```bash
python -m sentinel.garden materialize-eval --config configs/garden/seed_v4.yaml
cd artifacts/garden/seed_v4 && sha256sum -c SHA256SUMS.evaluation && cd ~/quantum-sentinel-alpha
python -m sentinel.garden materialize --config configs/garden/seed_v4.yaml
cd artifacts/garden/seed_v4 && sha256sum -c SHA256SUMS && cd ~/quantum-sentinel-alpha
cat artifacts/garden/seed_v4/development/manifest.json
```

**Step 1 – diagnosis before any training (about 30 min of GPU time).** Evaluate the foundation and the first run's `step-000200` on the v4 development set. Its contexts no longer carry the abstention-only sentence, so the abstention results of `step-000200` show whether it relied on that sentence. These two runs are also the references for A and B:

```bash
REV=489cb97981b8654bcfcf30ce1f94ed1b62e07b53; DEV=artifacts/garden/seed_v4/development
python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases $DEV/dev.eval_cases.jsonl --output runs/eval/dev-v4/runs/foundation.json
python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases $DEV/dev.eval_cases.jsonl --adapter runs/train/qsa-qlora-ornith15-9b-v0/checkpoints/step-000200/adapter --output runs/eval/dev-v4/runs/v0-step-000200.json
python -m sentinel.garden analyze --config configs/garden/seed_v4.yaml --output runs/eval/dev-v4/diagnosis.json --runs runs/eval/dev-v4/runs/foundation.json runs/eval/dev-v4/runs/v0-step-000200.json --cases $DEV/dev.eval_cases.jsonl
```

`diagnosis.json` → `development_runs` holds the confusion matrix (expected → predicted) and recall by CWE family and language. Bring it back together with the first run's `run.json` and `runs/eval/dev/*.json` from the v3 session.

**Step 2 – dry runs.** A must find and accept the step-200 checkpoint:

```bash
python -m sentinel.train --config configs/train/qlora_ornith15_9b_v4_fresh.yaml --dry-run --check-tokens
python -m sentinel.train --config configs/train/qlora_ornith15_9b_v4_continue.yaml --dry-run
```

**Step 3 – training.** B: about 244 optimizer steps. A: about 122. The first run's 234 steps are the reference for the run time. Run them one after the other on one GPU:

```bash
python -m sentinel.train --config configs/train/qlora_ornith15_9b_v4_fresh.yaml
python -m sentinel.train --config configs/train/qlora_ornith15_9b_v4_continue.yaml
```

**Step 4 – development evaluation of every checkpoint of both runs, then the rule:**

```bash
for RUN in runs/train/qsa-qlora-ornith15-9b-v4-fresh runs/train/qsa-qlora-ornith15-9b-v4-continue; do for ckpt in $RUN/checkpoints/step-*; do python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases $DEV/dev.eval_cases.jsonl --adapter $ckpt/adapter --output runs/eval/dev-v4/runs/$(basename $RUN)-$(basename $ckpt).json; done; done
python -m sentinel.eval.select_checkpoint --devset $DEV/manifest.json runs/eval/dev-v4/runs/*.json --output runs/eval/dev-v4/selection.json
SEL=$(python -c "import json; print(json.load(open('runs/eval/dev-v4/selection.json'))['selected_adapter'])"); echo $SEL
python -m sentinel.garden analyze --config configs/garden/seed_v4.yaml --output runs/eval/dev-v4/analysis.json --runs runs/eval/dev-v4/runs/*.json --cases $DEV/dev.eval_cases.jsonl
```

The `v0-step-000200` run takes part as a candidate of its own run. If the rule selects it, neither v4 run beat the first run on the development set. That is also a result. Optional and for information: patch cases with `sentinel.eval.patch --cases $DEV/dev.patch_cases.jsonl`, as in § 12.

**Step 5 – regression check** of `$SEL` on QuickEval v1 and `coding-v1`. Use the same commands as § 12, with a new output file.

**Step 6 – final test, once, after the selection is fixed.** `final_test` has 45 cases from 23 repositories that no evaluation of ours has used. `previous_holdout` has 431 cases. That is Garden v3's holdout reservation, rebuilt with v4 contexts: untouched only if § 12's holdout runs were never made. Report the two separately:

```bash
for SET in final_test holdout_reserved; do
  python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases artifacts/garden/seed_v4/$SET.eval_cases.jsonl --output runs/eval/$SET-v4/foundation.json
  python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases artifacts/garden/seed_v4/$SET.eval_cases.jsonl --adapter $SEL --output runs/eval/$SET-v4/selected.json
  python -m sentinel.eval.compare runs/eval/$SET-v4/foundation.json runs/eval/$SET-v4/selected.json
done
```

Save `runs/eval/` and both training runs' `run.json` and `checkpoints/*/provenance.json` with the outputs (§ 11). Do not save case files or records: they contain upstream code.

## 14. Garden v5: fresh training, selection, final test (**GPU**)

Data: `docs/GARDEN_V5.md`. Settings: `docs/TRAINING_V5.md`. Train only after the approval decisions in `docs/GARDEN_V5.md` § Summary (otherwise the data is almost the first run's).

```bash
cd ~/quantum-sentinel-alpha && git pull --ff-only && git log --oneline -1 && . .venv/bin/activate && pip install -e '.[train,quant,dev]'
python -m sentinel.foundation
python -m sentinel.garden materialize --config configs/garden/seed_v5.yaml
cd artifacts/garden/seed_v5 && sha256sum -c SHA256SUMS && cd ~/quantum-sentinel-alpha
python -m sentinel.garden materialize-eval --config configs/garden/seed_v5.yaml
cd artifacts/garden/seed_v5 && sha256sum -c SHA256SUMS.evaluation && cd ~/quantum-sentinel-alpha
python -m sentinel.train --config configs/train/qlora_ornith15_9b_v5.yaml --dry-run --check-tokens
python -m sentinel.train --config configs/train/qlora_ornith15_9b_v5.yaml
```

Development evaluation of the foundation, the first run's `step-000200` (reference) and every v5 checkpoint, then rule v1:

```bash
REV=489cb97981b8654bcfcf30ce1f94ed1b62e07b53; DEV=artifacts/garden/seed_v5/development; RUN=runs/train/qsa-qlora-ornith15-9b-v5
python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases $DEV/dev.eval_cases.jsonl --output runs/eval/dev-v5/runs/foundation.json
python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases $DEV/dev.eval_cases.jsonl --adapter runs/train/qsa-qlora-ornith15-9b-v0/checkpoints/step-000200/adapter --output runs/eval/dev-v5/runs/v0-step-000200.json
for ckpt in $RUN/checkpoints/step-*; do python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases $DEV/dev.eval_cases.jsonl --adapter $ckpt/adapter --output runs/eval/dev-v5/runs/v5-$(basename $ckpt).json; done
python -m sentinel.eval.select_checkpoint --devset $DEV/manifest.json runs/eval/dev-v5/runs/*.json --output runs/eval/dev-v5/selection.json
SEL=$(python -c "import json; print(json.load(open('runs/eval/dev-v5/selection.json'))['selected_adapter'])"); echo $SEL
python -m sentinel.garden analyze --config configs/garden/seed_v5.yaml --output runs/eval/dev-v5/analysis.json --runs runs/eval/dev-v5/runs/*.json --cases $DEV/dev.eval_cases.jsonl
```

Regression check of `$SEL` as in § 12 (new output files). Then, once, the final test, with the two sets reported separately:

```bash
for SET in final_test holdout_reserved; do
  python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases artifacts/garden/seed_v5/$SET.eval_cases.jsonl --output runs/eval/$SET-v5/foundation.json
  python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases artifacts/garden/seed_v5/$SET.eval_cases.jsonl --adapter $SEL --output runs/eval/$SET-v5/selected.json
  python -m sentinel.eval.compare runs/eval/$SET-v5/foundation.json runs/eval/$SET-v5/selected.json
done
```

Save `runs/eval/`, the run's `run.json` and the checkpoints' `provenance.json` (§ 11); never the case files.
