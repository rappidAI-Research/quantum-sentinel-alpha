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

## 15. Garden v6: fresh training, selection (rule v2), final test, backup (**GPU**)

Data: `docs/GARDEN_V6.md`. Settings: `docs/TRAINING_V6.md`. Instance: AWS EC2 `g6e.xlarge` (1 × NVIDIA L40S 48 GB, 4 vCPU, 32 GB RAM); at least 200 GB of gp3 storage for the foundation, the Git cache and 20 checkpoints (about 0.5 GB each). Train only after the owner decisions in `data/garden/seed_v6/cohort/OWNER_DECISION.md`; without them the authorized data is the Garden v5 set.

**1. Code, environment, foundation pin.**

```bash
cd ~/quantum-sentinel-alpha && git pull --ff-only && git log --oneline -1 && . .venv/bin/activate && pip install -e '.[train,quant,dev]'
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m sentinel.foundation
```

**2. Materialize the authorized records only and check their integrity.** `materialize` rebuilds every training-approved record from its pinned upstream commit (and, if approved, the CodeQL tests from the pinned revision) and refuses any content or supervision hash that differs from the committed manifest.

```bash
python -m sentinel.garden check --config configs/garden/seed_v6.yaml
python -m sentinel.garden materialize --config configs/garden/seed_v6.yaml
cd artifacts/garden/seed_v6 && sha256sum -c SHA256SUMS && cd ~/quantum-sentinel-alpha
python -m sentinel.garden materialize-eval --config configs/garden/seed_v6.yaml
cd artifacts/garden/seed_v6 && sha256sum -c SHA256SUMS.evaluation && cd ~/quantum-sentinel-alpha
python -m sentinel.train --config configs/train/qlora_ornith15_9b_v6.yaml --dry-run --check-tokens
```

The dry run must report the example and token counts of `docs/TRAINING_V6.md` (or the counts after the owner's decisions). Stop if it refuses a record.

**3. Baseline evaluation of the foundation** (before training, on the development set and the regression suite):

```bash
REV=489cb97981b8654bcfcf30ce1f94ed1b62e07b53; DEV=artifacts/garden/seed_v6/development; RUN=runs/train/qsa-qlora-ornith15-9b-v6
python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases $DEV/dev.eval_cases.jsonl --output runs/eval/dev-v6/runs/foundation.json
python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases $DEV/dev.eval_cases.jsonl --adapter runs/train/qsa-qlora-ornith15-9b-v0/checkpoints/step-000200/adapter --output runs/eval/dev-v6/runs/v0-step-000200.json
```

**4. Fresh training** (new LoRA on the pinned foundation; never `init_adapter`):

```bash
nohup python -m sentinel.train --config configs/train/qlora_ornith15_9b_v6.yaml > runs/train-v6.log 2>&1 &
tail -f runs/train-v6.log
```

**5. Checkpoint selection, rule v2** (fixed on 2026-11-25, before any v6 result; `docs/EVALUATION.md`): eligible only with invalid rate ≤ 0.02, recall ≥ 0.30 and false-positive rate ≤ 0.20; then the highest decision accuracy. If no checkpoint is eligible, nothing is selected: report that, do not relax the rule.

```bash
for ckpt in $RUN/checkpoints/step-*; do python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases $DEV/dev.eval_cases.jsonl --adapter $ckpt/adapter --output runs/eval/dev-v6/runs/v6-$(basename $ckpt).json; done
python -m sentinel.eval.select_checkpoint --rule 2 --devset $DEV/manifest.json runs/eval/dev-v6/runs/*.json --output runs/eval/dev-v6/selection.json
SEL=$(python -c "import json; print(json.load(open('runs/eval/dev-v6/selection.json'))['selected_adapter'])"); echo $SEL
python -m sentinel.garden analyze --config configs/garden/seed_v6.yaml --output runs/eval/dev-v6/analysis.json --runs runs/eval/dev-v6/runs/*.json --cases $DEV/dev.eval_cases.jsonl
```

Regression check of `$SEL` as in § 12. Then, once, the final test; `final_test` and the previous holdout are reported separately (the previous holdout is untouched only if § 12's holdout runs were never made):

```bash
for SET in final_test holdout_reserved; do
  python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases artifacts/garden/seed_v6/$SET.eval_cases.jsonl --output runs/eval/$SET-v6/foundation.json
  python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision $REV --cases artifacts/garden/seed_v6/$SET.eval_cases.jsonl --adapter $SEL --output runs/eval/$SET-v6/selected.json
  python -m sentinel.eval.compare runs/eval/$SET-v6/foundation.json runs/eval/$SET-v6/selected.json
done
```

Function-level results only: never combine them with a repository-level benchmark score (`docs/GARDEN_V6.md` § Evaluation).

**6. Backup before terminating the instance** (§ 11): adapters and metadata only, never case files, model weights of the foundation or Garden exports with code.

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf ~/qsa-v6-results-$STAMP.tgz runs/eval/dev-v6 runs/eval/final_test-v6 runs/eval/holdout_reserved-v6 \
  $RUN/run.json $RUN/checkpoints/*/provenance.json $RUN/checkpoints/*/checksums.json runs/train-v6.log
tar czf ~/qsa-v6-adapters-$STAMP.tgz $RUN/checkpoints/*/adapter $RUN/adapter 2>/dev/null
sha256sum ~/qsa-v6-*-$STAMP.tgz > ~/qsa-v6-$STAMP.SHA256SUMS
aws s3 cp ~/qsa-v6-results-$STAMP.tgz s3://<private-bucket>/qsa/v6/ && aws s3 cp ~/qsa-v6-adapters-$STAMP.tgz s3://<private-bucket>/qsa/v6/ && aws s3 cp ~/qsa-v6-$STAMP.SHA256SUMS s3://<private-bucket>/qsa/v6/
```

The adapters stay private (no publication of weights). Verify both archives against the checksum file on the destination before terminating the instance (§ 11); `runs/` is Git-ignored and nothing from it is committed.

## 16. Training Optimization v1: decision diagnostics, calibration, controlled runs (**GPU**)

Analysis, hypotheses and the fixed calibration and screening protocol: `docs/TRAINING_OPTIMIZATION_V1.md`. Same instance and storage as § 15. **Approved by the owner on 2026-11-26** (`docs/TRAINING_AUTHORIZATION.md`): E0–E4 and calibrated candidates. Order: E0 (no training) and its report; then E1–E4, one after another; then selection over all runs; the final test last and once. Never score `final_test` or the reserved holdout with `decision_scores`, and never choose a bias, checkpoint or setting on them.

**One command.** `scripts/to1_gpu.sh all` runs steps 0–8 below in order, under `nohup` or `tmux`. Its stages are `check`, `e0`, `train`, `evaluate <experiment>`, `select` and `final`.
- A rerun skips finished outputs and resumes an interrupted training run from its last checkpoint.
- The final test runs once (`runs/eval/to1-final.done`).
- `DRY_RUN=1 scripts/to1_gpu.sh all` prints every command without running it.

The step-by-step commands follow.

```bash
cd ~/quantum-sentinel-alpha && . .venv/bin/activate && DRY_RUN=1 scripts/to1_gpu.sh all 2>&1 | less
nohup scripts/to1_gpu.sh all > runs/to1.log 2>&1 &
tail -f runs/to1.log
```

**0. Variables, code, the V6 run's identity.**

```bash
cd ~/quantum-sentinel-alpha && git pull --ff-only && git log --oneline -1 && . .venv/bin/activate && pip install -e '.[train,quant,dev]'
REV=489cb97981b8654bcfcf30ce1f94ed1b62e07b53; M=ornith-ai/Ornith-1.5-9B; DEV=artifacts/garden/seed_v6/development
CAL=artifacts/garden/seed_v6/validation.jsonl; RUN=runs/train/qsa-qlora-ornith15-9b-v6; V0=runs/train/qsa-qlora-ornith15-9b-v0; TO1=runs/eval/to1
python -c "import json; r=json.load(open('$RUN/run.json')); print(r['data']['sha256'], r['config']['sha256'])"
```

The two hashes must be `51ab4c9d…` (data) and `f9a0edf6…` (config). Otherwise the order reconstruction below describes other data: stop and report.

**1. E0, CPU part: order, schedule, tokens and every logged step of the V6 run.**

```bash
TOK=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$M', revision='$REV', allow_patterns=['*.json', '*.txt', '*.jinja']))")
python -m sentinel.train.diagnostics --config configs/train/qlora_ornith15_9b_v6.yaml --tokenizer-dir $TOK --log runs/train-v6.log --window 126-175 --output $TO1/e0/diagnostics.json
```

The output reports clipped steps and the loss and gradient norm of steps 126–175. `run.json` alone keeps only the last 50 steps.

**2. E0, decision scores (GPU, one forward pass per case, several adapters per base load).** All V6 checkpoints and the first run's `step-000200` on the development set and the calibration set (the training validation split), then the foundation, then NF4 for four checkpoints:

```bash
ADAPTERS="$(for c in $RUN/checkpoints/step-*; do printf -- '--adapter %s/adapter ' $c; done) --adapter $V0/checkpoints/step-000200/adapter"
python -m sentinel.eval.decision_scores score --model $M --revision $REV $ADAPTERS --cases $DEV/dev.eval_cases.jsonl --output $TO1/e0/dev-bf16
python -m sentinel.eval.decision_scores score --model $M --revision $REV $ADAPTERS --records $CAL --output $TO1/e0/cal-bf16
python -m sentinel.eval.decision_scores score --model $M --revision $REV --cases $DEV/dev.eval_cases.jsonl --output $TO1/e0/dev-bf16/foundation.json
python -m sentinel.eval.decision_scores score --model $M --revision $REV --quantization bnb-4bit-dq \
  $(for s in 125 150 175 200; do printf -- '--adapter %s/checkpoints/step-%06d/adapter ' $RUN $s; done) --cases $DEV/dev.eval_cases.jsonl --output $TO1/e0/dev-nf4
```

Each line reports `generation-equivalent`: the share of cases whose teacher-forced decision is the one greedy generation makes. It should be close to 1.

**3. E0, analysis (CPU).** AUC, mean finding score, pair outcomes, flips between consecutive checkpoints, per language and vulnerability family:

```bash
python -m sentinel.eval.decision_scores analyze $TO1/e0/dev-bf16/*.json --cases $DEV/dev.eval_cases.jsonl --output $TO1/e0/analysis-bf16.json
python -m sentinel.eval.decision_scores analyze $TO1/e0/dev-nf4/*.json --cases $DEV/dev.eval_cases.jsonl --output $TO1/e0/analysis-nf4.json
python -m sentinel.eval.decision_scores analyze runs/eval/dev-v6/runs/*.json --cases $DEV/dev.eval_cases.jsonl --output $TO1/e0/analysis-generated.json
```

Read H1/H2 from `analysis-bf16.json`:
- the AUC of steps 150, 175 and 200;
- the 150 → 175 flips, their direction and how many pairs flip together.

Read H3 by comparing the same steps in `analysis-nf4.json`.

**4. E0, calibration on the validation split, screening, calibrated generation runs.**

```bash
mkdir -p $TO1/e0/calibration $TO1/e0/runs
for f in $TO1/e0/cal-bf16/*.json; do
  python -m sentinel.eval.decision_scores calibrate $f --max-fpr 0.15 --output $TO1/e0/calibration/$(basename $f) > /dev/null || echo "no bias within FPR 0.15: $f"
done
python -m sentinel.eval.decision_scores screen $TO1/e0/dev-bf16/*.json --calibrations $TO1/e0/calibration/*.json --output $TO1/e0/screen.json > $TO1/e0/shortlist.tsv
cat $TO1/e0/shortlist.tsv
while IFS=$'\t' read -r mode adapter cal label; do
  [ "$mode" = calibrated ] || continue  # the greedy V6 runs exist already (§ 15)
  python -m sentinel.eval.run_model --model $M --revision $REV --cases $DEV/dev.eval_cases.jsonl --adapter $adapter --calibration $cal \
    --output $TO1/e0/runs/${label//\//__}-calibrated.json < /dev/null
done < $TO1/e0/shortlist.tsv
python -m sentinel.eval.select_checkpoint --rule 2 --devset $DEV/manifest.json runs/eval/dev-v6/runs/*.json $TO1/e0/runs/*.json --output $TO1/e0/selection.json
```

`select_checkpoint` accepts a calibrated run only with its calibration provenance, and rule v2 is unchanged. **Write the E0 report** using the decision rules in § 7 of the analysis document, then continue with step 5. Even if a calibrated V6 checkpoint is eligible, do not run the final test yet.

**5. E1–E4 training** (approved; one run at a time; fresh LoRA, never `init_adapter`). Dry runs first, then all four configs in sequence:

```bash
for E in baseline_dense decision_w10 detection_only lr5e-5; do python -m sentinel.train --config configs/train/qlora_ornith15_9b_v6_to1_$E.yaml --dry-run --check-tokens || break; done
nohup sh -c 'for E in baseline_dense decision_w10 detection_only lr5e-5; do python -m sentinel.train --config configs/train/qlora_ornith15_9b_v6_to1_$E.yaml > runs/train-to1-$E.log 2>&1 || exit 1; done' > /dev/null 2>&1 &
tail -f runs/train-to1-baseline_dense.log
```

Every evaluation line now carries `validation.decision` (accuracy, recall, FPR, finding rate, mean margins) and `validation.loss_by_task`. The four runs take about 11–12 hours in total.

**6. Each training run: scores, calibration, screening, generation runs** (repeat with `E=baseline_dense`, `decision_w10`, `detection_only`, `lr5e-5`).

```bash
E=decision_w10; R=runs/train/qsa-qlora-ornith15-9b-v6-to1-$E; O=$TO1/$E; mkdir -p $O/calibration $O/runs
A=$(for c in $R/checkpoints/step-*; do printf -- '--adapter %s/adapter ' $c; done)
python -m sentinel.eval.decision_scores score --model $M --revision $REV $A --cases $DEV/dev.eval_cases.jsonl --output $O/dev-bf16
python -m sentinel.eval.decision_scores score --model $M --revision $REV $A --records $CAL --output $O/cal-bf16
for f in $O/cal-bf16/*.json; do python -m sentinel.eval.decision_scores calibrate $f --output $O/calibration/$(basename $f) > /dev/null || echo "no bias: $f"; done
python -m sentinel.eval.decision_scores analyze $O/dev-bf16/*.json --cases $DEV/dev.eval_cases.jsonl --output $O/analysis.json
python -m sentinel.eval.decision_scores screen $O/dev-bf16/*.json --calibrations $O/calibration/*.json --output $O/screen.json > $O/shortlist.tsv
while IFS=$'\t' read -r mode adapter cal label; do
  python -m sentinel.eval.run_model --model $M --revision $REV --cases $DEV/dev.eval_cases.jsonl --adapter $adapter \
    $([ "$mode" = calibrated ] && echo --calibration $cal) --output $O/runs/${label//\//__}-$mode.json < /dev/null
done < $O/shortlist.tsv
python -m sentinel.train.diagnostics --config configs/train/qlora_ornith15_9b_v6_to1_$E.yaml --log runs/train-to1-$E.log --window 150-200 --output $O/diagnostics.json
```

For `baseline_dense` (E1), the swing's position relative to the epoch boundary (after step 173) is in `analysis.json` (flips between consecutive checkpoints) and in the training log's `validation.decision`. E2–E4 are compared with E1, not with V6.

**7. Selection over all runs** (rule v2, unchanged):

```bash
python -m sentinel.eval.select_checkpoint --rule 2 --devset $DEV/manifest.json runs/eval/dev-v6/runs/*.json $TO1/*/runs/*.json --output $TO1/selection.json
```

**8. Final test, once, after selection over all runs (approved on 2026-11-26).** The selected checkpoint is run with its calibration file, if it has one:

```bash
SEL=$(python -c "import json; print(json.load(open('$TO1/selection.json'))['selected_adapter'])")
SELCAL=$(python -c "import json; print(json.load(open('$TO1/selection.json'))['selected_calibration'] or '')")
for SET in final_test holdout_reserved; do
  python -m sentinel.eval.run_model --model $M --revision $REV --cases artifacts/garden/seed_v6/$SET.eval_cases.jsonl --output runs/eval/$SET-to1/foundation.json
  python -m sentinel.eval.run_model --model $M --revision $REV --cases artifacts/garden/seed_v6/$SET.eval_cases.jsonl --adapter $SEL ${SELCAL:+--calibration $SELCAL} --output runs/eval/$SET-to1/selected.json
  python -m sentinel.eval.compare runs/eval/$SET-to1/foundation.json runs/eval/$SET-to1/selected.json
done
```

**9. Backup before terminating the instance** (§ 11): scores, analyses, calibrations, run artifacts and adapters. Never case files, foundation weights or Garden exports with code.

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar czf ~/qsa-to1-results-$STAMP.tgz $TO1 runs/eval/final_test-to1 runs/eval/holdout_reserved-to1 runs/train-to1-*.log \
  runs/train/qsa-qlora-ornith15-9b-v6-to1-*/run.json runs/train/qsa-qlora-ornith15-9b-v6-to1-*/checkpoints/*/provenance.json runs/train/qsa-qlora-ornith15-9b-v6-to1-*/checkpoints/*/checksums.json 2>/dev/null
tar czf ~/qsa-to1-adapters-$STAMP.tgz runs/train/qsa-qlora-ornith15-9b-v6-to1-*/checkpoints/*/adapter runs/train/qsa-qlora-ornith15-9b-v6-to1-*/adapter 2>/dev/null
sha256sum ~/qsa-to1-*-$STAMP.tgz > ~/qsa-to1-$STAMP.SHA256SUMS
aws s3 cp ~/qsa-to1-results-$STAMP.tgz s3://<private-bucket>/qsa/to1/ && aws s3 cp ~/qsa-to1-adapters-$STAMP.tgz s3://<private-bucket>/qsa/to1/ && aws s3 cp ~/qsa-to1-$STAMP.SHA256SUMS s3://<private-bucket>/qsa/to1/
```

Score files contain no code: they hold case ids, labels and log-probabilities. The adapters stay private, and nothing from `runs/` is committed.

## 17. Training Optimization v2: F0, pairwise ranking, lower learning rate (**GPU**, not approved yet)

Analysis, pre-registered metrics and decision points: `docs/TRAINING_OPTIMIZATION_V2.md`. Same instance as §§ 15–16. **Nothing in this section is approved yet.** The owner approves each stage, and each stage ends at a decision point (D0–D3) that is reported before the next one starts. The final test runs once over v1 and v2 together, and never from a stage other than `final`.

`scripts/to2_gpu.sh` runs one stage per call and reuses § 16's helpers. Finished outputs are skipped on a rerun, and an interrupted training run resumes. `DRY_RUN=1` prints the commands.

**0. Code and a dry run.** The v1 results must be in place: `runs/eval/to1/{e0,decision_w10,detection_only}`, E2's checkpoints under `runs/train/qsa-qlora-ornith15-9b-v6-to1-decision_w10/checkpoints`, and the training logs `runs/train-to1-*.log`.

```bash
cd ~/quantum-sentinel-alpha && git pull --ff-only && . .venv/bin/activate && pip install -e '.[train,quant,dev]'
DRY_RUN=1 scripts/to2_gpu.sh f0 2>&1 | less
```

**1. F0, no training** (≈ 1.5–2.5 GPU h). It runs the following, then the D0 report:
- the supervision audit;
- the v1 score files and run artifacts re-analysed: AUC, pairs, flips, per source, true-positive quality;
- gradient diagnostics over all steps of E2 and E3;
- whether E2 step 200 greedy and calibrated decide every case alike;
- E2 on the NF4 training precision;
- exact averages of E2 checkpoints, calibrated, screened, generated;
- selection.

```bash
nohup scripts/to2_gpu.sh f0 > runs/to2-f0.log 2>&1 &
cat runs/eval/to2/f0/e2-step200-greedy-vs-calibrated.txt
python -c "import json; print(json.load(open('runs/eval/to2/f0/decision_w10-dev-scores.json'))['plateau'])"
python -c "import json; print(json.load(open('runs/eval/to2/f0/avg/analysis.json'))['plateau'])"
```

If E2 step 200 has no greedy run artifact (it was not shortlisted in v1), generate it first:

```bash
python -m sentinel.eval.run_model --model ornith-ai/Ornith-1.5-9B --revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53 --cases artifacts/garden/seed_v6/development/dev.eval_cases.jsonl --adapter runs/train/qsa-qlora-ornith15-9b-v6-to1-decision_w10/checkpoints/step-000200/adapter --output runs/eval/to1/decision_w10/runs/qsa-qlora-ornith15-9b-v6-to1-decision_w10__step-000200-greedy.json
```

**D0 (report before anything else):**
- E2's plateau AUC (checkpoints from step 180 on);
- NF4 against bf16;
- the averages against the checkpoints;
- per-source results;
- true-positive quality;
- whether the E2 step-200 predictions are identical.

The rules are in `docs/TRAINING_OPTIMIZATION_V2.md` § 6.

**2. F1, pairwise ranking** (after approval; ≈ 3 h training + 2–3 h evaluation):

```bash
nohup sh -c 'scripts/to2_gpu.sh train pair_rank && scripts/to2_gpu.sh evaluate pair_rank' > runs/to2-f1.log 2>&1 &
tail -f runs/train-to2-pair_rank.log
```

Every evaluation line carries `validation.pair_ranking` (the share of the 58 validation pairs ranked correctly, and the mean gap) next to `validation.decision`. **D1:** compare plateau AUC, pair discrimination and the selected rule-v2 candidate with E2. The `plateau` blocks are in `runs/eval/to2/pair_rank/analysis.json` and `runs/eval/to2/f0/decision_w10-dev-scores.json`. `python -m sentinel.eval.compare <E2 run> <F1 run>` gives the paired McNemar p of two selected candidates.

**3. F2, learning rate 5e-5** (after approval; can run before or after F1):

```bash
nohup sh -c 'scripts/to2_gpu.sh train lr5e-5 && scripts/to2_gpu.sh evaluate lr5e-5' > runs/to2-f2.log 2>&1 &
```

**D2:** the swing (changed decisions between consecutive checkpoints) and plateau AUC against E2.

**4. F3, only if D1 and D2 both adopt:**

```bash
nohup sh -c 'scripts/to2_gpu.sh train pair_rank_lr5e-5 && scripts/to2_gpu.sh evaluate pair_rank_lr5e-5' > runs/to2-f3.log 2>&1 &
```

**5. Selection and the final test**, once, after the plan has ended:

```bash
scripts/to2_gpu.sh select
scripts/to2_gpu.sh final
```

The selection covers V6, v1, v2 and the checkpoint averages under rule v2. `final` refuses to run if `runs/eval/to1-final.done` or `runs/eval/to2-final.done` exists.

**6. Backup** before terminating the instance, as in § 16 step 9. It covers `runs/eval/to2`, `runs/train-to2-*.log`, the v2 runs' `run.json`, provenance and checksums, the adapters, and E2's `averaged/` directory. Adapters stay private and nothing from `runs/` is committed.
