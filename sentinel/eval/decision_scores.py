"""Decision scores: how strongly a model prefers finding / no_finding / abstain, case by case.

    # GPU: one teacher-forced forward pass per case (no generation); several --adapter share one base model
    python -m sentinel.eval.decision_scores score --model ornith-ai/Ornith-1.5-9B --revision <sha> \\
        [--adapter <checkpoint>/adapter ...] [--quantization none|bnb-4bit] \\
        (--cases <dev.eval_cases.jsonl> | --records <workdir>/validation.jsonl) --output <scores.json | directory>
    # CPU: compare checkpoints (scores and/or run_model artifacts), pairs, flips, operating points
    python -m sentinel.eval.decision_scores analyze <scores-or-runs...> --cases <dev.eval_cases.jsonl> --output <analysis.json>
    # CPU: choose a finding-logit bias on a CALIBRATION set (never the set the choice is evaluated on)
    python -m sentinel.eval.decision_scores calibrate <calibration-scores.json> --max-fpr 0.15 --output <calibration.json>
    # GPU: generation with that bias, provenance recorded in the artifact
    python -m sentinel.eval.run_model ... --adapter <checkpoint>/adapter --calibration <calibration.json>

Every detection reply starts with `{"decision": "`, and the next token decides the class: `finding`,
`no` (no_finding) or `ab` (abstain). Scoring appends that prefix to the evaluation prompt (rendered exactly
as `HFBackend.generate` renders it) and reads the log-probabilities of the three decision tokens. When
the model would also generate the prefix greedily and its greedy next token is a decision token, the
score's greedy decision IS the decision greedy generation makes; `prefix_greedy` and `greedy_in_decisions`
record whether that holds for each case. The finding score `s = logp(finding) - logp(no)` ranks cases
independently of any threshold: its ROC-AUC separates "the model cannot tell vulnerable from fixed code"
from "the model's decision threshold moved".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from sentinel.train.data import DECISION_PREFIX

DECISIONS = ("finding", "no_finding", "abstain")
SCORES_VERSION = "1"


def decision_token_ids(tokenizer: Any) -> dict[str, int]:
    """The token that follows the reply prefix for each decision (it must differ between decisions)."""
    prefix = list(tokenizer(DECISION_PREFIX, add_special_tokens=False)["input_ids"])
    ids: dict[str, int] = {}
    for decision in DECISIONS:
        full = list(tokenizer(f'{DECISION_PREFIX}{decision}"', add_special_tokens=False)["input_ids"])
        if full[: len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError(f"the tokenizer does not split {DECISION_PREFIX!r} off the {decision!r} reply")
        ids[decision] = full[len(prefix)]
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"decisions share their first token: {ids}")
    return ids


def finding_score(row: dict[str, Any]) -> float:
    return float(row["logprob"]["finding"] - row["logprob"]["no_finding"])


# --------------------------------------------------------------------------------------------
# Scoring (the model part is a callable, so everything else runs and is tested on CPU)
# --------------------------------------------------------------------------------------------


def score_rows(cases: list[dict[str, Any]], scorer: Callable[[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per case. `scorer(case)` returns {"logprob": {decision: float}, "greedy_token": decision|None,
    "prefix_greedy": bool}."""
    rows = []
    for case in cases:
        scored = scorer(case)
        greedy = scored.get("greedy_token")
        rows.append(
            {
                "case_id": case["case_id"],
                "expected": case["expected_decision"],
                "language": case.get("language"),
                "pair_id": case.get("pair_id"),
                "cwe": (case.get("ground_truth") or {}).get("cwe"),
                "logprob": {d: round(float(scored["logprob"][d]), 6) for d in DECISIONS},
                "greedy_in_decisions": greedy in DECISIONS,
                "prefix_greedy": bool(scored.get("prefix_greedy", True)),
                # the generation's decision whenever the two flags hold; otherwise the best of the three decision tokens
                "decision": greedy if greedy in DECISIONS else max(DECISIONS, key=lambda d: scored["logprob"][d]),
            }
        )
    return rows


class HFDecisionScorer:
    """Teacher-forced decision log-probabilities from a loaded `HFBackend` (same prompt rendering)."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.tokenizer = backend._tokenizer
        self.ids = decision_token_ids(self.tokenizer)
        self.prefix = list(self.tokenizer(DECISION_PREFIX, add_special_tokens=False)["input_ids"])

    def __call__(self, case: dict[str, Any]) -> dict[str, Any]:
        import torch

        from sentinel.eval.prompt import PROMPT_VERSION, build_messages

        template_kwargs = {"add_generation_prompt": True, "enable_thinking": self.backend.generation.enable_thinking}
        rendered = self.tokenizer.apply_chat_template(build_messages(case, PROMPT_VERSION), tokenize=False, **template_kwargs)
        prompt = list(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])
        model = self.backend._model
        ids = torch.tensor([prompt + self.prefix], device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=ids).logits[0].float()
        logprobs = torch.log_softmax(logits[-1], dim=-1)
        # would greedy decoding itself have produced the prefix? (logits at i predict token i + 1)
        start = len(prompt) - 1
        predicted = logits[start : start + len(self.prefix)].argmax(dim=-1).tolist()
        greedy_id = int(logprobs.argmax())
        greedy = next((d for d, i in self.ids.items() if i == greedy_id), None)
        return {
            "logprob": {d: float(logprobs[i]) for d, i in self.ids.items()},
            "greedy_token": greedy,
            "prefix_greedy": predicted == self.prefix,
        }


# --------------------------------------------------------------------------------------------
# Metrics (same definitions as sentinel.eval.metrics: recall over finding cases, FPR over no_finding cases)
# --------------------------------------------------------------------------------------------


def decision_metrics(rows: list[dict[str, Any]], key: str = "decision") -> dict[str, Any]:
    positives = [r for r in rows if r["expected"] == "finding"]
    negatives = [r for r in rows if r["expected"] == "no_finding"]
    abstains = [r for r in rows if r["expected"] == "abstain"]
    tp = sum(1 for r in positives if r[key] == "finding")
    fp = sum(1 for r in negatives if r[key] == "finding")
    correct = sum(1 for r in rows if r[key] == r["expected"])
    return {
        "cases": len(rows),
        "recall": round(tp / len(positives), 4) if positives else None,
        "false_positive_rate": round(fp / len(negatives), 4) if negatives else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "decision_accuracy": round(correct / len(rows), 4) if rows else None,
        "abstention_recall": round(sum(1 for r in abstains if r[key] == "abstain") / len(abstains), 4) if abstains else None,
        "predicted": dict(sorted(Counter(r[key] for r in rows).items())),
        "confusion": {f"{e}->{p}": n for (e, p), n in sorted(Counter((r["expected"], r[key]) for r in rows).items())},
    }


def roc_auc(positive_scores: list[float], negative_scores: list[float]) -> float | None:
    """Probability that a random finding case scores above a random no_finding case (ties count half)."""
    if not positive_scores or not negative_scores:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive_scores for n in negative_scores)
    return round(wins / (len(positive_scores) * len(negative_scores)), 4)


def with_bias(row: dict[str, Any], bias: float) -> str:
    """The decision when `bias` is added to the finding log-probability (abstain and no_finding unchanged)."""
    scores = {**row["logprob"], "finding": row["logprob"]["finding"] + bias}
    return max(DECISIONS, key=lambda d: scores[d])


def operating_points(rows: list[dict[str, Any]], biases: list[float] | None = None) -> list[dict[str, Any]]:
    """Recall / FPR / precision / accuracy for a range of finding biases (bias 0 = argmax of the three)."""
    if biases is None:
        candidates = sorted({round(max(r["logprob"]["no_finding"], r["logprob"]["abstain"]) - r["logprob"]["finding"], 4) for r in rows})
        biases = sorted({0.0, *(c + 1e-4 for c in candidates)})
    points = []
    for bias in biases:
        decided = [{**r, "biased": with_bias(r, bias)} for r in rows]
        m = decision_metrics(decided, key="biased")
        points.append({"bias": round(bias, 4), **{k: m[k] for k in ("recall", "false_positive_rate", "precision", "decision_accuracy", "abstention_recall")}})
    return points


def best_bias(rows: list[dict[str, Any]], max_fpr: float) -> dict[str, Any] | None:
    """The bias with the highest recall whose FPR stays <= max_fpr (ties: higher accuracy, then smaller |bias|)."""
    feasible = [p for p in operating_points(rows) if p["false_positive_rate"] is not None and p["false_positive_rate"] <= max_fpr]
    if not feasible:
        return None
    return max(feasible, key=lambda p: (p["recall"] or 0.0, p["decision_accuracy"] or 0.0, -abs(p["bias"])))


def cross_fitted_bias(rows: list[dict[str, Any]], max_fpr: float, folds: int = 2) -> dict[str, Any]:
    """Honest estimate of a calibrated bias: choose it on all folds but one, measure on the held-out fold.
    Folds are by pair (a vulnerable case and its fixed twin never sit in different folds)."""
    def fold_of(row: dict[str, Any]) -> int:
        key = row.get("pair_id") or row["case_id"]
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % folds

    held_out: list[dict[str, Any]] = []
    chosen = []
    for k in range(folds):
        fit = [r for r in rows if fold_of(r) != k]
        test = [r for r in rows if fold_of(r) == k]
        point = best_bias(fit, max_fpr)
        if point is None:
            continue
        chosen.append(point["bias"])
        held_out += [{**r, "biased": with_bias(r, point["bias"])} for r in test]
    metrics = decision_metrics(held_out, key="biased") if held_out else None
    return {"folds": folds, "biases": chosen, "held_out": metrics}


def pair_outcomes(rows: list[dict[str, Any]], key: str = "decision") -> dict[str, Any]:
    """Vulnerable/fixed pairs: discriminated (finding / not finding), both flagged, both missed, inverted."""
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        if r.get("pair_id") and r["expected"] in {"finding", "no_finding"}:
            by_pair.setdefault(r["pair_id"], {})[r["expected"]] = r
    outcome: Counter[str] = Counter()
    margins = []
    for pair in by_pair.values():
        if set(pair) != {"finding", "no_finding"}:
            continue
        vulnerable, fixed = pair["finding"][key] == "finding", pair["no_finding"][key] == "finding"
        outcome[{(True, False): "discriminated", (True, True): "both_flagged", (False, False): "both_missed", (False, True): "inverted"}[(vulnerable, fixed)]] += 1
        if "logprob" in pair["finding"] and "logprob" in pair["no_finding"]:
            margins.append(finding_score(pair["finding"]) - finding_score(pair["no_finding"]))
    result: dict[str, Any] = {"pairs": sum(outcome.values()), **{k: outcome.get(k, 0) for k in ("discriminated", "both_flagged", "both_missed", "inverted")}}
    if margins:
        margins.sort()
        result["score_margin_vulnerable_minus_fixed"] = {
            "median": round(margins[len(margins) // 2], 4),
            "share_positive": round(sum(1 for m in margins if m > 0) / len(margins), 4),
        }
    return result


def breakdown(rows: list[dict[str, Any]], field: str, key: str = "decision") -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get(field)), []).append(r)
    return {g: {k: v for k, v in decision_metrics(rs, key).items() if k in ("cases", "recall", "false_positive_rate", "decision_accuracy")} for g, rs in sorted(groups.items())}


def summarize(rows: list[dict[str, Any]], max_fpr: float = 0.20) -> dict[str, Any]:
    """Everything one checkpoint's rows say (decisions only, or decisions and scores)."""
    from sentinel.garden.audit import family_of

    family_rows = [{**r, "family": family_of(r.get("cwe"))} for r in rows]
    result: dict[str, Any] = {
        "greedy": decision_metrics(rows),
        "pairs": pair_outcomes(rows),
        "by_language": breakdown(rows, "language"),
        "findings_by_family": breakdown([r for r in family_rows if r["expected"] == "finding"], "family"),
    }
    if rows and "logprob" in rows[0]:
        positives = [finding_score(r) for r in rows if r["expected"] == "finding"]
        negatives = [finding_score(r) for r in rows if r["expected"] == "no_finding"]
        every = [finding_score(r) for r in rows]
        result["scores"] = {
            "auc_finding_vs_no_finding": roc_auc(positives, negatives),
            "mean_finding_score": round(sum(every) / len(every), 4),
            "mean_finding_score_on_findings": round(sum(positives) / len(positives), 4) if positives else None,
            "mean_finding_score_on_negatives": round(sum(negatives) / len(negatives), 4) if negatives else None,
            "greedy_matches_generation_conditions": round(sum(1 for r in rows if r["prefix_greedy"] and r["greedy_in_decisions"]) / len(rows), 4),
            f"best_bias_at_fpr_{max_fpr}": best_bias(rows, max_fpr),
        }
    return result


def flips(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    """Cases whose decision changed between two checkpoints, by expected class and direction; a flip of
    both members of a pair in the same direction points to a moved threshold, not to new evidence."""
    a = {r["case_id"]: r for r in before}
    changed = [(a[r["case_id"]], r) for r in after if r["case_id"] in a and a[r["case_id"]]["decision"] != r["decision"]]
    by_pair: dict[str, list[tuple[str, str]]] = {}
    for old, new in changed:
        if new.get("pair_id"):
            by_pair.setdefault(new["pair_id"], []).append((old["decision"], new["decision"]))
    return {
        "changed": len(changed),
        "by_direction": dict(sorted(Counter(f"{o['expected']}: {o['decision']}->{n['decision']}" for o, n in changed).items())),
        "pairs_flipping_together": sum(1 for moves in by_pair.values() if len(moves) == 2 and moves[0] == moves[1]),
    }


# --------------------------------------------------------------------------------------------
# Inputs: score files and run_model artifacts
# --------------------------------------------------------------------------------------------

_STEP = re.compile(r"step-(\d{6})")


def label_of(document: dict[str, Any], path: Path) -> str:
    adapter = ((document.get("model") or {}).get("adapter") or {}).get("path") or ""
    match = _STEP.search(adapter) or _STEP.search(path.name)
    if match:
        run = re.search(r"runs/train/([^/]+)/", adapter)
        return f"{run.group(1) + '/' if run else ''}step-{match.group(1)}"
    return "foundation" if not adapter else Path(adapter).name


def rows_from_document(document: dict[str, Any], cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Score files carry rows; run_model artifacts carry predictions (decisions only)."""
    if "rows" in document:
        return list(document["rows"])
    rows = []
    for prediction in document["predictions"]:
        case = cases.get(prediction["case_id"], {})
        rows.append(
            {
                "case_id": prediction["case_id"],
                "expected": prediction["expected_decision"],
                "language": case.get("language"),
                "pair_id": case.get("pair_id"),
                "cwe": (case.get("ground_truth") or {}).get("cwe"),
                "decision": prediction["prediction"]["decision"],
            }
        )
    return rows


def analyze(paths: list[Path], cases: list[dict[str, Any]], max_fpr: float = 0.20) -> dict[str, Any]:
    by_id = {c["case_id"]: c for c in cases}
    series = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        rows = rows_from_document(document, by_id)
        unknown = {r["case_id"] for r in rows} - set(by_id)
        if unknown or len(rows) != len(by_id):
            raise ValueError(f"{path} was not made on these cases ({len(rows)} rows, {len(unknown)} unknown)")
        series.append({"label": label_of(document, path), "kind": "scores" if "rows" in document else "run", "path": str(path), "rows": rows})
    series.sort(key=lambda s: (s["label"] != "foundation", s["label"]))
    result = {"cases": len(cases), "checkpoints": [], "flips_between_consecutive": []}
    for item in series:
        result["checkpoints"].append({"label": item["label"], "kind": item["kind"], "path": item["path"], **summarize(item["rows"], max_fpr)})
    ordered = [s for s in series if s["label"] != "foundation"]
    for before, after in zip(ordered, ordered[1:]):
        result["flips_between_consecutive"].append({"from": before["label"], "to": after["label"], **flips(before["rows"], after["rows"])})
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# evaluation roles a calibration set may never come from: what selection or the final test is measured on
PROTECTED_ROLES = {"development", "final_test", "previous_holdout", "external_benchmark", "temporal_quarantine"}


def calibration_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Eval cases from Garden records of the training validation split (the calibration set). Records of
    the development set, a final test, a holdout, an external benchmark or quarantine are refused."""
    from sentinel.garden import render

    refused = sorted({r["record_id"] for r in records if r.get("split") != "validation" or r.get("evaluation_role") in PROTECTED_ROLES})
    if refused:
        raise ValueError(f"{len(refused)} records are not training-validation records (first: {refused[:3]}); calibrate on validation.jsonl only")
    return [{**render.to_eval_case(r), "source_kind": "calibration"} for r in records]


def refuse_protected_cases(cases: list[dict[str, Any]]) -> None:
    """Final-test and holdout exports (`source_kind: private_holdout`) are never scored for optimization."""
    protected = [c["case_id"] for c in cases if c.get("source_kind") == "private_holdout"]
    if protected:
        raise ValueError(f"{len(protected)} final-test/holdout cases (first: {protected[:3]}); decision scores are for the development and calibration sets only")


CALIBRATION_VERSION = "1"


def calibration_document(document: dict[str, Any], max_fpr: float) -> dict[str, Any]:
    """The finding bias chosen on a calibration score file, with what it was chosen on.

    Only a score file of the training validation split (`cases.kind == "calibration"`) is accepted, so the
    bias is never chosen on the development set or a test set. `run_model --calibration` applies it to the
    same model (id, revision, precision, adapter weights) and records this provenance in the run artifact."""
    if (document.get("cases") or {}).get("kind") != "calibration":
        raise ValueError("calibrate needs a score file of the calibration set (score --records <workdir>/validation.jsonl), not the development set")
    rows = document["rows"]
    chosen = best_bias(rows, max_fpr)
    model = document["model"]
    return {
        "calibration_version": CALIBRATION_VERSION,
        "max_fpr": max_fpr,
        "bias": chosen["bias"] if chosen else None,
        "chosen": chosen,
        "cross_fitted_estimate": cross_fitted_bias(rows, max_fpr),
        "model": {
            "id": model["id"],
            "revision": model["revision"],
            "quantization": model["quantization"],
            "dtype": model["dtype"],
            "adapter_weights_sha256": (model.get("adapter") or {}).get("adapter_model_sha256"),
        },
        "cases": document["cases"],
    }


def check_calibration(document: dict[str, Any], *, model_id: str, revision: str | None, quantization: str, dtype: str, adapter_weights_sha256: str | None) -> float:
    """The calibrated bias, if `document` was made for exactly this model; otherwise ValueError."""
    if document.get("calibration_version") != CALIBRATION_VERSION:
        raise ValueError(f"unknown calibration version {document.get('calibration_version')!r}")
    if (document.get("cases") or {}).get("kind") != "calibration":
        raise ValueError("the bias was not chosen on the calibration set")
    if document.get("bias") is None:
        raise ValueError("the calibration found no bias within its false-positive ceiling")
    expected = {"id": model_id, "revision": revision, "quantization": quantization, "dtype": dtype, "adapter_weights_sha256": adapter_weights_sha256}
    different = sorted(k for k, v in expected.items() if document["model"].get(k) != v)
    if different:
        raise ValueError(f"calibration was made for another model setup (differs in {', '.join(different)})")
    return float(document["bias"])


def score_outputs(output: Path, adapters: list[Path | None]) -> list[Path]:
    """Where `score` writes each model's file: `output` itself for one model, `<output>/<run>__step-NNNNNN.json`
    for several adapters. Checked before the model is loaded, so two adapters never overwrite each other."""
    if len(adapters) == 1:
        return [output]
    paths = [output / (label_of({"model": {"adapter": {"path": str(a)}}}, Path(str(a))).replace("/", "__") + ".json") for a in adapters]
    duplicates = sorted({str(p) for p in paths if paths.count(p) > 1})
    if duplicates:
        raise ValueError(f"adapters map to the same output file: {duplicates}")
    return paths


def adapter_digest(adapter: Path) -> str | None:
    weights = adapter / "adapter_model.safetensors"
    return hashlib.sha256(weights.read_bytes()).hexdigest() if weights.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="teacher-forced decision scores of one model (GPU)")
    score.add_argument("--model", required=True)
    score.add_argument("--revision", required=True)
    score.add_argument("--adapter", type=Path, action="append", default=[], help="repeat to score several checkpoints on one loaded base model")
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases", type=Path, help="eval cases (the development set)")
    source.add_argument("--records", type=Path, help="Garden records of the training validation split (calibration set)")
    score.add_argument("--quantization", default="none", choices=("none", "bnb-4bit"), help="bnb-4bit = the NF4 base the adapter was trained on")
    score.add_argument("--dtype", default="bfloat16")
    score.add_argument("--device", default="auto")
    score.add_argument("--output", type=Path, required=True, help="a file for one model, a directory for several adapters")
    an = sub.add_parser("analyze", help="compare score files and/or run_model artifacts (CPU)")
    an.add_argument("inputs", nargs="+", type=Path)
    an.add_argument("--cases", type=Path, required=True)
    an.add_argument("--max-fpr", type=float, default=0.20)
    an.add_argument("--output", type=Path)
    cal = sub.add_parser("calibrate", help="choose a finding bias on a calibration score file (CPU)")
    cal.add_argument("scores", type=Path)
    cal.add_argument("--max-fpr", type=float, default=0.15, help="kept below the selection ceiling (0.20) as a margin")
    cal.add_argument("--output", type=Path, help="calibration file for run_model --calibration")
    args = parser.parse_args(argv)

    if args.command == "score":
        from sentinel.eval.backends import GenerationSettings, HFBackend, RuntimeSettings

        source = args.cases or args.records
        cases = _read_jsonl(args.cases) if args.cases else calibration_cases(_read_jsonl(args.records))
        refuse_protected_cases(cases)
        adapters: list[Path | None] = list(args.adapter) or [None]
        outputs = score_outputs(args.output, adapters)
        backend = HFBackend(args.model, args.revision, RuntimeSettings(device=args.device, dtype=args.dtype, quantization=args.quantization), GenerationSettings(), adapters[0])
        backend.load()
        active = "default"
        for index, adapter in enumerate(adapters):
            if index:  # swap the LoRA weights on the loaded base model (PEFT), instead of reloading 9B parameters
                name = f"adapter{index}"
                backend._model.load_adapter(str(adapter), adapter_name=name)
                backend._model.set_adapter(name)
                backend._model.delete_adapter(active)
                active = name
            rows = score_rows(cases, HFDecisionScorer(backend))
            document = {
                "scores_version": SCORES_VERSION,
                "model": {
                    "id": args.model,
                    "revision": args.revision,
                    "adapter": {"path": str(adapter), "adapter_model_sha256": adapter_digest(adapter)} if adapter else None,
                    "quantization": args.quantization,
                    "dtype": args.dtype,
                },
                "backend": backend.describe() if index == 0 else {"loaded_with": "the first adapter's base model, adapter swapped in place"},
                "cases": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "count": len(cases), "kind": "development" if args.cases else "calibration"},
                "decision_token_ids": decision_token_ids(backend._tokenizer),
                "summary": summarize(rows),
                "rows": rows,
            }
            output = outputs[index]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            g, s = document["summary"]["greedy"], document["summary"]["scores"]
            print(f"{output}: recall {g['recall']}, FPR {g['false_positive_rate']}, AUC {s['auc_finding_vs_no_finding']}, generation-equivalent {s['greedy_matches_generation_conditions']}")
        return 0
    if args.command == "analyze":
        result = analyze(args.inputs, _read_jsonl(args.cases), args.max_fpr)
        for c in result["checkpoints"]:
            g, p = c["greedy"], c["pairs"]
            auc = (c.get("scores") or {}).get("auc_finding_vs_no_finding")
            print(f"{c['label']:<32} recall {g['recall']}  FPR {g['false_positive_rate']}  AUC {auc}  pairs disc/both+/both-/inv {p['discriminated']}/{p['both_flagged']}/{p['both_missed']}/{p['inverted']}")
        for f in result["flips_between_consecutive"]:
            print(f"{f['from']} -> {f['to']}: {f['changed']} changed, {f['pairs_flipping_together']} pairs together; {f['by_direction']}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    calibration = calibration_document(json.loads(args.scores.read_text(encoding="utf-8")), args.max_fpr)
    calibration["scores"] = {"path": str(args.scores), "sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest()}
    text = json.dumps(calibration, indent=1, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if calibration["bias"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
