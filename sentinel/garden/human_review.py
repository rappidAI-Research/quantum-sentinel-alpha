"""Evidence packages for an independent human review of Sentinel Garden labels.

The earlier manual sample (audit/manual_review.yaml) was read by the implementing AI engineer. This
module prepares a separate, reproducible review that a person who did not build the pipeline can do
without the tooling, and keeps their decisions apart from every automated or AI judgement:

- `draw_sample`: a deterministic, stratified sample of the population that would be exported for
  training (accepted train/validation groups), per language: pairs round-robin over CWE families and
  abstention items, with its own salt (independent of the AI sample).
- `build_package`: one self-contained Markdown package per item - advisory text and links, fix and
  vulnerable commit links, the vulnerable excerpt with the labelled span marked, the function-level
  upstream patch, the fixed excerpt (hard negative), the abstention excerpt and withheld callee, the
  expected labels and a checklist. The package is blind: no AI verdict and no automated audit signal
  is included (those are written next to it in `signals.json`, to be consulted only after deciding).
  Packages contain upstream code, so they are written to the Git-ignored work directory; only the
  index (ids, links, package sha256) is committed.
- `load_human_review`: validates `audit/human_review.yaml`, which only a human reviewer writes. A
  decision is bound to the sha256 of the package it was made on; a decision on a changed package is
  stale and is not counted. Nothing in this repository fills that file.
- Refreshing the packages never invalidates a review: an item that is no longer drawn (for example
  because the reviewer's own exclusion removed its pair from the population) stays in the index as
  `retired` with its last package hash. Its decision keeps its exclusions but is not counted in the
  error rates, which describe the current sample only.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from sentinel.garden.audit import family_of

SAMPLE_ID = "sentinel-garden-human-review-v1"
SALT = "sentinel-garden-human-review-v1"
PAIRS_PER_LANGUAGE = 12
ABSTENTIONS_PER_LANGUAGE = 4
EXPORT_SPLITS = {"train", "validation"}

QUESTIONS: dict[str, dict[str, Any]] = {
    "advisory_matches_code": {
        "items": {"pair"},
        "allowed": {"yes", "partly", "no"},
        "ask": "Does the advisory describe a weakness that is visible in the vulnerable excerpt?",
    },
    "fix_addresses_weakness": {
        "items": {"pair"},
        "allowed": {"yes", "partly", "no"},
        "ask": "Does the upstream patch address that weakness (rather than an unrelated change)?",
    },
    "label_correct": {
        "items": {"pair"},
        "allowed": {"yes", "no", "unclear"},
        "ask": "Is the vulnerable excerpt really vulnerable as labelled (decision `finding`)?",
    },
    "cwe": {
        "items": {"pair"},
        "allowed": {"correct", "defensible", "ambiguous", "wrong"},
        "ask": "Is the labelled CWE the most specific fitting weakness? If not, name the better CWE in `suggested_cwe`.",
    },
    "localization": {
        "items": {"pair"},
        "allowed": {"precise", "acceptable", "imprecise", "wrong"},
        "ask": "Does the marked span contain the vulnerable code (or the point where the missing check belongs), and little else?",
    },
    "hard_negative_safe": {
        "items": {"pair"},
        "allowed": {"yes", "no", "unclear"},
        "ask": "Is the fixed excerpt safe with respect to this weakness (expected decision `no_finding`)?",
    },
    "pair_consistent": {
        "items": {"pair"},
        "allowed": {"yes", "no"},
        "ask": "Are the two excerpts the same function before and after the fix, with no unrelated difference that could teach the wrong signal?",
    },
    "abstention_valid": {
        "items": {"abstention"},
        "allowed": {"yes", "no", "unclear"},
        "ask": "Without the withheld callee, is the right answer really `abstain` (safety cannot be decided from the excerpt and its context)?",
    },
    "leakage_or_duplicate": {
        "items": {"pair", "abstention"},
        "allowed": {"none_seen", "suspected"},
        "ask": "Do you recognise this code as copied from another project or as part of a public benchmark? (Optional search.)",
    },
}
REVIEWER_KEYS = {"name", "affiliation", "independent_of_implementation", "conflicts"}


def _order(key: str) -> str:
    return hashlib.sha256(f"{SALT}\x00{key}".encode()).hexdigest()


def draw_sample(records: list[dict[str, Any]], repositories: set[str] | None = None) -> list[dict[str, Any]]:
    """Deterministic stratified sample over accepted train/validation groups.

    `repositories` restricts the population to a proposed training cohort, so the reviewer checks
    exactly what would be trained if that cohort were approved."""
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if repositories is not None and record["provenance"]["repository_key"] not in repositories:
            continue
        if record["quality"]["status"] == "accepted" and record["split"] in EXPORT_SPLITS:
            groups[record["group_id"]][record["kind"]] = record
    by_language: dict[str, list[str]] = defaultdict(list)
    for group_id, group in groups.items():
        anchor = next(iter(group.values()))
        by_language[anchor["input"]["language"]].append(group_id)
    sample: list[dict[str, Any]] = []
    for language in sorted(by_language):
        families: dict[str, list[str]] = defaultdict(list)
        complete = [g for g in by_language[language] if {"vulnerable", "post_fix"} <= set(groups[g])]
        for group_id in sorted(complete, key=_order):
            families[family_of(groups[group_id]["vulnerable"]["target"]["cwe"])].append(group_id)
        chosen: list[str] = []
        while len(chosen) < PAIRS_PER_LANGUAGE and any(families.values()):
            for family in sorted(families):
                if families[family] and len(chosen) < PAIRS_PER_LANGUAGE:
                    chosen.append(families[family].pop(0))
        abstentions = sorted((g for g in by_language[language] if "caller_abstention" in groups[g]), key=_order)[:ABSTENTIONS_PER_LANGUAGE]
        for item, group_ids in (("pair", chosen), ("abstention", abstentions)):
            for group_id in group_ids:
                group = groups[group_id]
                kinds = ("vulnerable", "post_fix") if item == "pair" else ("caller_abstention",)
                anchor = group.get("vulnerable") or group["caller_abstention"]
                sample.append(
                    {
                        "item_id": f"{item}-{group_id}",
                        "group_id": group_id,
                        "item": item,
                        "language": language,
                        "split": anchor["split"],
                        "cwe_family": family_of(anchor["target"]["cwe"]) if item == "pair" else None,
                        "advisory_id": anchor["provenance"]["advisory_id"],
                        "record_ids": [group[k]["record_id"] for k in kinds],
                    }
                )
    return sample


def _numbered(code: str, mark: tuple[int, int] | None = None) -> str:
    lines = code.splitlines()
    width = len(str(len(lines)))
    out = []
    for number, line in enumerate(lines, start=1):
        flag = ">>" if mark and mark[0] <= number <= mark[1] else "  "
        out.append(f"{flag} {number:>{width}} | {line}")
    return "\n".join(out)


def _fence(text: str, language: str = "") -> str:
    ticks = "````" if "```" in text else "```"
    return f"{ticks}{language}\n{text}\n{ticks}"


def _commit_url(repository: str, sha: str) -> str:
    return f"{repository}/commit/{sha}"


def _blob_url(repository: str, sha: str, path: str, lines: list[list[int]] | None) -> str:
    anchor = f"#L{lines[0][0]}-L{lines[0][1]}" if lines else ""
    return f"{repository}/blob/{sha}/{path}{anchor}"


def build_package(
    item: dict[str, Any],
    records: dict[str, dict[str, Any]],
    advisory: dict[str, Any] | None,
    decisions_path: str = "data/garden/seed_v1/audit/human_review.yaml",
) -> str:
    """The blind evidence package for one sample item (deterministic Markdown)."""
    first = records[item["record_ids"][0]]
    provenance = first["provenance"]
    repository = provenance["repository"]
    advisory_id = provenance["advisory_id"]
    lines = [
        f"# Review item `{item['item_id']}`",
        "",
        f"Sample `{SAMPLE_ID}` - {item['item']} - {item['language']} - split `{item['split']}`.",
        "Decide every question from the evidence below before opening `signals.json`. Do not run the code.",
        "",
        "## Advisory",
        "",
        f"- ID: [{advisory_id}](https://github.com/advisories/{advisory_id}) ([OSV](https://osv.dev/vulnerability/{advisory_id}))",
        f"- Aliases: {', '.join(provenance['aliases']) or 'none'}",
        f"- Ecosystem / packages: {provenance['ecosystem']} / {', '.join(provenance['packages'])}",
        f"- Published: {provenance.get('published') or 'unknown'}",
        "",
    ]
    if advisory:
        text = f"{advisory.get('summary', '')}\n\n{advisory.get('details', '')}".strip()
        lines += ["Advisory text (OSV record, untrusted data):", "", _fence(text[:6000]), ""]
    else:
        lines += ["Advisory text: not in the local OSV cache; read it at the links above.", ""]
    lines += [
        "## Upstream fix",
        "",
        f"- Repository: {repository}",
        f"- Fix commit: [{provenance['fixed_revision']}]({_commit_url(repository, provenance['fixed_revision'])}), committed {provenance['fix_committed_at']}",
        f"- Vulnerable revision (first parent): [{provenance['vulnerable_revision']}]({_commit_url(repository, provenance['vulnerable_revision'])})",
        f"- File: `{provenance['file_path']}`",
        "",
    ]
    if item["item"] == "pair":
        vulnerable, fixed = records[item["record_ids"][0]], records[item["record_ids"][1]]
        target = vulnerable["target"]
        span = (target["start_line"], target["end_line"])
        lines += [
            "## Expected labels",
            "",
            f"- Vulnerable excerpt (`{vulnerable['record_id']}`): `finding`, {target['cwe']}"
            + (f" (alternates {', '.join(target['alternate_cwes'])})" if target["alternate_cwes"] else "")
            + f", lines {span[0]}-{span[1]} ({vulnerable['localization'].get('span_kind', 'span')}, derived from the upstream diff).",
            f"- Fixed excerpt (`{fixed['record_id']}`): `no_finding` for this weakness (hard negative).",
            f"- Training evidence text: {target['evidence']!r} (source: {target['evidence_source']}).",
            "",
            f"## Vulnerable excerpt (`>>` marks the labelled span; [file at the vulnerable revision]({_blob_url(repository, provenance['vulnerable_revision'], provenance['file_path'], provenance['excerpt_file_lines'])}))",
            "",
            f"Function `{provenance['excerpt_function']}`. Context given to the model:",
            "",
            _fence(vulnerable["input"]["context"]),
            "",
            _fence(_numbered(vulnerable["input"]["code"], span)),
            "",
            "## Upstream patch, restricted to the function",
            "",
            _fence(target["patch"] or "(none)", "diff"),
            "",
            f"## Fixed excerpt ([file at the fix commit]({_blob_url(repository, provenance['fixed_revision'], provenance['file_path'], fixed['provenance']['excerpt_file_lines'])}))",
            "",
            _fence(_numbered(fixed["input"]["code"])),
            "",
        ]
    else:
        abstention = records[item["record_ids"][0]]
        lines += [
            "## Expected label",
            "",
            f"- Caller excerpt (`{abstention['record_id']}`): `abstain` - its safety depends on `{abstention['localization'].get('withheld_callee')}`,"
            " the vulnerable function, which is withheld.",
            "",
            "## Caller excerpt and the context given to the model",
            "",
            _fence(abstention["input"]["context"]),
            "",
            _fence(_numbered(abstention["input"]["code"], (abstention["localization"].get("call_line") or 0,) * 2)),
            "",
        ]
    lines += ["## Checklist", ""]
    for key, question in QUESTIONS.items():
        if item["item"] in question["items"]:
            lines.append(f"- `{key}` ({' / '.join(sorted(question['allowed']))}): {question['ask']}")
    lines += [
        "",
        f"Record the answers in `{decisions_path}` under this `item_id`, together with the",
        "`package_sha256` from `audit/human_review_sample.jsonl`. List records that should not be trained on under",
        "`exclude_records` with a reason; labels are never edited.",
        "",
    ]
    return "\n".join(lines)


def automated_signals(item: dict[str, Any], automated: dict[str, dict[str, Any]], manual: dict[str, Any]) -> dict[str, Any]:
    """Everything the tooling or the AI review said, kept out of the blind package."""
    ai = [i for i in manual.get("items") or [] if i["group_id"] == item["group_id"] and i.get("item", "pair") == item["item"]]
    return {"automated_audit": automated.get(item["group_id"]), "ai_review": ai[0] if ai else None, "ai_reviewer": manual.get("reviewer")}


def _decisions_path(committed_audit_dir: Path) -> str:
    """Where the reviewer records decisions, as a repository-relative path (dataset-specific)."""
    root = Path(__file__).resolve().parents[2]
    target = committed_audit_dir.resolve() / "human_review.yaml"
    # outside the repository (tests, scratch runs) the text must not depend on the location
    return target.relative_to(root).as_posix() if target.is_relative_to(root) else "audit/human_review.yaml"


def write_packages(
    records: list[dict[str, Any]],
    osv_dir: Path,
    automated: list[dict[str, Any]],
    manual: dict[str, Any],
    package_dir: Path,
    committed_audit_dir: Path,
    repositories: set[str] | None = None,
) -> list[dict[str, Any]]:
    by_id = {r["record_id"]: r for r in records}
    by_group = {g["group_id"]: g for g in automated}
    sample = draw_sample(records, repositories)
    index_path = committed_audit_dir / "human_review_sample.jsonl"
    previous = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()] if index_path.is_file() else []
    package_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for item in sample:
        provenance = by_id[item["record_ids"][0]]["provenance"]
        path = osv_dir / provenance["ecosystem"] / f"{provenance['advisory_id']}.json"
        advisory = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        text = build_package(item, by_id, advisory, _decisions_path(committed_audit_dir))
        item_dir = package_dir / item["item_id"]
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "package.md").write_text(text, encoding="utf-8")
        (item_dir / "signals.json").write_text(json.dumps(automated_signals(item, by_group, manual), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        index.append({**item, "package_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    current = {item["item_id"] for item in index}
    index += [{**old, "retired": True} for old in previous if old["item_id"] not in current]
    committed_audit_dir.mkdir(parents=True, exist_ok=True)
    (committed_audit_dir / "human_review_sample.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in index), encoding="utf-8")
    (committed_audit_dir / "human_review.TEMPLATE.yaml").write_text(template(index), encoding="utf-8")
    return index


def template(index: list[dict[str, Any]]) -> str:
    """An unfilled decisions file. It is never loaded; a reviewer copies it to human_review.yaml."""
    document = {
        "sample_id": SAMPLE_ID,
        "reviewer": {"name": None, "affiliation": None, "independent_of_implementation": None, "conflicts": None},
        "reviewed_at": None,
        "items": [
            {
                "item_id": item["item_id"],
                "package_sha256": item["package_sha256"],
                **{key: None for key, q in QUESTIONS.items() if item["item"] in q["items"]},
                **({"suggested_cwe": None} if item["item"] == "pair" else {}),
                "exclude_records": [],
                "notes": None,
            }
            for item in index
            if not item.get("retired")
        ],
    }
    return (
        "# Template for an independent human review of audit/human_review_sample.jsonl (docs/SENTINEL_GARDEN.md).\n"
        "# Copy to human_review.yaml and fill in by hand. Never generated or filled by tooling.\n"
        + yaml.safe_dump(document, sort_keys=False, width=200)
    )


def load_human_review(path: Path, index: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a reviewer's decisions; returns them with stale decisions separated."""
    if not path.is_file():
        return {"reviewer": None, "items": [], "stale": [], "retired": [], "independent": False}
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if document.get("sample_id") != SAMPLE_ID:
        raise ValueError(f"{path}: sample_id must be {SAMPLE_ID}")
    reviewer = document.get("reviewer") or {}
    if set(reviewer) != REVIEWER_KEYS or not reviewer.get("name") or not document.get("reviewed_at"):
        raise ValueError(f"{path}: reviewer needs {sorted(REVIEWER_KEYS)} and reviewed_at must be set")
    if not isinstance(reviewer["independent_of_implementation"], bool):
        raise ValueError(f"{path}: reviewer.independent_of_implementation must be true or false")
    by_id = {i["item_id"]: i for i in index}
    items, stale, retired = [], [], []
    for entry in document.get("items") or []:
        sample_item = by_id.get(entry.get("item_id"))
        if sample_item is None:
            raise ValueError(f"{path}: unknown item {entry.get('item_id')!r}")
        answered = {k: entry.get(k) for k, q in QUESTIONS.items() if sample_item["item"] in q["items"]}
        if any(v is None for v in answered.values()):
            continue  # not reviewed yet
        for key, value in answered.items():
            if value not in QUESTIONS[key]["allowed"]:
                raise ValueError(f"{entry['item_id']}: {key}={value!r} not in {sorted(QUESTIONS[key]['allowed'])}")
        for record_id in entry.get("exclude_records") or []:
            if record_id not in sample_item["record_ids"]:
                raise ValueError(f"{entry['item_id']}: exclude_records may only name this item's records")
        if entry.get("package_sha256") != sample_item["package_sha256"]:
            stale.append(entry)
        elif sample_item.get("retired"):
            retired.append(entry)  # still applies its exclusions; not part of the current sample's rates
        else:
            items.append(entry)
    return {
        "reviewer": reviewer,
        "items": items,
        "stale": [e["item_id"] for e in stale],
        "retired": retired,
        "independent": reviewer["independent_of_implementation"],
    }


def human_exclusions(review: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"record_id": record_id, "reason": "human_review:" + (entry.get("notes") or "excluded").split("\n")[0][:80]}
        for entry in review["items"] + review.get("retired", [])
        for record_id in entry.get("exclude_records") or []
    ]


def summarize_human(review: dict[str, Any], index: list[dict[str, Any]]) -> dict[str, Any]:
    """Error counts from a human reviewer only; never mixed with automated or AI judgements."""
    from sentinel.garden.audit import wilson_interval

    errors = {
        "label_wrong": ("label_correct", {"no"}),
        "cwe_wrong_or_ambiguous": ("cwe", {"wrong", "ambiguous"}),
        "localization_imprecise_or_wrong": ("localization", {"imprecise", "wrong"}),
        "hard_negative_not_safe": ("hard_negative_safe", {"no"}),
        "pair_inconsistent": ("pair_consistent", {"no"}),
        "abstention_invalid": ("abstention_valid", {"no"}),
    }
    rates = {}
    for name, (key, bad) in errors.items():
        judged = [i for i in review["items"] if key in i and i[key] in QUESTIONS[key]["allowed"]]
        count = sum(1 for i in judged if i[key] in bad)
        low, high = wilson_interval(count, len(judged)) if judged else (None, None)
        rates[name] = {"errors": count, "judged": len(judged), "wilson95": [low, high]}
    return {
        "sample_id": SAMPLE_ID,
        "sample_items": sum(1 for i in index if not i.get("retired")),
        "retired_decisions": [e["item_id"] for e in review.get("retired", [])],
        "reviewed_items": len(review["items"]),
        "stale_decisions": review["stale"],
        "reviewer": review["reviewer"],
        "independent_of_implementation": review["independent"],
        "status": "not_performed" if not review["items"] else ("independent" if review["independent"] else "not_independent"),
        "rates": rates,
    }
