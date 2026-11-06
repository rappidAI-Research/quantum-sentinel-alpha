"""Supervised examples derived from Garden records, and the hashes that bind them to the manifest.

Integrity (manifest field `supervision`, written by `curate`, committed in Git):
- `input_sha256` / `target_sha256`: canonical JSON hashes of the full record input (language, file,
  context, code) and the full target (decision, CWE, span, evidence text, evidence source, patch);
- `detection_example_sha256`: hash of the rendered detection chat example (prompt + reply);
- `patch_example_sha256`: hash of the patch-generation chat example, or null when the record is not
  patch-eligible, with `patch_eligibility` stating why.
The training loader recomputes all of them from the Git-ignored exports and refuses any mismatch, so
an edited evidence text, patch, context or label is detected even when the code hash is unchanged.

Patch supervision (task `patch`, prompt `sentinel-patch-v1`): only a vulnerable record whose
post-fix sibling is accepted (it survived every quality check and the audit: not fixed again later,
not reported incomplete, not unverifiable, not manually excluded) and whose upstream patch applies
strictly and exactly turns the vulnerable excerpt into the accepted fixed excerpt.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sentinel.eval.patch import PATCH_PROMPT_VERSION, PatchError, build_patch_messages, check_coordinates, parse_patch
from sentinel.garden import render
from sentinel.garden.quality import apply_unified_patch

SUPERVISION_VERSION = "1"
PATCH_RENDER_VERSION = "patch-v1"
HASH_FIELDS = ("input_sha256", "target_sha256", "detection_example_sha256", "patch_example_sha256")


def canonical_sha256(value: Any) -> str:
    """sha256 of the canonical JSON form (sorted keys, no whitespace, UTF-8)."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pairs_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record.get("pair_id"):
            pairs.setdefault(record["pair_id"], {})[record["kind"]] = record
    return pairs


def patch_eligibility(record: dict[str, Any], fixed: dict[str, Any] | None) -> str:
    """'eligible' or the first reason the record cannot be patch ground truth."""
    if record["kind"] != "vulnerable" or record["target"]["decision"] != "finding":
        return "not_a_vulnerable_record"
    if record["quality"]["status"] != "accepted":
        return "vulnerable_record_rejected"
    if not record["target"].get("patch"):
        return "no_upstream_patch"
    if fixed is None:
        return "no_fixed_sibling"
    if fixed["quality"]["status"] != "accepted":
        # the audit excluded the post-fix record: fixed again later, reported incomplete, unverifiable
        return "fixed_sibling_rejected"
    if fixed["provenance"]["repository_key"] != record["provenance"]["repository_key"] or fixed["split"] != record["split"]:
        return "fixed_sibling_inconsistent"
    if apply_unified_patch(record["input"]["code"], record["target"]["patch"]) != fixed["input"]["code"]:
        return "patch_does_not_reproduce_fix"
    try:
        _, hunks = parse_patch(record["target"]["patch"], record["input"]["file"])
        check_coordinates(hunks, len(record["input"]["code"].splitlines()))
    except PatchError:
        return "patch_not_in_output_contract"
    return "eligible"


def patch_case(record: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    """A patch task built from a verified vulnerable/fixed pair (eval case and training input)."""
    target, provenance = record["target"], record["provenance"]
    return {
        "case_id": record["record_id"],
        "task": "patch",
        "language": record["input"]["language"],
        "file": record["input"]["file"],
        "context": record["input"]["context"],
        "code": record["input"]["code"],
        "finding": {"cwe": target["cwe"], "start_line": target["start_line"], "end_line": target["end_line"]},
        "advisory_summary": target["evidence"],
        "reference_patch": parse_patch(target["patch"], record["input"]["file"])[0],
        "reference_fixed_code": fixed["input"]["code"],
        "provenance": {
            "advisory_id": provenance["advisory_id"],
            "repository": provenance["repository"],
            "vulnerable_revision": provenance["vulnerable_revision"],
            "fixed_revision": provenance["fixed_revision"],
            "fixed_record_id": fixed["record_id"],
            "split": record["split"],
        },
    }


def patch_chat_example(case: dict[str, Any]) -> dict[str, Any]:
    messages = build_patch_messages(case)
    messages.append({"role": "assistant", "content": case["reference_patch"].rstrip("\n")})
    return {
        "record_id": case["case_id"],
        "task": "patch",
        "prompt_version": PATCH_PROMPT_VERSION,
        "render_version": PATCH_RENDER_VERSION,
        "messages": messages,
    }


def supervision(record: dict[str, Any], pairs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    fixed = (pairs.get(record.get("pair_id") or "") or {}).get("post_fix")
    eligibility = patch_eligibility(record, fixed) if record["kind"] == "vulnerable" else "not_a_vulnerable_record"
    patch_example = patch_chat_example(patch_case(record, fixed)) if eligibility == "eligible" else None
    return {
        "version": SUPERVISION_VERSION,
        "input_sha256": canonical_sha256(record["input"]),
        "target_sha256": canonical_sha256(record["target"]),
        "detection_example_sha256": canonical_sha256(render.to_chat_example(record)),
        "patch_eligibility": eligibility,
        "patch_example_sha256": canonical_sha256(patch_example) if patch_example else None,
    }


def patch_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Patch chat examples for the eligible vulnerable records among `records` (sorted by id)."""
    pairs = pairs_by_id(records)
    out = []
    for record in sorted(records, key=lambda r: r["record_id"]):
        if record["kind"] != "vulnerable":
            continue
        fixed = pairs.get(record.get("pair_id") or "", {}).get("post_fix")
        if patch_eligibility(record, fixed) == "eligible":
            out.append(patch_chat_example(patch_case(record, fixed)))
    return out


def patch_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = pairs_by_id(records)
    out = []
    for record in sorted(records, key=lambda r: r["record_id"]):
        if record["kind"] != "vulnerable":
            continue
        fixed = pairs.get(record.get("pair_id") or "", {}).get("post_fix")
        if patch_eligibility(record, fixed) == "eligible":
            out.append(patch_case(record, fixed))
    return out


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def manifest_supervision_problem(row: dict[str, Any]) -> str | None:
    """Structural check of a committed manifest row's `supervision` block (offline, no code needed)."""
    block = row.get("supervision")
    if not isinstance(block, dict) or block.get("version") != SUPERVISION_VERSION:
        return f"manifest row has no supervision block of version {SUPERVISION_VERSION}"
    for key in ("input_sha256", "target_sha256", "detection_example_sha256"):
        if not isinstance(block.get(key), str) or not _HEX64.match(block[key]):
            return f"supervision.{key} is not a sha256"
    eligible = block.get("patch_eligibility") == "eligible"
    digest = block.get("patch_example_sha256")
    if eligible != (digest is not None) or (digest is not None and not _HEX64.match(digest)):
        return "supervision.patch_example_sha256 must be a sha256 exactly when the record is patch-eligible"
    if eligible and row.get("kind") != "vulnerable":
        return "only vulnerable records can be patch-eligible"
    return None
