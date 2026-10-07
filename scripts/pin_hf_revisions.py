#!/usr/bin/env python3
"""Pin each bake-off candidate to the current immutable commit SHA of its Hugging Face `main` branch.

    python scripts/pin_hf_revisions.py [--config configs/model/bakeoff_qwen35.yaml] [--dry-run]

Only candidates with `revision: null` are pinned; an existing pin is never moved. Needs network
access to huggingface.co (set HF_TOKEN if the repositories require it). Commit the result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel.eval.bakeoff import load_bakeoff  # noqa: E402
from sentinel.eval.runner import is_immutable_revision  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model" / "bakeoff_qwen35.yaml"


def fetch_main_sha(model_id: str) -> str:
    request = urllib.request.Request(f"https://huggingface.co/api/models/{model_id}/revision/main")
    if token := os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        sha = json.load(response).get("sha")
    if not is_immutable_revision(sha):
        raise ValueError(f"{model_id}: API returned no 40-character commit SHA: {sha!r}")
    return sha


def set_revisions(text: str, revisions: dict[str, str]) -> str:
    """Replace `revision: null` directly following each `model_id:` line, keeping comments and layout."""
    for model_id, sha in revisions.items():
        pattern = re.compile(rf"(model_id:\s*{re.escape(model_id)}\s*\n(\s*)revision:\s*)null\b")
        text, count = pattern.subn(rf"\g<1>{sha}", text)
        if count != 1:
            raise ValueError(f"expected exactly one unpinned revision for {model_id}, found {count}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_bakeoff(args.config)
    pending = {c.model_id: fetch_main_sha(c.model_id) for c in config.candidates.values() if c.revision is None}
    for model_id, sha in pending.items():
        print(f"{model_id}: {sha}")
    if not pending:
        print("all candidates already pinned")
        return 0
    updated = set_revisions(args.config.read_text(encoding="utf-8"), pending)
    if not args.dry_run:
        args.config.write_text(updated, encoding="utf-8")
        load_bakeoff(args.config)
        print(f"updated {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
