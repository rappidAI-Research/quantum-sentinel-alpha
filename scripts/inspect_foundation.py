#!/usr/bin/env python3
"""Inspect the pinned foundation from its small files only (no weights) and write the LoRA module list.

    python scripts/inspect_foundation.py --files-dir DIR            # print the module list
    python scripts/inspect_foundation.py --files-dir DIR --write    # update configs/train/<list>
    python scripts/inspect_foundation.py --files-dir DIR --check    # exit 1 if the committed list differs

DIR holds the foundation's config.json and model.safetensors.index.json at the pinned revision,
e.g. downloaded with:

    rev=$(python -c "import yaml;print(yaml.safe_load(open('configs/model/foundation.yaml'))['revision'])")
    for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \\
             chat_template.jinja vocab.json merges.txt; do
      curl -sfLo "$DIR/$f" "https://huggingface.co/ornith-ai/Ornith-1.5-9B/resolve/$rev/$f"; done

The text model is instantiated on the meta device exactly as the trainer builds it
(AutoModelForCausalLM, text-only); every linear module is listed with its shape, and every module
must exist under the checkpoint's own weight names (model.language_model.* in the index).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.train.config import DEFAULT_TRAIN_CONFIG, LINEAR_MODULES, load_train_config, matching_modules  # noqa: E402


def module_list(files_dir: Path, model_id: str, revision: str) -> tuple[str, list[str], list[str]]:
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(files_dir)
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config)
    rows = [
        (name, module.in_features, module.out_features)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    stored = json.loads((files_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    mapped = {re.sub(r"^model\.language_model\.", "model.", key) for key in stored}
    missing = [name for name, _, _ in rows if f"{name}.weight" not in mapped]
    config_sha = hashlib.sha256((files_dir / "config.json").read_bytes()).hexdigest()
    header = (
        f"# Linear modules of {model_id}@{revision} as built by AutoModelForCausalLM\n"
        f"# ({type(model).__name__}, transformers {transformers.__version__}), instantiated on the meta device from config.json\n"
        f"# (sha256 {config_sha}). No weights were downloaded. Regenerate: scripts/inspect_foundation.py --write\n"
        "# Format: <module name> <in_features> <out_features>\n"
    )
    body = [f"{name} {inp} {out}" for name, inp, out in rows]
    return header, body, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    train = load_train_config(args.config)
    header, body, missing = module_list(args.files_dir, train.model_id, train.revision)
    if missing:
        print(f"modules without a checkpoint weight: {missing[:5]}", file=sys.stderr)
        return 1
    targets = matching_modules(train.lora.target_modules, [line.split()[0] for line in body])
    rank = train.lora.rank
    params = sum(rank * (int(i) + int(o)) for name, i, o in (line.split() for line in body) if name in set(targets))
    print(f"{len(body)} linear modules; {len(targets)} LoRA targets (expected {train.lora.expected_target_count}); "
          f"{params:,} trainable LoRA parameters at rank {rank}", file=sys.stderr)
    if len(targets) != train.lora.expected_target_count:
        return 1
    target = LINEAR_MODULES
    text = header + "\n".join(body) + "\n"
    if args.write:
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target.relative_to(PROJECT_ROOT)}", file=sys.stderr)
    elif args.check:
        committed = [line for line in target.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
        if committed != body:
            print(f"{target.relative_to(PROJECT_ROOT)} differs from the model built from {args.files_dir}", file=sys.stderr)
            return 1
        print(f"{target.relative_to(PROJECT_ROOT)} matches", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
