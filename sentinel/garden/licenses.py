"""Heuristic license *detection* for upstream repositories.

Detection is evidence for a human reviewer, never an approval. Every detected license is recorded
with `status: unreviewed`; only an entry in `data/garden/approvals.yaml` can change that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LICENSE_FILE = re.compile(r"^(licen[cs]e|copying|unlicense)([-._][A-Za-z0-9.-]+)?$", re.IGNORECASE)

PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD", "Unlicense", "Zlib", "PSF-2.0"}
WEAK_COPYLEFT = {"MPL-2.0", "LGPL-2.1", "LGPL-3.0", "EPL-2.0"}
STRONG_COPYLEFT = {"GPL-2.0", "GPL-3.0", "AGPL-3.0"}


@dataclass(frozen=True)
class DetectedLicense:
    spdx: str  # SPDX identifier, "unknown" or "none"
    category: str  # permissive | weak_copyleft | strong_copyleft | unknown | none


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_license_text(text: str) -> str:
    t = _norm(text)
    if "gnu affero general public license" in t:
        return "AGPL-3.0"
    if "gnu lesser general public license" in t or "gnu library general public license" in t:
        return "LGPL-3.0" if "version 3" in t else "LGPL-2.1"
    if "gnu general public license" in t:
        return "GPL-3.0" if "version 3" in t else "GPL-2.0"
    if "mozilla public license" in t and "2.0" in t:
        return "MPL-2.0"
    if "eclipse public license" in t and "2.0" in t:
        return "EPL-2.0"
    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "this is free and unencumbered software released into the public domain" in t:
        return "Unlicense"
    if "permission is hereby granted, free of charge, to any person obtaining a copy" in t:
        return "MIT"
    if "permission to use, copy, modify, and/or distribute this software for any purpose with or without fee" in t or (
        "permission to use, copy, modify, and distribute this software for any purpose with or without fee" in t
    ):
        return "0BSD" if "copyright notice and this permission notice appear in all copies" not in t else "ISC"
    if "redistribution and use in source and binary forms" in t:
        if "neither the name" in t or "the names of its contributors may not be used" in t:
            return "BSD-3-Clause"
        return "BSD-2-Clause"
    if "this software is provided 'as-is', without any express or implied warranty" in t and "altered source versions" in t:
        return "Zlib"
    if "python software foundation" in t and "license agreement" in t:
        return "PSF-2.0"
    return "unknown"


def category_of(spdx: str) -> str:
    if spdx == "none":
        return "none"
    if spdx in PERMISSIVE:
        return "permissive"
    if spdx in WEAK_COPYLEFT:
        return "weak_copyleft"
    if spdx in STRONG_COPYLEFT:
        return "strong_copyleft"
    return "unknown"


def detect(files: dict[str, str]) -> DetectedLicense:
    """Detect a license from root license files (name -> text). Conflicting files give 'unknown'."""
    if not files:
        return DetectedLicense("none", "none")
    found = {detect_license_text(text) for _, text in sorted(files.items())}
    if len(found) > 1:
        found.discard("unknown")  # e.g. LICENSE (MIT) next to an unrecognised NOTICE-style file
    if len(found) == 1:
        spdx = found.pop()
        return DetectedLicense(spdx, category_of(spdx))
    # several different licenses (dual licensing or per-directory terms): a reviewer must decide which
    # applies; only an all-permissive combination keeps the permissive category
    categories = {category_of(spdx) for spdx in found}
    return DetectedLicense("multiple:" + "|".join(sorted(found)), "permissive" if categories == {"permissive"} else "unknown")
