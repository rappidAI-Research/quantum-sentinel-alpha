"""Garden v2 extraction refinements: each recovers a localizable fix and still rejects anything ambiguous.

The refinements are off by default, so Garden v1 rebuilds unchanged; `configs/garden/seed_v2.yaml`
enables them.
"""

from __future__ import annotations

import dataclasses

import pytest

from sentinel.garden.build import Rejected, build_records
from sentinel.garden.quality import apply_unified_patch
from tests.garden_fixtures import PY_NEW, PY_OLD, SETTINGS, FakeRepo, candidate

V2 = dataclasses.replace(SETTINGS, member_additions=True, version_hunks_as_noise=True, tooling_files_excluded=True)

PY_CLASS_OLD = '''import os


class Store:
    def __init__(self, root):
        self.root = root

    def read(self, name):
        path = os.path.join(self.root, name)
        with open(path) as handle:
            return handle.read()
'''

PY_CLASS_NEW = '''import os


class Store:
    def __init__(self, root):
        self.root = root

    def _inside_root(self, path):
        root = os.path.realpath(self.root)
        return os.path.commonpath([root, os.path.realpath(path)]) == root

    def read(self, name):
        path = os.path.join(self.root, name)
        if not self._inside_root(path):
            raise ValueError("path escapes the store")
        with open(path) as handle:
            return handle.read()
'''

JS_CLASS_OLD = '''class Renderer {
  constructor(target) {
    this.target = target;
  }

  render(html) {
    this.target.innerHTML = html;
  }
}

module.exports = Renderer;
'''

JS_CLASS_NEW = '''class Renderer {
  constructor(target) {
    this.target = target;
  }

  escape(text) {
    return String(text).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  }

  render(html) {
    this.target.textContent = this.escape(html);
  }
}

module.exports = Renderer;
'''


def _build(repo, settings, cand=None):
    return build_records(cand or candidate(), repo, settings, "sentinel-garden-test", "ghsa-osv-fix-commits", ["osv"])


def test_defaults_reproduce_garden_v1() -> None:
    assert not (SETTINGS.member_additions or SETTINGS.version_hunks_as_noise or SETTINGS.tooling_files_excluded)


@pytest.mark.parametrize(
    "path, old, new, ecosystem, helper",
    [
        ("store/files.py", PY_CLASS_OLD, PY_CLASS_NEW, "PyPI", "def _inside_root"),
        ("lib/renderer.js", JS_CLASS_OLD, JS_CLASS_NEW, "npm", "escape(text)"),
    ],
)
def test_new_helper_method_in_class_body_belongs_to_the_fix(path, old, new, ecosystem, helper) -> None:
    repo = FakeRepo({path: old}, {path: new})
    with pytest.raises(Rejected, match="change_outside_function"):
        _build(repo, SETTINGS, candidate(ecosystem=ecosystem))
    records = {r["kind"]: r for r in _build(repo, V2, candidate(ecosystem=ecosystem))}
    vulnerable, fixed = records["vulnerable"], records["post_fix"]
    assert helper not in vulnerable["input"]["code"] and helper in fixed["input"]["code"]
    assert vulnerable["provenance"]["excerpt_function"] in {"read", "render"}
    assert vulnerable["localization"]["diff_consistent"] is True
    assert vulnerable["localization"]["fix_added_top_level_lines"] > 0
    assert apply_unified_patch(vulnerable["input"]["code"], vulnerable["target"]["patch"]) == fixed["input"]["code"]
    # the span stays inside the changed method; the helper is never labelled as vulnerable
    code_lines = vulnerable["input"]["code"].splitlines()
    span = code_lines[vulnerable["target"]["start_line"] - 1 : vulnerable["target"]["end_line"]]
    assert span and not any(helper in line for line in span)


def test_inserted_field_is_not_a_member_addition() -> None:
    new = PY_CLASS_NEW.replace("    def _inside_root(self, path):", "    strict = True\n\n    def _inside_root(self, path):")
    with pytest.raises(Rejected, match="change_outside_function"):
        _build(FakeRepo({"store/files.py": PY_CLASS_OLD}, {"store/files.py": new}), V2)


def test_member_addition_needs_exactly_one_changed_function() -> None:
    new = PY_CLASS_NEW.replace("        self.root = root\n", "        self.root = os.fspath(root)\n", 1)
    with pytest.raises(Rejected, match="multiple_functions_changed|change_outside_function"):
        _build(FakeRepo({"store/files.py": PY_CLASS_OLD}, {"store/files.py": new}), V2)


def test_helper_added_without_any_changed_function_is_rejected() -> None:
    new = PY_CLASS_NEW.replace(
        "        if not self._inside_root(path):\n            raise ValueError(\"path escapes the store\")\n", ""
    )
    with pytest.raises(Rejected, match="top_level_addition_without_changed_function|no_logic_change"):
        _build(FakeRepo({"store/files.py": PY_CLASS_OLD}, {"store/files.py": new}), V2)


def test_version_bump_is_metadata_not_logic() -> None:
    old = '__version__ = "1.4.1"\n\n' + PY_OLD
    new = '__version__ = "1.4.2"\n\n' + PY_NEW
    repo = FakeRepo({"shop/orders.py": old}, {"shop/orders.py": new})
    with pytest.raises(Rejected, match="change_outside_function"):
        _build(repo, SETTINGS)
    records = {r["kind"]: r for r in _build(repo, V2)}
    assert "__version__" not in records["vulnerable"]["target"]["patch"]
    assert records["vulnerable"]["localization"]["diff_consistent"] is True


@pytest.mark.parametrize("old_line, new_line", [('DEBUG = False', 'DEBUG = True'), ('__version__ = "1.0"', 'ALLOWED = "1.0"'),
                                                ('VERSION = "1.0"', 'VERSION = os.environ["V"]')])
def test_other_module_level_changes_stay_rejected(old_line, new_line) -> None:
    repo = FakeRepo({"shop/orders.py": old_line + "\n\n" + PY_OLD}, {"shop/orders.py": new_line + "\n\n" + PY_NEW})
    with pytest.raises(Rejected, match="change_outside_function"):
        _build(repo, V2)


@pytest.mark.parametrize("tooling", ["eslint.config.mjs", "test.js", "gulpfile.js", "vite.config.ts"])
def test_tooling_files_are_not_runtime_source(tooling) -> None:
    js_old, js_new = JS_CLASS_OLD.replace("  escape", "  x"), JS_CLASS_NEW
    old = {"lib/renderer.js": JS_CLASS_OLD, tooling: "module.exports = {\n  rules: { eqeqeq: 'error' },\n};\n"}
    new = {"lib/renderer.js": js_new, tooling: "module.exports = {\n  rules: { eqeqeq: ['error', 'allow-null'] },\n};\n"}
    assert js_old  # the tooling file is the only extra change
    repo = FakeRepo(old, new)
    with pytest.raises(Rejected):
        _build(repo, SETTINGS, candidate(ecosystem="npm"))
    records = _build(repo, V2, candidate(ecosystem="npm"))
    assert {r["input"]["file"] for r in records} == {"lib/renderer.js"}


def test_runtime_files_with_tooling_like_names_stay_source() -> None:
    from sentinel.garden.build import is_runtime_source

    for path in ["lib/config.js", "src/settings.py", "app/tasks.py", "lib/contest.py", "server/test_helpers.go"]:
        assert is_runtime_source(path, V2) == is_runtime_source(path, SETTINGS), path


def test_genuine_multi_function_fix_stays_rejected() -> None:
    old = PY_OLD
    new = PY_NEW.replace('status = request.args.get("status", "open")', 'status = request.args.get("status", "open")[:32]')
    with pytest.raises(Rejected, match="multiple_functions_changed"):
        _build(FakeRepo({"shop/orders.py": old}, {"shop/orders.py": new}), V2)
