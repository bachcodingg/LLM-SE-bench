#!/usr/bin/env python3
"""
scripts/check_packaging.py — every first-party import resolves to a tracked file.

A module that exists locally but is not in git is invisible until someone
clones the repository, at which point everything that imports it fails. It
is the one class of bug that a full local test suite cannot catch, because
locally the file is right there.

This repository shipped exactly that: ``.gitignore`` listed ``quality/``
under "outputs", so the whole C3 component was untracked while ``cli.py``
and three MCP tool modules imported it. Every test passed. A fresh clone
could not start.

The check: collect the first-party modules that tracked files import, and
assert each one is itself tracked.

Usage
-----
::

    python scripts/check_packaging.py
    python scripts/check_packaging.py --json

Exit codes: 0 clean, 1 something imported is not tracked, 2 not a git repo.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Top-level names that belong to this project rather than to a dependency.
#: Derived from the directory layout so a new package is covered without
#: editing this list.
def first_party_names() -> set[str]:
    names = {
        entry.name for entry in REPO_ROOT.iterdir()
        if entry.is_dir() and (entry / "__init__.py").exists()
    }
    names |= {
        entry.stem for entry in REPO_ROOT.glob("*.py")
        if not entry.name.startswith("_")
    }
    return names


def tracked_files() -> set[str]:
    output = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if output.returncode != 0:
        raise RuntimeError("not a git repository")
    return {line.replace("\\", "/") for line in output.stdout.split() if line}


def imported_modules(path: Path) -> set[str]:
    """First-party top-level module names imported by *path*.

    Parsed rather than grepped, so a name inside a string or a comment does
    not count and a lazy import inside a function does.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import; it cannot leave the package.
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def is_tracked(module: str, tracked: set[str]) -> bool:
    """Whether *module* resolves to something git knows about."""
    return (
        f"{module}.py" in tracked
        or f"{module}/__init__.py" in tracked
        or any(entry.startswith(f"{module}/") for entry in tracked)
    )


def check() -> dict[str, Any]:
    """Find first-party imports that are not tracked."""
    tracked = tracked_files()
    first_party = first_party_names()

    missing: dict[str, list[str]] = {}
    checked = 0

    for rel in sorted(tracked):
        if not rel.endswith(".py"):
            continue
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        checked += 1
        for module in imported_modules(path) & first_party:
            if not is_tracked(module, tracked):
                missing.setdefault(module, []).append(rel)

    return {
        "tracked_python_files": checked,
        "first_party_packages": sorted(first_party),
        "untracked_but_imported": {
            module: sorted(importers)
            for module, importers in sorted(missing.items())
        },
        "passed": not missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    try:
        result = check()
    except RuntimeError as exc:
        print(f"check_packaging: {exc}")
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1

    print(
        f"packaging check — {result['tracked_python_files']} tracked Python "
        f"file(s), {len(result['first_party_packages'])} first-party packages"
    )
    if result["passed"]:
        print("CLEAN — every first-party import resolves to a tracked file.")
        return 0

    print("\nImported but NOT tracked by git:")
    for module, importers in result["untracked_but_imported"].items():
        print(f"\n  {module}  — imported by {len(importers)} tracked file(s):")
        for importer in importers[:8]:
            print(f"    {importer}")
    print(
        "\nA fresh clone of this repository cannot run. Check .gitignore: a "
        "rule meant for generated output is probably matching source too."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
