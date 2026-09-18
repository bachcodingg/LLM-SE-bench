#!/usr/bin/env python3
"""
scripts/build_lockfile.py — pin the versions of every declared dependency.

``pyproject.toml`` declares floors (``pandas>=2.2``).  Floors are right for
a library and wrong for a benchmark: a reader who installs three years from
now gets different numerics and cannot tell whether a difference in results
came from the model or from SciPy.  This writes the resolved versions to
``requirements.lock``.

Direct dependencies only.  A full transitive lock wants ``uv`` or
``poetry``; when this repo grows one, ``build_run_artifact.py`` picks up
``uv.lock`` or ``poetry.lock`` automatically and this script becomes
redundant.

Usage
-----
::

    python scripts/build_lockfile.py            # write requirements.lock
    python scripts/build_lockfile.py --check    # fail if it would change
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKFILE = REPO_ROOT / "requirements.lock"

#: Leading distribution name of a PEP 508 requirement string.
_NAME_RE = re.compile(r"^([A-Za-z0-9_.\-]+)")


def dependency_groups() -> dict[str, list[str]]:
    """Return ``{group: [requirement, ...]}`` from pyproject."""
    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    groups = {"runtime": list(project.get("dependencies", []))}
    groups.update(
        {name: list(deps) for name, deps in project.get("optional-dependencies", {}).items()}
    )
    return groups


def render() -> str:
    """Render the lockfile from the currently installed environment."""
    lines = [
        "# requirements.lock — resolved versions of every dependency declared",
        "# in pyproject.toml, as installed in the environment that generated",
        "# this file. Regenerate with `python scripts/build_lockfile.py`.",
        "#",
        "# This pins direct dependencies only. It records the environment at",
        "# generation time; it is not a reconstruction of the environment that",
        "# produced results/2026-05-run, which predates the file. Treat a",
        "# version difference as a reason to re-check numeric results, not as",
        "# proof that they changed.",
        "#",
        f"# python {sys.version.split()[0]}",
        "",
    ]
    for group, deps in dependency_groups().items():
        lines.append(f"# --- {group} ---")
        for dep in deps:
            match = _NAME_RE.match(dep)
            if match is None:
                lines.append(f"# unparseable requirement: {dep}")
                continue
            name = match.group(1)
            try:
                lines.append(f"{name}=={version(name)}")
            except PackageNotFoundError:
                lines.append(f"# {name}: declared but not installed here")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if the lockfile is out of date.")
    args = parser.parse_args(argv)

    rendered = render()
    if args.check:
        current = LOCKFILE.read_text(encoding="utf-8") if LOCKFILE.exists() else ""
        if current != rendered:
            print("requirements.lock is out of date; run scripts/build_lockfile.py")
            return 1
        print("requirements.lock is up to date")
        return 0

    LOCKFILE.write_text(rendered, encoding="utf-8")
    pinned = sum(1 for line in rendered.splitlines() if "==" in line)
    print(f"wrote requirements.lock — {pinned} pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
