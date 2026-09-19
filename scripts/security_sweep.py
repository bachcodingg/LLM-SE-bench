#!/usr/bin/env python3
"""
scripts/security_sweep.py — pre-publication secret and data-leak sweep.

Run this before every push to a public remote.  It is deliberately
dependency-free (stdlib + ``git``) so it can run in CI without installing
the project.

Checks
------
1.  ``tracked``    Secret-shaped strings in files git currently tracks.
2.  ``history``    Secret-shaped strings anywhere in ``git log -p --all``.
3.  ``paths``      Sensitive files that are tracked but should never be
                   (``.env``, SQLite caches, raw result dumps).
4.  ``ignore``     ``.gitignore`` coverage for every sensitive pattern.
5.  ``envexample`` Every environment variable the code reads is documented
                   in ``.env.example`` — with no value attached.

Usage
-----
::

    python scripts/security_sweep.py              # full sweep
    python scripts/security_sweep.py --skip-history   # fast, working tree only
    python scripts/security_sweep.py --json       # machine-readable

Exit codes: 0 = clean, 1 = findings, 2 = could not run (e.g. not a git repo).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Detection rules
# --------------------------------------------------------------------------

# Each rule is (name, compiled pattern).  Patterns target the *shape* of a
# live credential, not the variable name, so that documentation and
# placeholders such as ``sk-ant-...`` do not trip them.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic_api_key", re.compile(r"sk-ant-(?:api|admin)[A-Za-z0-9_\-]{20,}")),
    ("openai_project_key", re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}")),
    ("openai_legacy_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("bearer_header", re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{30,}")),
]

# Paths that must never be tracked by git.
FORBIDDEN_TRACKED = [
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)\.env\.(?!example$)[^/]+$"),
    re.compile(r"\.db$"),
    re.compile(r"\.sqlite3?$"),
    re.compile(r"^results/raw/"),
    re.compile(r"^cache/"),
]

# Each entry is a thing that must be ignored, followed by the .gitignore
# lines that would achieve it.  Any one of them satisfies the check, so a
# repo may use a broader rule (``results/*``) than the narrow one the
# checklist names (``results/raw/``).
REQUIRED_IGNORES: list[tuple[str, tuple[str, ...]]] = [
    ("local .env", (".env",)),
    ("per-environment .env files", (".env.*",)),
    ("SQLite databases", ("*.db",)),
    ("SQLite databases (alt suffix)", ("*.sqlite3", "*.sqlite")),
    ("raw result dumps", ("results/raw/", "results/", "results/*")),
    ("on-disk caches", ("cache/",)),
]

# Binary / generated extensions that are pointless to scan line-by-line.
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".woff", ".woff2",
    ".zip", ".gz", ".tar", ".whl", ".db", ".sqlite", ".sqlite3", ".pyc",
}


@dataclass
class Finding:
    """A single problem worth a human's attention before publishing."""

    check: str
    severity: str  # "critical" | "warning"
    location: str
    detail: str


# --------------------------------------------------------------------------
# git helpers
# --------------------------------------------------------------------------

def _git(*args: str) -> str:
    """Run a git command in the repo root and return stdout (never raises)."""
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _is_git_repo() -> bool:
    return bool(_git("rev-parse", "--git-dir").strip())


def _tracked_files() -> list[str]:
    return [line for line in _git("ls-files").splitlines() if line]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _scan_text(text: str, location: str, check: str) -> list[Finding]:
    """Apply every secret pattern to *text* and report matches, redacted."""
    findings: list[Finding] = []
    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group(0)
            redacted = f"{token[:8]}...{token[-4:]}" if len(token) > 16 else "<short match>"
            findings.append(
                Finding(
                    check=check,
                    severity="critical",
                    location=location,
                    detail=f"{name}: {redacted}",
                )
            )
    return findings


def check_tracked_files() -> list[Finding]:
    """Scan the content of every tracked file for secret-shaped strings."""
    findings: list[Finding] = []
    for rel in _tracked_files():
        path = REPO_ROOT / rel
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # unreadable file is itself worth reporting
            findings.append(Finding("tracked", "warning", rel, f"unreadable: {exc}"))
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            findings.extend(_scan_text(line, f"{rel}:{line_no}", "tracked"))
    return findings


def check_history() -> list[Finding]:
    """Scan every diff in every branch for secrets that were once committed.

    A secret found here is not fixed by deleting the file: the credential
    must be rotated first, and only then is rewriting history worthwhile.
    """
    patch = _git("log", "-p", "--all", "--no-color")
    if not patch:
        return []
    findings: list[Finding] = []
    commit = "<unknown>"
    for line in patch.splitlines():
        if line.startswith("commit "):
            commit = line.split()[1][:10]
            continue
        if not line.startswith(("+", "-")):
            continue
        findings.extend(_scan_text(line, f"commit {commit}", "history"))
    return findings


def check_forbidden_paths() -> list[Finding]:
    """Report tracked files that should never have been added to git."""
    findings: list[Finding] = []
    for rel in _tracked_files():
        for pattern in FORBIDDEN_TRACKED:
            if pattern.search(rel):
                findings.append(
                    Finding(
                        check="paths",
                        severity="critical",
                        location=rel,
                        detail="sensitive path is tracked by git; remove and rotate any keys",
                    )
                )
                break
    return findings


def check_gitignore() -> list[Finding]:
    """Verify .gitignore covers every sensitive pattern."""
    gitignore = REPO_ROOT / ".gitignore"
    if not gitignore.exists():
        return [Finding("ignore", "critical", ".gitignore", "file is missing")]
    lines = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return [
        Finding(
            "ignore",
            "warning",
            ".gitignore",
            f"nothing ignores {label}; add one of: {', '.join(accepted)}",
        )
        for label, accepted in REQUIRED_IGNORES
        if not any(pattern in lines for pattern in accepted)
    ]


ENV_VAR_RE = re.compile(r"""env_var\s*=\s*["']([A-Z][A-Z0-9_]*)["']|os\.environ(?:\.get)?[\[(]\s*["']([A-Z][A-Z0-9_]*)["']|os\.getenv\(\s*["']([A-Z][A-Z0-9_]*)["']""")

# Variables the code reads that belong to the toolchain, not this project.
ENV_VAR_IGNORE = {"PATH", "HOME", "TEST_LLM_KEY", "PYTHONPATH"}


def check_env_example() -> list[Finding]:
    """Every env var the code reads must appear in .env.example, valueless."""
    findings: list[Finding] = []
    example = REPO_ROOT / ".env.example"
    if not example.exists():
        return [Finding("envexample", "warning", ".env.example", "file is missing")]

    documented: set[str] = set()
    for line_no, line in enumerate(example.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        documented.add(key.strip())
        if value.strip():
            findings.append(
                Finding(
                    "envexample",
                    "critical",
                    f".env.example:{line_no}",
                    f"{key.strip()} has a value; .env.example must list names only",
                )
            )

    referenced: set[str] = set()
    for rel in _tracked_files():
        # Test files reference environment variables as fixtures, not as
        # configuration; documenting a name only a test invents would be
        # misleading.
        if not rel.endswith(".py") or "/tests/" in rel or "/test_" in f"/{rel}":
            continue
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for match in ENV_VAR_RE.finditer(text):
            name = next(g for g in match.groups() if g)
            if name not in ENV_VAR_IGNORE:
                referenced.add(name)

    for name in sorted(referenced - documented):
        findings.append(
            Finding("envexample", "warning", ".env.example", f"undocumented variable: {name}")
        )
    return findings


CHECKS = {
    "paths": check_forbidden_paths,
    "ignore": check_gitignore,
    "envexample": check_env_example,
    "tracked": check_tracked_files,
    "history": check_history,
}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_sweep(skip_history: bool = False) -> list[Finding]:
    """Run every check and return the combined findings."""
    findings: list[Finding] = []
    for name, fn in CHECKS.items():
        if skip_history and name == "history":
            continue
        findings.extend(fn())
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--skip-history", action="store_true",
                        help="Skip the full-history diff scan (much faster).")
    parser.add_argument("--json", action="store_true",
                        help="Emit findings as JSON instead of text.")
    args = parser.parse_args(argv)

    if not _is_git_repo():
        print("security_sweep: not a git repository", file=sys.stderr)
        return 2

    findings = run_sweep(skip_history=args.skip_history)

    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        checks_run = [c for c in CHECKS if not (args.skip_history and c == "history")]
        print(f"security sweep — {len(checks_run)} checks: {', '.join(checks_run)}")
        if not findings:
            print("CLEAN — no findings.")
        else:
            critical = sum(1 for f in findings if f.severity == "critical")
            print(f"{len(findings)} finding(s), {critical} critical:\n")
            for f in findings:
                print(f"  [{f.severity:8s}] {f.check:11s} {f.location}")
                print(f"             {f.detail}")
            print("\nA credential found in history is not fixed by deleting the file.")
            print("Rotate the key first, then rewrite history or re-init the repo.")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
