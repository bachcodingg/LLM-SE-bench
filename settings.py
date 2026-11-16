"""
settings.py — environment-driven settings shared across components.

Every variable documented in ``.env.example`` is read here and nowhere
else, so the template and the code cannot drift apart.  ``scripts/
security_sweep.py`` enforces that correspondence.

Nothing in this module reads a credential; API keys stay in
``llm_gateway.config``, which resolves them lazily and never logs them.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DEFAULT_USD_PER_EUR",
    "cache_db_path",
    "usd_per_eur",
    "budget_ceiling_eur",
    "eur_to_usd",
    "usd_to_eur",
    "load_dotenv",
]

#: Fallback conversion rate when ``LLM_SE_BENCH_USD_PER_EUR`` is unset.
#: Pin the variable for a published run — an unpinned rate makes reported
#: euro costs irreproducible.
DEFAULT_USD_PER_EUR = 1.08

#: Where the response cache and cost records live when unconfigured.
DEFAULT_CACHE_FILENAME = "llm_cache.db"


def load_dotenv(path: Path | str | None = None) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Existing environment variables always win, so an explicit shell
    export overrides the file.  Returns the variables that were applied.

    Parameters
    ----------
    path
        The file to read.  Defaults to ``.env`` beside this module.
    """
    env_path = Path(path) if path else Path(__file__).resolve().parent / ".env"
    applied: dict[str, str] = {}
    if not env_path.exists():
        return applied

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


def _env(name: str) -> str | None:
    """Return a non-empty environment variable, or None."""
    value = os.environ.get(name, "").strip()
    return value or None


def cache_db_path() -> Path:
    """Path to the SQLite file holding cached responses and cost records.

    Honours ``LLM_SE_BENCH_CACHE_PATH``.  The file contains full prompts
    and model outputs and must stay out of version control.
    """
    configured = _env("LLM_SE_BENCH_CACHE_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path(DEFAULT_CACHE_FILENAME)


def usd_per_eur() -> float:
    """USD per 1 EUR, from ``LLM_SE_BENCH_USD_PER_EUR``.

    Falls back to :data:`DEFAULT_USD_PER_EUR` when unset or unparseable.
    Provider price lists are in USD; every euro figure this project
    reports is a conversion using exactly this rate.
    """
    raw = _env("LLM_SE_BENCH_USD_PER_EUR")
    if raw is None:
        return DEFAULT_USD_PER_EUR
    try:
        rate = float(raw)
    except ValueError:
        return DEFAULT_USD_PER_EUR
    return rate if rate > 0 else DEFAULT_USD_PER_EUR


def budget_ceiling_eur() -> float | None:
    """Default spend ceiling in EUR, from ``LLM_SE_BENCH_BUDGET_EUR``.

    ``None`` means no ceiling was configured.  Callers that can spend
    money must treat ``None`` as "ask the caller", never as "unlimited".
    """
    raw = _env("LLM_SE_BENCH_BUDGET_EUR")
    if raw is None:
        return None
    try:
        ceiling = float(raw)
    except ValueError:
        return None
    return ceiling if ceiling > 0 else None


def eur_to_usd(amount_eur: float) -> float:
    """Convert EUR to USD at :func:`usd_per_eur`."""
    return amount_eur * usd_per_eur()


def usd_to_eur(amount_usd: float) -> float:
    """Convert USD to EUR at :func:`usd_per_eur`."""
    return amount_usd / usd_per_eur()
