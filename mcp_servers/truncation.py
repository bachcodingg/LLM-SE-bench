"""
mcp_servers.truncation — bounded tool output.

A 40,000-line Maven log will destroy an agent's context window, so every
tool that can emit unbounded text passes it through here first.

The strategy is deliberately not "take the last N lines".  When a build
fails, the *first* error is the one that matters and the tail is a summary;
when it hangs, the tail is all there is.  So a truncated payload keeps both
ends and says so.

Every truncation is recorded in a :class:`Truncation` attached to the
response.  An agent that failed because its stack trace was cut is the
harness's bug, not the model's, and that distinction is only provable if
the cut is visible in the output.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

__all__ = ["Truncation", "truncate_log", "DEFAULT_MAX_LINES", "DEFAULT_MAX_CHARS"]

#: Lines kept by default.  Roughly 3-4k tokens of build log.
DEFAULT_MAX_LINES = 200

#: Hard character ceiling, applied after the line budget.  Guards against a
#: log that is 200 lines of minified output.
DEFAULT_MAX_CHARS = 20_000

#: Lines matching these are treated as the start of an error block worth
#: keeping even when the head budget has run out.
_ERROR_MARKERS = re.compile(
    r"""
    (^|\s)
    (
      error:                 # javac
    | ERROR\b                # maven, gradle
    | FAILURE\b
    | FAILED\b
    | Exception\b
    | Caused\ by:
    | Tests\ run:            # junit summary with failures
    | \[ERROR\]
    | compilation\ failed
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ELLIPSIS = "... [{dropped} line(s) omitted by llm-se-bench] ..."


class Truncation(BaseModel):
    """What was removed from a tool's output, and why.

    ``truncated`` is False and every count is zero when the payload fitted,
    so a caller can check one field.
    """

    truncated: bool = Field(
        default=False,
        description="True when content was removed. Check this before "
                    "concluding anything from an absence in the text.",
    )
    strategy: str = Field(
        default="none",
        description="How content was removed: 'none', 'head+errors+tail' "
                    "or 'char-cap'.",
    )
    original_lines: int = Field(default=0, description="Line count before truncation.")
    original_chars: int = Field(default=0, description="Character count before truncation.")
    kept_lines: int = Field(default=0, description="Line count after truncation.")
    dropped_lines: int = Field(default=0, description="Lines removed.")
    error_lines_preserved: int = Field(
        default=0,
        description="Lines matching an error marker that were kept even "
                    "though they fell outside the head and tail windows.",
    )


def _error_line_indices(lines: list[str], head: int, tail: int) -> list[int]:
    """Indices of error-marker lines in the middle region, in order."""
    start, end = head, len(lines) - tail
    return [i for i in range(start, end) if _ERROR_MARKERS.search(lines[i])]


def truncate_log(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_chars: int = DEFAULT_MAX_CHARS,
    head_lines: int = 40,
) -> tuple[str, Truncation]:
    """Bound *text* to a line and character budget, keeping what matters.

    Keeps, in order: the first ``head_lines`` lines, then every line in the
    middle that looks like the start of an error, then as much of the tail
    as the remaining budget allows.  Omitted regions are replaced by a
    marker stating how many lines went.

    Parameters
    ----------
    text
        The raw log. May be empty.
    max_lines
        Total lines the result may contain, excluding omission markers.
    max_chars
        Hard character ceiling applied afterwards. A result over this is
        cut mid-line from the front, because the tail of a log is more
        informative than its head once the head has already been kept.
    head_lines
        How much of the beginning to keep. Must be less than *max_lines*.

    Returns
    -------
    tuple[str, Truncation]
        The bounded text and a record of what was removed.
    """
    info = Truncation(
        original_chars=len(text),
        original_lines=len(text.splitlines()) if text else 0,
    )
    if not text:
        return "", info

    lines = text.splitlines()
    info.kept_lines = len(lines)

    if len(lines) <= max_lines and len(text) <= max_chars:
        return text, info

    if len(lines) > max_lines:
        head = min(head_lines, max_lines // 2)
        tail = max_lines - head

        error_indices = _error_line_indices(lines, head, tail)
        # Errors eat into the tail budget rather than the head: the head is
        # where the command and its arguments are.
        if error_indices:
            tail = max(1, tail - len(error_indices))
            error_indices = error_indices[: max_lines - head]

        kept_indices = sorted(
            set(range(head))
            | set(error_indices)
            | set(range(len(lines) - tail, len(lines)))
        )

        pieces: list[str] = []
        previous: int | None = None
        for index in kept_indices:
            if previous is not None and index > previous + 1:
                pieces.append(_ELLIPSIS.format(dropped=index - previous - 1))
            pieces.append(lines[index])
            previous = index

        result = "\n".join(pieces)
        info.truncated = True
        info.strategy = "head+errors+tail"
        info.kept_lines = len(kept_indices)
        info.dropped_lines = len(lines) - len(kept_indices)
        info.error_lines_preserved = len(error_indices)
    else:
        result = text

    if len(result) > max_chars:
        result = result[-max_chars:]
        info.truncated = True
        info.strategy = "char-cap" if info.strategy == "none" else info.strategy + "+char-cap"
        info.kept_lines = len(result.splitlines())

    return result, info
