"""
agent.workspace — the files an agent may read and edit, and what it did to them.

In-memory rather than on-disk, for three reasons that all matter later:

* **An episode is serialisable.** The whole workspace is a dict of strings,
  so a trajectory can carry the state at any step and replay can reconstruct
  it without a filesystem.
* **Escape is impossible, not merely forbidden.** There is no path outside
  the dict to traverse to. A path check that rejects ``..`` is a check that
  can be wrong; not having a filesystem cannot be.
* **Mutation is observable.** Every write is recorded, which is what the
  no-progress termination condition and the edit-churn metric are computed
  from.

Files are flushed to a real directory only when something needs to compile
them, and then into a fresh temporary directory that is discarded after.
"""

from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["FileEdit", "Workspace", "WorkspaceError"]


class WorkspaceError(ValueError):
    """A rejected operation, with a message meant for the agent to read."""


@dataclass
class FileEdit:
    """One mutation, recorded.

    ``before_hash`` and ``after_hash`` are what make churn measurable: an
    agent that writes a file and then writes it back is visible as two
    edits whose hashes return to where they started.
    """

    step_index: int
    path: str
    operation: str  # "create" | "update" | "delete" | "no-op"
    before_hash: str
    after_hash: str
    lines_added: int = 0
    lines_removed: int = 0

    @property
    def reverted(self) -> bool:
        """True when the write left the file exactly as it was."""
        return self.before_hash == self.after_hash


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalise(path: str) -> str:
    """Canonical form of *path*, rejecting anything that tries to escape.

    There is nothing to escape *to* — the workspace is a dict — but a path
    containing ``..`` still signals a confused agent, and telling it so is
    more useful than silently flattening the path.
    """
    cleaned = path.strip().replace("\\", "/")
    if not cleaned:
        raise WorkspaceError("Empty path.")
    if cleaned.startswith("/"):
        raise WorkspaceError(f"{path!r} is absolute; use a path relative to the workspace root.")
    # A literal "./" prefix, removed as a prefix. str.lstrip takes a set of
    # characters, not a prefix, so lstrip("./") would turn "../secrets" into
    # "secrets" — silently granting exactly what the check below refuses.
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if ".." in parts:
        raise WorkspaceError(f"{path!r} contains '..'; workspace paths may not traverse upward.")
    if not parts:
        raise WorkspaceError("Empty path.")
    return "/".join(parts)


class Workspace:
    """The agent's view of a repository: a flat map of path to content.

    Parameters
    ----------
    files
        Initial contents, ``{path: content}``.
    read_only
        Paths the agent may read but not write. The test suite goes here:
        an agent that can edit the tests it is judged by is not solving the
        task, and catching that after the fact is much harder than
        preventing it.
    max_file_chars
        Ceiling on a single file's size.
    """

    def __init__(
        self,
        files: dict[str, str] | None = None,
        read_only: set[str] | None = None,
        max_file_chars: int = 400_000,
    ) -> None:
        self._files: dict[str, str] = {
            _normalise(path): content for path, content in (files or {}).items()
        }
        self._read_only = {_normalise(path) for path in (read_only or set())}
        self.max_file_chars = max_file_chars
        self.edits: list[FileEdit] = []
        self._initial = dict(self._files)

    # ── reading ───────────────────────────────────────────────────────

    def exists(self, path: str) -> bool:
        return _normalise(path) in self._files

    def read(self, path: str) -> str:
        """Return the content of *path*.

        Raises:
            WorkspaceError: when the file does not exist, naming what does.
        """
        key = _normalise(path)
        if key not in self._files:
            raise WorkspaceError(
                f"{path!r} does not exist. Files present: {sorted(self._files)}"
            )
        return self._files[key]

    def list_paths(self, pattern: str = "") -> list[str]:
        """Sorted paths, optionally filtered by a glob such as ``*.java``."""
        paths = sorted(self._files)
        if pattern:
            paths = [p for p in paths if fnmatch.fnmatch(p, pattern) or fnmatch.fnmatch(Path(p).name, pattern)]
        return paths

    def is_read_only(self, path: str) -> bool:
        return _normalise(path) in self._read_only

    def snapshot(self) -> dict[str, str]:
        """A copy of every file, safe to store in a trajectory."""
        return dict(self._files)

    # ── writing ───────────────────────────────────────────────────────

    def write(self, path: str, content: str, step_index: int = 0) -> FileEdit:
        """Create or replace *path*, recording the edit.

        Raises:
            WorkspaceError: when the path is read-only or the content is
                over the size ceiling.
        """
        key = _normalise(path)
        if key in self._read_only:
            raise WorkspaceError(
                f"{path!r} is read-only and cannot be modified. It is part of "
                f"how this task is scored; change the code under test instead."
            )
        if len(content) > self.max_file_chars:
            raise WorkspaceError(
                f"Content for {path!r} is {len(content)} characters; the limit "
                f"is {self.max_file_chars}."
            )

        before = self._files.get(key, "")
        existed = key in self._files
        self._files[key] = content

        before_lines = before.splitlines()
        after_lines = content.splitlines()
        edit = FileEdit(
            step_index=step_index,
            path=key,
            operation="update" if existed else "create",
            before_hash=_hash(before) if existed else "",
            after_hash=_hash(content),
            lines_added=max(0, len(after_lines) - len(before_lines)),
            lines_removed=max(0, len(before_lines) - len(after_lines)),
        )
        if existed and before == content:
            edit.operation = "no-op"
        self.edits.append(edit)
        return edit

    def delete(self, path: str, step_index: int = 0) -> FileEdit:
        """Remove *path*, recording the edit."""
        key = _normalise(path)
        if key in self._read_only:
            raise WorkspaceError(f"{path!r} is read-only and cannot be deleted.")
        if key not in self._files:
            raise WorkspaceError(f"{path!r} does not exist.")

        before = self._files.pop(key)
        edit = FileEdit(
            step_index=step_index,
            path=key,
            operation="delete",
            before_hash=_hash(before),
            after_hash="",
            lines_removed=len(before.splitlines()),
        )
        self.edits.append(edit)
        return edit

    # ── history ───────────────────────────────────────────────────────

    def files_touched(self) -> list[str]:
        """Every path the agent wrote to, in first-touch order."""
        seen: list[str] = []
        for edit in self.edits:
            if edit.path not in seen:
                seen.append(edit.path)
        return seen

    def effective_edits(self) -> int:
        """Writes that actually changed something.

        A write whose content equals what was already there is a no-op. It
        still costs a step and still costs money, and the no-progress
        condition must not count it as progress.
        """
        return sum(1 for edit in self.edits if edit.operation != "no-op")

    def churn(self) -> int:
        """Lines written and then written away again.

        An agent that edits a file, breaks something, and reverts has done
        work that left no trace in the final diff. This is the size of that
        invisible work.
        """
        total = 0
        for path in self.files_touched():
            if path not in self._files:
                continue
            original = self._initial.get(path, "")
            current = self._files[path]
            written = sum(
                edit.lines_added + edit.lines_removed
                for edit in self.edits
                if edit.path == path
            )
            net = abs(len(current.splitlines()) - len(original.splitlines()))
            total += max(0, written - net)
        return total

    def diff_summary(self) -> dict[str, str]:
        """``{path: "created" | "modified" | "deleted"}`` against the start."""
        summary: dict[str, str] = {}
        for path in set(self._initial) | set(self._files):
            before = self._initial.get(path)
            after = self._files.get(path)
            if before == after:
                continue
            if before is None:
                summary[path] = "created"
            elif after is None:
                summary[path] = "deleted"
            else:
                summary[path] = "modified"
        return summary

    # ── materialisation ───────────────────────────────────────────────

    def materialise(self, directory: Path | str) -> Path:
        """Write every file into *directory* so a compiler can see them.

        The directory is the caller's to create and remove; nothing here
        cleans up, because the sandbox that consumes it knows when it is
        finished and this does not.
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for path, content in self._files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return root

    def __len__(self) -> int:
        return len(self._files)

    def __contains__(self, path: str) -> bool:
        try:
            return _normalise(path) in self._files
        except WorkspaceError:
            return False
