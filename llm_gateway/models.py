"""
Gateway-local model helpers and the Jinja2 prompt template renderer.

Re-exports the C1-relevant contract types so that other gateway modules can
``from llm_gateway.models import Prompt, LLMResponse, CostRecord`` without
reaching into the top-level ``contracts`` module directly.

The ``PromptRenderer`` class provides:
* Loading ``.j2`` templates from a directory.
* Variable injection with required-variable validation.
* Few-shot example formatting.
* Rendering a ``Prompt`` object ready for an ``LLMClient``.

Template directory layout::

    templates/
    ├── codegen_zero_shot.j2
    ├── codegen_few_shot.j2
    ├── codegen_cot.j2
    ├── bugfix_basic.j2
    ├── bugfix_with_test.j2
    ├── bugfix_cot.j2
    ├── refactor_basic.j2
    ├── refactor_with_metrics.j2
    ├── refactor_cot.j2
    ├── system_default.j2
    ├── system_strict.j2
    └── system_expert.j2
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

# Re-export the three C1 contract types for internal convenience.
from contracts import CostRecord, LLMResponse, Prompt  # noqa: F401


def _new_id(prefix: str = "pmt") -> str:
    """Generate a short unique id with an optional prefix.

    Args:
        prefix: Short string prepended to the UUID segment.

    Returns:
        A string like ``"pmt-a3f8b2c1"``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class PromptRenderer:
    """Jinja2-based prompt template engine.

    Loads ``.j2`` template files from *template_dir*, renders them with
    caller-supplied variables, and returns fully-formed ``Prompt`` objects.

    Attributes:
        env:          The Jinja2 ``Environment`` used for rendering.
        template_dir: Resolved path to the template directory.
    """

    def __init__(self, template_dir: str | Path | None = None) -> None:
        """Initialise the renderer.

        Args:
            template_dir: Directory containing ``.j2`` files.  Defaults to
                ``llm_gateway/templates/``.
        """
        if template_dir is None:
            template_dir = Path(__file__).resolve().parent / "templates"
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ── core API ───────────────────────────────────────────────────────

    def render_template(self, template_name: str, variables: dict[str, Any]) -> str:
        """Render a named template with the given variables.

        Args:
            template_name: Filename inside *template_dir* (e.g.
                ``"codegen_zero_shot.j2"``).
            variables: Key-value pairs injected into the template.

        Returns:
            The rendered string.

        Raises:
            jinja2.TemplateNotFound: If the template does not exist.
            jinja2.UndefinedError: If the template references a variable not
                supplied in *variables*.
        """
        tpl = self.env.get_template(template_name)
        return tpl.render(**variables)

    def render_prompt(
        self,
        *,
        template_name: str,
        problem_id: str,
        model_id: str,
        variables: dict[str, Any] | None = None,
        system_template: str | None = None,
        system_variables: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        metadata: dict[str, Any] | None = None,
    ) -> Prompt:
        """Render a full ``Prompt`` object from templates and metadata.

        Args:
            template_name: User-message template filename.
            problem_id:    Identifier of the target benchmark problem.
            model_id:      Target LLM model string.
            variables:     Variables for the user-message template.
            system_template: Optional system-message template filename.
            system_variables: Variables for the system-message template.
            temperature:   Sampling temperature.
            max_tokens:    Maximum completion tokens.
            metadata:      Extra metadata dict attached to the ``Prompt``.

        Returns:
            A ``Prompt`` ready to pass to an ``LLMClient``.
        """
        variables = variables or {}
        system_variables = system_variables or {}
        metadata = metadata or {}

        user_message = self.render_template(template_name, variables)

        system_message = ""
        if system_template:
            system_message = self.render_template(system_template, system_variables)

        prompt_id = _new_id("pmt")

        # Embed template provenance in metadata.
        metadata.update({
            "template_name": template_name,
            "system_template": system_template or "",
            "prompt_hash": self.hash_prompt(system_message, user_message),
        })

        return Prompt(
            prompt_id=prompt_id,
            problem_id=problem_id,
            model_id=model_id,
            system_message=system_message,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            created_at=datetime.utcnow(),
            metadata=metadata,
        )

    # ── utilities ──────────────────────────────────────────────────────

    def list_templates(self) -> list[str]:
        """Return sorted names of all ``.j2`` files in the template directory.

        Returns:
            List of template filenames.
        """
        if not self.template_dir.exists():
            return []
        return sorted(p.name for p in self.template_dir.glob("*.j2"))

    @staticmethod
    def hash_prompt(system_message: str, user_message: str) -> str:
        """Compute a deterministic SHA-256 hash for a prompt pair.

        The hash is used as the cache key in ``ResponseCache``.

        Args:
            system_message: The system prompt text.
            user_message:   The user prompt text.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        content = f"SYSTEM:{system_message}\nUSER:{user_message}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def extract_code_blocks(raw_text: str) -> str:
        """Extract fenced code blocks from LLM raw output.

        Finds all triple-backtick code fences, strips the optional language
        tag, and concatenates them with blank-line separators.

        Args:
            raw_text: Raw completion text from an LLM.

        Returns:
            Concatenated code blocks, or the original text if no fences found.
        """
        pattern = r"```(?:\w+)?\s*\n(.*?)```"
        blocks = re.findall(pattern, raw_text, re.DOTALL)
        if blocks:
            return "\n\n".join(b.strip() for b in blocks)
        # Handle truncated response: opening fence present but no closing fence
        open_fence = re.match(r"^```(?:\w+)?\s*\n(.*)", raw_text.strip(), re.DOTALL)
        if open_fence:
            return open_fence.group(1).strip()
        return raw_text.strip()

    @staticmethod
    def format_few_shot_examples(
        examples: list[dict[str, str]],
        input_label: str = "Input",
        output_label: str = "Output",
    ) -> str:
        """Format a list of few-shot examples into a single string.

        Each example dict should contain ``"input"`` and ``"output"`` keys.

        Args:
            examples:     List of ``{"input": ..., "output": ...}`` dicts.
            input_label:  Header for the input section.
            output_label: Header for the output section.

        Returns:
            Formatted multi-example string.
        """
        parts: list[str] = []
        for i, ex in enumerate(examples, 1):
            parts.append(
                f"Example {i}:\n{input_label}: {ex['input']}\n"
                f"{output_label}: {ex['output']}"
            )
        return "\n\n".join(parts)
