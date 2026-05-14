"""
Tests for the prompt template renderer and Jinja2 templates.

Validates template rendering, Prompt construction, code extraction, and
few-shot formatting.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from contracts import Prompt
from llm_gateway.models import PromptRenderer


# ── PromptRenderer construction ───────────────────────────────────────

class TestPromptRendererInit:
    """Tests for PromptRenderer initialisation."""

    def test_default_template_dir(self) -> None:
        """Default template_dir points to llm_gateway/templates/."""
        renderer = PromptRenderer()
        assert renderer.template_dir.name == "templates"
        assert renderer.template_dir.parent.name == "llm_gateway"

    def test_custom_template_dir(self, tmp_path: Path) -> None:
        """Custom template_dir is accepted."""
        renderer = PromptRenderer(template_dir=tmp_path)
        assert renderer.template_dir == tmp_path

    def test_list_templates(self) -> None:
        """list_templates returns .j2 files from the template directory."""
        renderer = PromptRenderer()
        templates = renderer.list_templates()
        assert len(templates) >= 12
        assert "codegen_zero_shot.j2" in templates
        assert "system_default.j2" in templates

    def test_list_templates_empty_dir(self, tmp_path: Path) -> None:
        """list_templates returns empty list for a dir with no .j2 files."""
        renderer = PromptRenderer(template_dir=tmp_path)
        assert renderer.list_templates() == []


# ── render_template ───────────────────────────────────────────────────

class TestRenderTemplate:
    """Tests for raw template rendering."""

    def test_codegen_zero_shot(self) -> None:
        """codegen_zero_shot.j2 renders with title and description."""
        renderer = PromptRenderer()
        result = renderer.render_template(
            "codegen_zero_shot.j2",
            {"title": "FizzBuzz", "description": "Print 1-100 with FizzBuzz rules."},
        )
        assert "FizzBuzz" in result
        assert "Print 1-100" in result

    def test_codegen_zero_shot_with_signature(self) -> None:
        """codegen_zero_shot.j2 includes method signature when provided."""
        renderer = PromptRenderer()
        result = renderer.render_template(
            "codegen_zero_shot.j2",
            {
                "title": "Add",
                "description": "Add two numbers.",
                "signature": "public static int add(int a, int b)",
            },
        )
        assert "Method Signature" in result
        assert "public static int add" in result

    def test_codegen_few_shot(self) -> None:
        """codegen_few_shot.j2 renders examples and target problem."""
        renderer = PromptRenderer()
        examples = [
            {"problem": "Return 1", "solution": "return 1;"},
            {"problem": "Return 2", "solution": "return 2;"},
        ]
        result = renderer.render_template(
            "codegen_few_shot.j2",
            {"title": "Return 3", "description": "Return 3.", "examples": examples},
        )
        assert "Example 1" in result
        assert "Example 2" in result
        assert "Return 3" in result

    def test_codegen_cot(self) -> None:
        """codegen_cot.j2 renders with chain-of-thought instructions."""
        renderer = PromptRenderer()
        result = renderer.render_template(
            "codegen_cot.j2",
            {"title": "Sort Array", "description": "Sort an integer array."},
        )
        assert "step by step" in result.lower()
        assert "Sort Array" in result

    def test_bugfix_basic(self) -> None:
        """bugfix_basic.j2 renders with buggy code and description."""
        renderer = PromptRenderer()
        result = renderer.render_template(
            "bugfix_basic.j2",
            {
                "bug_description": "Off-by-one error in loop.",
                "buggy_code": "for (int i = 0; i <= arr.length; i++)",
            },
        )
        assert "Off-by-one" in result
        assert "i <= arr.length" in result

    def test_bugfix_with_test(self) -> None:
        """bugfix_with_test.j2 renders with failing test."""
        renderer = PromptRenderer()
        result = renderer.render_template(
            "bugfix_with_test.j2",
            {
                "bug_description": "NPE on null input.",
                "buggy_code": "s.length()",
                "failing_test": "assertThrows(NPE, () -> foo(null));",
            },
        )
        assert "Failing Test" in result
        assert "NPE" in result

    def test_bugfix_cot(self) -> None:
        """bugfix_cot.j2 renders with chain-of-thought debugging instructions."""
        renderer = PromptRenderer()
        result = renderer.render_template(
            "bugfix_cot.j2",
            {"bug_description": "Wrong return value.", "buggy_code": "return -1;"},
        )
        assert "step by step" in result.lower()

    def test_refactor_basic(self) -> None:
        """refactor_basic.j2 renders with class source."""
        renderer = PromptRenderer()
        result = renderer.render_template(
            "refactor_basic.j2",
            {
                "class_name": "GodClass",
                "source_code": "public class GodClass { }",
                "task_description": "Decompose into smaller classes.",
            },
        )
        assert "GodClass" in result
        assert "single-responsibility" in result.lower()

    def test_refactor_with_metrics(self) -> None:
        """refactor_with_metrics.j2 renders with CK metric values."""
        renderer = PromptRenderer()
        result = renderer.render_template(
            "refactor_with_metrics.j2",
            {
                "class_name": "BigService",
                "source_code": "public class BigService {}",
                "task_description": "Reduce coupling.",
                "wmc": 45,
                "cbo": 22,
                "lcom": 18,
                "rfc": 70,
                "loc": 800,
            },
        )
        assert "45" in result
        assert "22" in result
        assert "WMC" in result

    def test_refactor_cot(self) -> None:
        """refactor_cot.j2 renders with analysis instructions."""
        renderer = PromptRenderer()
        result = renderer.render_template(
            "refactor_cot.j2",
            {
                "class_name": "MonolithService",
                "source_code": "public class MonolithService {}",
                "task_description": "Split by responsibility.",
            },
        )
        assert "step by step" in result.lower()
        assert "cohesion" in result.lower()

    def test_system_default(self) -> None:
        """system_default.j2 renders without variables."""
        renderer = PromptRenderer()
        result = renderer.render_template("system_default.j2", {})
        assert "expert" in result.lower()

    def test_system_strict(self) -> None:
        """system_strict.j2 renders without variables."""
        renderer = PromptRenderer()
        result = renderer.render_template("system_strict.j2", {})
        assert "ONLY" in result

    def test_system_expert(self) -> None:
        """system_expert.j2 renders without variables."""
        renderer = PromptRenderer()
        result = renderer.render_template("system_expert.j2", {})
        assert "Javadoc" in result

    def test_missing_template_raises(self) -> None:
        """Requesting a non-existent template raises TemplateNotFound."""
        from jinja2 import TemplateNotFound

        renderer = PromptRenderer()
        with pytest.raises(TemplateNotFound):
            renderer.render_template("does_not_exist.j2", {})

    def test_missing_variable_raises(self) -> None:
        """Missing required variable raises UndefinedError."""
        from jinja2 import UndefinedError

        renderer = PromptRenderer()
        with pytest.raises(UndefinedError):
            renderer.render_template("codegen_zero_shot.j2", {})


# ── render_prompt ─────────────────────────────────────────────────────

class TestRenderPrompt:
    """Tests for full Prompt object construction."""

    def test_basic_prompt(self) -> None:
        """render_prompt returns a valid Prompt object."""
        renderer = PromptRenderer()
        p = renderer.render_prompt(
            template_name="codegen_zero_shot.j2",
            problem_id="prob-001",
            model_id="gpt-4o",
            variables={"title": "Sum", "description": "Return a+b."},
        )
        assert isinstance(p, Prompt)
        assert p.problem_id == "prob-001"
        assert p.model_id == "gpt-4o"
        assert "Sum" in p.user_message
        assert p.prompt_id.startswith("pmt-")

    def test_with_system_template(self) -> None:
        """render_prompt includes rendered system message."""
        renderer = PromptRenderer()
        p = renderer.render_prompt(
            template_name="codegen_zero_shot.j2",
            problem_id="prob-002",
            model_id="claude-3-5-sonnet-20241022",
            variables={"title": "Max", "description": "Return max of two ints."},
            system_template="system_strict.j2",
        )
        assert "ONLY" in p.system_message
        assert p.metadata["system_template"] == "system_strict.j2"

    def test_metadata_contains_hash(self) -> None:
        """Metadata includes prompt_hash for cache keying."""
        renderer = PromptRenderer()
        p = renderer.render_prompt(
            template_name="codegen_zero_shot.j2",
            problem_id="prob-003",
            model_id="gpt-4o",
            variables={"title": "Min", "description": "Return min."},
        )
        assert "prompt_hash" in p.metadata
        assert len(p.metadata["prompt_hash"]) == 64  # SHA-256 hex

    def test_custom_temperature_and_tokens(self) -> None:
        """Custom temperature and max_tokens are propagated."""
        renderer = PromptRenderer()
        p = renderer.render_prompt(
            template_name="codegen_zero_shot.j2",
            problem_id="prob-004",
            model_id="gpt-4o",
            variables={"title": "X", "description": "Y."},
            temperature=0.7,
            max_tokens=8192,
        )
        assert p.temperature == 0.7
        assert p.max_tokens == 8192


# ── hash_prompt ───────────────────────────────────────────────────────

class TestHashPrompt:
    """Tests for deterministic prompt hashing."""

    def test_deterministic(self) -> None:
        """Same inputs produce the same hash."""
        h1 = PromptRenderer.hash_prompt("sys", "user")
        h2 = PromptRenderer.hash_prompt("sys", "user")
        assert h1 == h2

    def test_different_inputs(self) -> None:
        """Different inputs produce different hashes."""
        h1 = PromptRenderer.hash_prompt("sys1", "user")
        h2 = PromptRenderer.hash_prompt("sys2", "user")
        assert h1 != h2

    def test_hash_length(self) -> None:
        """Hash is a 64-char hex string (SHA-256)."""
        h = PromptRenderer.hash_prompt("a", "b")
        assert len(h) == 64


# ── extract_code_blocks ───────────────────────────────────────────────

class TestExtractCodeBlocks:
    """Tests for code fence extraction."""

    def test_single_fence(self) -> None:
        """Extracts code from a single triple-backtick fence."""
        raw = "Some text\n```java\nint x = 1;\n```\nMore text"
        assert PromptRenderer.extract_code_blocks(raw) == "int x = 1;"

    def test_multiple_fences(self) -> None:
        """Extracts and concatenates multiple fences."""
        raw = "```python\na = 1\n```\nMiddle\n```java\nint b = 2;\n```"
        result = PromptRenderer.extract_code_blocks(raw)
        assert "a = 1" in result
        assert "int b = 2;" in result

    def test_no_fences(self) -> None:
        """Returns stripped original text when no fences found."""
        raw = "  just plain text  "
        assert PromptRenderer.extract_code_blocks(raw) == "just plain text"

    def test_fence_without_language(self) -> None:
        """Extracts code from fence without language tag."""
        raw = "```\nfoo()\n```"
        assert PromptRenderer.extract_code_blocks(raw) == "foo()"


# ── format_few_shot_examples ──────────────────────────────────────────

class TestFormatFewShot:
    """Tests for few-shot example formatting."""

    def test_basic_format(self) -> None:
        """Formats examples with default labels."""
        examples = [
            {"input": "1+1", "output": "2"},
            {"input": "2+2", "output": "4"},
        ]
        result = PromptRenderer.format_few_shot_examples(examples)
        assert "Example 1:" in result
        assert "Example 2:" in result
        assert "Input: 1+1" in result
        assert "Output: 4" in result

    def test_custom_labels(self) -> None:
        """Custom input/output labels are used."""
        examples = [{"input": "x", "output": "y"}]
        result = PromptRenderer.format_few_shot_examples(
            examples, input_label="Question", output_label="Answer"
        )
        assert "Question: x" in result
        assert "Answer: y" in result

    def test_empty_examples(self) -> None:
        """Empty example list returns empty string."""
        assert PromptRenderer.format_few_shot_examples([]) == ""
