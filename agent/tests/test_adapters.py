"""
Tests for the three provider adapters.

The adapters exist because the providers disagree about every part of tool
calling. Each of those disagreements is a place to be silently wrong — a
dropped tool result, an argument string parsed as a dict, a cached token
counted twice — and every one of them gets a test here.

Provider responses are stubbed with objects shaped like the SDK's, which is
what the adapters actually read. No network, no SDK, no API key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_gateway.adapters import ADAPTERS, get_adapter
from llm_gateway.adapters.anthropic import AnthropicAdapter
from llm_gateway.adapters.gemini import GeminiAdapter
from llm_gateway.adapters.openai import OpenAIAdapter
from llm_gateway.conversation import (
    Conversation,
    Message,
    ToolCall,
    ToolResult,
    ToolSchema,
)

TOOLS = [
    ToolSchema(
        name="read_file",
        description="Read a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
]


def conversation_with_a_tool_round_trip() -> Conversation:
    """user -> assistant(tool_call) -> user(tool_result): the shape that matters."""
    return Conversation(
        system="You are a helpful agent.",
        messages=[
            Message.user("Fix the bug."),
            Message(
                role="assistant",
                text="Let me look.",
                tool_calls=[
                    ToolCall(call_id="c1", name="read_file", arguments={"path": "A.java"})
                ],
            ),
            Message(
                role="user",
                tool_results=[
                    ToolResult(call_id="c1", name="read_file", content="class A {}")
                ],
            ),
        ],
    )


class TestRegistry:
    def test_every_provider_has_an_adapter(self):
        assert set(ADAPTERS) == {"claude", "gpt4", "gemini"}

    @pytest.mark.parametrize("provider", ["claude", "gpt4", "gemini"])
    def test_get_adapter_returns_the_right_one(self, provider):
        assert get_adapter(provider).provider_name == provider

    def test_unknown_provider_names_the_known_ones(self):
        with pytest.raises(KeyError, match="claude"):
            get_adapter("nonexistent")


class TestAnthropicAdapter:
    adapter = AnthropicAdapter()

    def test_tools_use_input_schema(self):
        encoded = self.adapter.encode_tools(TOOLS)
        assert encoded[0]["input_schema"] == TOOLS[0].parameters
        assert "parameters" not in encoded[0]

    def test_system_prompt_is_top_level_not_a_message(self):
        request = self.adapter.encode_request(
            conversation_with_a_tool_round_trip(), TOOLS, "claude-sonnet-4-6", 1024, 0.0
        )
        assert request["system"] == "You are a helpful agent."
        assert all(message["role"] != "system" for message in request["messages"])

    def test_tool_result_is_a_user_message_with_blocks(self):
        messages = self.adapter.encode_conversation(conversation_with_a_tool_round_trip())
        result_message = messages[-1]
        assert result_message["role"] == "user"
        assert result_message["content"][0]["type"] == "tool_result"
        assert result_message["content"][0]["tool_use_id"] == "c1"

    def test_assistant_tool_call_keeps_text_and_call_together(self):
        messages = self.adapter.encode_conversation(conversation_with_a_tool_round_trip())
        assistant = messages[1]
        types = [block["type"] for block in assistant["content"]]
        assert types == ["text", "tool_use"]

    def test_decodes_tool_use_block(self):
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Looking."),
                SimpleNamespace(type="tool_use", id="tu_1", name="read_file",
                                input={"path": "A.java"}),
            ],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            stop_reason="tool_use",
            model="claude-sonnet-4-6",
        )
        turn = self.adapter.decode_response(response, "claude-sonnet-4-6")
        assert turn.text == "Looking."
        assert turn.stop_reason == "tool_use"
        assert turn.wants_tools
        assert turn.tool_calls[0].arguments == {"path": "A.java"}
        assert turn.tool_calls[0].malformed is False

    def test_cache_tokens_are_kept_separate(self):
        """Cached input is priced differently; folding it in makes cost wrong."""
        response = SimpleNamespace(
            content=[],
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=20,
                cache_read_input_tokens=900, cache_creation_input_tokens=50,
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-6",
        )
        usage = self.adapter.decode_response(response, "claude-sonnet-4-6").usage
        assert usage.input_tokens == 100
        assert usage.cache_read_tokens == 900
        assert usage.cache_write_tokens == 50
        assert usage.total == 1070

    def test_max_tokens_stop_is_preserved(self):
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="truncated mid-")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="max_tokens",
            model="m",
        )
        assert self.adapter.decode_response(response, "m").stop_reason == "max_tokens"


class TestOpenAIAdapter:
    adapter = OpenAIAdapter()

    def test_tools_are_wrapped_in_a_function_object(self):
        encoded = self.adapter.encode_tools(TOOLS)
        assert encoded[0]["type"] == "function"
        assert encoded[0]["function"]["parameters"] == TOOLS[0].parameters

    def test_system_prompt_becomes_the_first_message(self):
        messages = self.adapter.encode_conversation(conversation_with_a_tool_round_trip())
        assert messages[0]["role"] == "system"

    def test_each_tool_result_is_its_own_message(self):
        """OpenAI has no multi-result message, unlike Anthropic."""
        conversation = Conversation(messages=[
            Message(role="user", tool_results=[
                ToolResult(call_id="a", name="t", content="1"),
                ToolResult(call_id="b", name="t", content="2"),
            ])
        ])
        messages = self.adapter.encode_conversation(conversation)
        assert len(messages) == 2
        assert all(message["role"] == "tool" for message in messages)
        assert [message["tool_call_id"] for message in messages] == ["a", "b"]

    def test_arguments_are_re_serialised_as_a_json_string(self):
        messages = self.adapter.encode_conversation(conversation_with_a_tool_round_trip())
        arguments = messages[2]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(arguments, str)
        assert json.loads(arguments) == {"path": "A.java"}

    def test_decodes_a_json_string_into_a_dict(self):
        response = _openai_response(
            tool_calls=[_openai_tool_call("tc_1", "read_file", '{"path": "A.java"}')]
        )
        turn = self.adapter.decode_response(response, "gpt-4o")
        assert turn.tool_calls[0].arguments == {"path": "A.java"}
        assert turn.tool_calls[0].malformed is False

    def test_invalid_json_is_flagged_not_raised(self):
        """Models do emit invalid JSON. Crashing loses the whole episode."""
        response = _openai_response(
            tool_calls=[_openai_tool_call("tc_1", "read_file", '{"path": "A.java')]
        )
        call = self.adapter.decode_response(response, "gpt-4o").tool_calls[0]
        assert call.malformed is True
        assert call.arguments == {}
        assert call.raw_arguments == '{"path": "A.java'

    def test_non_object_json_is_also_malformed(self):
        response = _openai_response(
            tool_calls=[_openai_tool_call("tc_1", "read_file", '"just a string"')]
        )
        assert self.adapter.decode_response(response, "gpt-4o").tool_calls[0].malformed

    def test_malformed_arguments_round_trip_unchanged(self):
        """Re-encoding the model's own broken call must not invent syntax."""
        broken = '{"path": "A.java'
        conversation = Conversation(messages=[
            Message(role="assistant", tool_calls=[
                ToolCall(call_id="c", name="read_file", malformed=True, raw_arguments=broken)
            ])
        ])
        messages = self.adapter.encode_conversation(conversation)
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == broken

    def test_cached_tokens_are_subtracted_not_added(self):
        """OpenAI counts cached tokens inside prompt_tokens, unlike Anthropic."""
        response = _openai_response(
            prompt_tokens=1000, completion_tokens=50, cached_tokens=800
        )
        usage = self.adapter.decode_response(response, "gpt-4o").usage
        assert usage.input_tokens == 200
        assert usage.cache_read_tokens == 800
        assert usage.input_tokens + usage.cache_read_tokens == 1000

    def test_tool_calls_finish_reason_maps_to_tool_use(self):
        response = _openai_response(
            tool_calls=[_openai_tool_call("t", "read_file", "{}")],
            finish_reason="tool_calls",
        )
        assert self.adapter.decode_response(response, "gpt-4o").stop_reason == "tool_use"

    def test_empty_choices_is_an_error_not_a_crash(self):
        response = SimpleNamespace(choices=[], usage=None, model="gpt-4o", id="x")
        turn = self.adapter.decode_response(response, "gpt-4o")
        assert turn.stop_reason == "error"
        assert turn.wants_tools is False


class TestGeminiAdapter:
    adapter = GeminiAdapter()

    def test_declarations_share_one_wrapper(self):
        encoded = self.adapter.encode_tools(TOOLS)
        assert len(encoded) == 1
        assert len(encoded[0]["function_declarations"]) == 1

    def test_assistant_role_is_model(self):
        contents = self.adapter.encode_conversation(conversation_with_a_tool_round_trip())
        assert contents[1]["role"] == "model"

    def test_system_prompt_is_not_in_contents(self):
        request = self.adapter.encode_request(
            conversation_with_a_tool_round_trip(), TOOLS, "gemini-2.5-flash", 1024, 0.0
        )
        assert request["system_instruction"] == "You are a helpful agent."
        assert all(
            "system" not in json.dumps(content) for content in request["contents"]
        )

    def test_results_link_by_name_because_there_is_no_call_id(self):
        contents = self.adapter.encode_conversation(conversation_with_a_tool_round_trip())
        response_part = contents[2]["parts"][0]["function_response"]
        assert response_part["name"] == "read_file"
        assert "c1" not in json.dumps(contents[2])

    def test_errors_are_marked_in_the_response_payload(self):
        conversation = Conversation(messages=[
            Message(role="user", tool_results=[
                ToolResult(call_id="c", name="t", content="boom", is_error=True)
            ])
        ])
        part = self.adapter.encode_conversation(conversation)[0]["parts"][0]
        assert part["function_response"]["response"] == {"error": "boom"}

    def test_decodes_a_function_call(self):
        response = _gemini_response(
            parts=[SimpleNamespace(
                function_call=SimpleNamespace(name="read_file", args={"path": "A.java"}),
                text="",
            )]
        )
        turn = self.adapter.decode_response(response, "gemini-2.5-flash")
        assert turn.tool_calls[0].name == "read_file"
        assert turn.tool_calls[0].arguments == {"path": "A.java"}

    def test_synthesised_call_ids_are_deterministic(self):
        """A recorded trajectory must replay to byte-identical ids."""
        response = _gemini_response(
            parts=[SimpleNamespace(
                function_call=SimpleNamespace(name="read_file", args={}), text=""
            )]
        )
        first = self.adapter.decode_response(response, "m").tool_calls[0].call_id
        second = self.adapter.decode_response(response, "m").tool_calls[0].call_id
        assert first == second

    def test_a_function_call_means_tool_use_even_though_gemini_says_stop(self):
        """Gemini reports STOP alongside a function call; the call decides."""
        response = _gemini_response(
            parts=[SimpleNamespace(
                function_call=SimpleNamespace(name="read_file", args={}), text=""
            )],
            finish_reason="STOP",
        )
        assert self.adapter.decode_response(response, "m").stop_reason == "tool_use"

    def test_plain_text_response_is_end_turn(self):
        response = _gemini_response(
            parts=[SimpleNamespace(function_call=None, text="All done.")]
        )
        turn = self.adapter.decode_response(response, "m")
        assert turn.stop_reason == "end_turn"
        assert turn.text == "All done."
        assert turn.wants_tools is False

    def test_cached_tokens_are_subtracted(self):
        response = _gemini_response(
            parts=[], prompt_tokens=1000, candidates_tokens=40, cached_tokens=700
        )
        usage = self.adapter.decode_response(response, "m").usage
        assert usage.input_tokens == 300
        assert usage.cache_read_tokens == 700

    def test_no_candidates_is_an_error(self):
        response = SimpleNamespace(candidates=[], usage_metadata=None)
        assert self.adapter.decode_response(response, "m").stop_reason == "error"


class TestCrossProviderConsistency:
    """The same conversation must survive every encoder."""

    @pytest.mark.parametrize("provider", ["claude", "gpt4", "gemini"])
    def test_every_adapter_encodes_the_round_trip(self, provider):
        adapter = get_adapter(provider)
        request = adapter.encode_request(
            conversation_with_a_tool_round_trip(), TOOLS, "some-model", 1024, 0.0
        )
        assert request["model"] == "some-model"
        encoded = json.dumps(request, default=str)
        # The tool name, the argument and the result must all survive.
        assert "read_file" in encoded
        assert "A.java" in encoded
        assert "class A {}" in encoded

    @pytest.mark.parametrize("provider", ["claude", "gpt4", "gemini"])
    def test_empty_tool_list_omits_the_tools_field(self, provider):
        adapter = get_adapter(provider)
        request = adapter.encode_request(
            Conversation(messages=[Message.user("hello")]), [], "m", 100, 0.0
        )
        assert "tools" not in request


# ── stub builders ─────────────────────────────────────────────────────

def _openai_tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _openai_response(
    tool_calls=None,
    text=None,
    finish_reason="stop",
    prompt_tokens=100,
    completion_tokens=20,
    cached_tokens=0,
):
    details = SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens else None
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=tool_calls or []),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=details,
        ),
        model="gpt-4o",
        id="chatcmpl-x",
    )


def _gemini_response(
    parts,
    finish_reason="STOP",
    prompt_tokens=100,
    candidates_tokens=20,
    cached_tokens=0,
):
    return SimpleNamespace(
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=parts),
            finish_reason=SimpleNamespace(name=finish_reason),
        )],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidates_tokens,
            cached_content_token_count=cached_tokens,
        ),
    )
