"""The agent's chat models: local Ollama by default, any OpenAI-compatible endpoint on demand."""

import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from steam_game_analyst.agent.llm import chat_models
from steam_game_analyst.config import Settings

WORKERS_AI = "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1"


def hosted(**kw) -> Settings:
    return Settings(
        llm_provider="openai",
        llm_base_url=WORKERS_AI,
        llm_api_key="test-key",
        **kw,
    )


def test_default_provider_is_local_ollama():
    s = Settings(OLLAMA_HOST="http://ollama:11434")
    llm, router = chat_models(s, "qwen2.5:14b")
    assert isinstance(llm, ChatOllama) and isinstance(router, ChatOllama)
    assert llm.base_url == "http://ollama:11434" and llm.model == "qwen2.5:14b"
    assert router.format == "json" and router.temperature == 0.0
    assert llm.temperature == 0.1 and llm.num_predict == 1536


def test_openai_provider_points_at_the_hosted_endpoint():
    llm, router = chat_models(hosted(), "@cf/qwen/qwen3-30b-a3b-fp8")
    assert isinstance(llm, ChatOpenAI) and isinstance(router, ChatOpenAI)
    assert llm.openai_api_base == WORKERS_AI and llm.model_name == "@cf/qwen/qwen3-30b-a3b-fp8"
    assert llm.openai_api_key is not None and llm.openai_api_key.get_secret_value() == "test-key"
    assert llm.temperature == 0.1 and router.temperature == 0.0
    assert llm.max_tokens == 1536


def test_openai_provider_needs_an_endpoint_and_a_key():
    with pytest.raises(ValidationError, match="SGA_LLM_BASE_URL"):
        Settings(llm_provider="openai", llm_api_key="k")
    with pytest.raises(ValidationError, match="SGA_LLM_API_KEY"):
        Settings(llm_provider="openai", llm_base_url=WORKERS_AI)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValidationError):
        Settings(llm_provider="bard")


def test_api_key_is_not_shown_in_the_settings_repr():
    assert "test-key" not in repr(hosted())


def test_qwen3_on_the_hosted_endpoint_runs_without_reasoning():
    from steam_game_analyst.agent.llm import HostedChatOpenAI

    llm, router = chat_models(hosted(), "@cf/qwen/qwen3-30b-a3b-fp8")
    assert isinstance(llm, HostedChatOpenAI) and isinstance(router, HostedChatOpenAI)
    assert llm.no_think and router.no_think
    other, _ = chat_models(hosted(), "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    assert isinstance(other, HostedChatOpenAI) and not other.no_think


def test_tool_call_turns_send_a_string_content():
    """Workers AI rejects the null content OpenAI clients send with assistant tool calls."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    llm, _ = chat_models(hosted(), "@cf/qwen/qwen3-30b-a3b-fp8")
    call = {"name": "query_catalog", "args": {"sql": "SELECT 1"}, "id": "T1"}
    msgs = [
        HumanMessage("q"),
        AIMessage("", tool_calls=[call]),
        ToolMessage("1", tool_call_id="T1"),
    ]
    payload = llm._get_request_payload(msgs)
    assert [m["content"] for m in payload["messages"]] == ["/no_think", "q", "", "1"]
    assert payload["messages"][2]["tool_calls"][0]["id"] == "T1"


def test_no_think_switch_goes_on_the_system_prompt_without_mutating_it():
    from langchain_core.messages import HumanMessage, SystemMessage

    from steam_game_analyst.agent.llm import no_think

    original = [SystemMessage("Route the question."), HumanMessage("hi")]
    out = no_think(original)
    assert out[0].content == "Route the question.\n/no_think" and out[1] is original[1]
    assert original[0].content == "Route the question."
    assert no_think(out)[0].content == out[0].content  # applied once
    bare = no_think([HumanMessage("hi")])
    assert isinstance(bare[0], SystemMessage) and bare[0].content == "/no_think"


def test_leftover_think_blocks_are_stripped():
    from steam_game_analyst.agent.llm import strip_think

    assert strip_think("<think>\n\n</think>\n\n391") == "391"
    assert strip_think("\n\n391") == "391"
    assert strip_think('<think>a {json} b</think>{"in_scope": true}') == '{"in_scope": true}'


def test_reasoning_can_be_left_on_with_room_for_it():
    from steam_game_analyst.agent.llm import MAX_OUTPUT_TOKENS

    llm, router = chat_models(hosted(llm_reasoning=True), "@cf/qwen/qwen3-30b-a3b-fp8")
    assert not llm.no_think and not router.no_think
    assert llm.max_tokens > MAX_OUTPUT_TOKENS
