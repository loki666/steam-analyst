"""The agent's chat models: the local Ollama server, or an OpenAI-compatible hosted endpoint."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from steam_game_analyst.config import Settings

# num_predict / max_tokens bound a runaway generation (answers and tool turns are far shorter)
MAX_OUTPUT_TOKENS = 1536
# with reasoning on, the hidden reasoning shares the output budget with the reply
REASONING_OUTPUT_TOKENS = 6144
ANSWER_TEMPERATURE = 0.1
HOSTED_TIMEOUT_S = 120.0

# Qwen3 reasons before answering by default. On a metered endpoint that hidden output costs
# ~900 billed tokens per call (output is the expensive side), so the soft switch turns it off.
NO_THINK = "/no_think"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def no_think(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The messages with Qwen3's no-reasoning switch on the system prompt (new list, once)."""
    if not messages or not isinstance(messages[0], SystemMessage):
        return [SystemMessage(NO_THINK), *messages]
    first = messages[0]
    if isinstance(first.content, str) and first.content.endswith(NO_THINK):
        return list(messages)
    switched = first.model_copy(update={"content": f"{first.content}\n{NO_THINK}"})
    return [switched, *messages[1:]]


def strip_think(text: str) -> str:
    """A reply without the (empty) reasoning block some servers leave in front of it."""
    return _THINK_BLOCK.sub("", text).strip()


class HostedChatOpenAI(ChatOpenAI):
    """ChatOpenAI for OpenAI-compatible hosts that are stricter than OpenAI (Workers AI wants a
    string content on every message), with Qwen3's reasoning optionally switched off."""

    no_think: bool = False

    def _get_request_payload(
        self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        messages = self._convert_input(input_).to_messages()
        if self.no_think:
            messages = no_think(messages)
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        payload["messages"] = [
            {**m, "content": ""} if m.get("content") is None else m for m in payload["messages"]
        ]
        return payload

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        if not self.no_think:
            return result
        generations = [
            ChatGeneration(
                message=g.message.model_copy(update={"content": strip_think(g.message.content)})
                if isinstance(g.message.content, str)
                else g.message,
                generation_info=g.generation_info,
            )
            for g in result.generations
        ]
        return ChatResult(generations=generations, llm_output=result.llm_output)


def chat_models(settings: Settings, model: str) -> tuple[BaseChatModel, BaseChatModel]:
    """(answer/tool model, router model) for the configured provider."""
    if settings.llm_provider == "openai":
        return _openai(settings, model)
    return _ollama(settings, model)


def _ollama(settings: Settings, model: str) -> tuple[BaseChatModel, BaseChatModel]:
    from langchain_ollama import ChatOllama

    common: dict[str, Any] = {
        "model": model,
        "base_url": settings.ollama_host,
        "num_ctx": 8192,
        "num_predict": MAX_OUTPUT_TOKENS,
    }
    llm = ChatOllama(temperature=ANSWER_TEMPERATURE, **common)
    router = ChatOllama(temperature=0.0, format="json", **common)
    return llm, router


def _openai(settings: Settings, model: str) -> tuple[BaseChatModel, BaseChatModel]:
    # no JSON mode for the router: support varies across providers, and parse_router already
    # pulls the JSON object out of free text
    common: dict[str, Any] = {
        "model": model,
        "base_url": settings.llm_base_url,
        "api_key": settings.llm_api_key,
        "max_tokens": REASONING_OUTPUT_TOKENS if settings.llm_reasoning else MAX_OUTPUT_TOKENS,
        "timeout": HOSTED_TIMEOUT_S,
        "max_retries": 1,  # a hung endpoint costs at most two timeouts
    }
    common["no_think"] = "qwen3" in model.lower() and not settings.llm_reasoning
    llm = HostedChatOpenAI(temperature=ANSWER_TEMPERATURE, **common)
    router = HostedChatOpenAI(temperature=0.0, **common)
    return llm, router
