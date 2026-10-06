"""Typed agent state and the router's structured decision."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator


class RouterDecision(BaseModel):
    in_scope: bool = True
    needs_retrieval: bool = True
    retrieval_query: str = ""
    tools: list[str] = Field(default_factory=list)
    plan: str = ""

    @field_validator("plan", mode="before")
    @classmethod
    def _plan_text(cls, v: Any) -> Any:
        if isinstance(v, list):  # models often return the steps as a list
            return " ".join(f"{i}. {step}" for i, step in enumerate(v, start=1))
        return v

    @field_validator("tools", mode="before")
    @classmethod
    def _tool_list(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v


class AgentState(TypedDict, total=False):
    question: str
    games: list[dict[str, Any]]  # {appid, name, in_review_corpus}
    route: RouterDecision
    plan: str
    chunks: list[dict[str, Any]]  # retrieved documents: {n, id, label, text, url}
    messages: Annotated[list[BaseMessage], add_messages]  # the tool-calling loop
    tool_results: list[dict[str, Any]]  # {id: "T1", tool, args, ok, content}
    tool_errors: list[dict[str, Any]]  # {tool, args, error} for calls that were refused
    iterations: int
    errors: int
    notes: list[str]  # degraded paths (retrieval down, router fallback, caps hit)
    answer: str
    citations: list[dict[str, Any]]
