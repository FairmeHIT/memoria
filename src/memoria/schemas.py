"""Strict public request and response schemas for the Leaderboard contract."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContractModel(BaseModel):
    """Reject undeclared fields so contract drift fails loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Message(ContractModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=200_000)
    timestamp: int | None = Field(default=None, ge=0)

    @field_validator("content")
    @classmethod
    def content_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class AddRequest(ContractModel):
    request_id: str = Field(min_length=1, max_length=512)
    messages: list[Message] = Field(min_length=1, max_length=20)
    user_id: str = Field(min_length=1, max_length=512)
    session_id: str = Field(min_length=1, max_length=512)

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def identifier_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must not be blank")
        return value


class AddResponse(ContractModel):
    success: Literal[True]
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(ContractModel):
    query: str = Field(min_length=1, max_length=20_000)
    options: list[str] | None = Field(default=None, max_length=20)
    user_id: str = Field(min_length=1, max_length=512)
    top_k: int = Field(ge=1, le=1_000)

    @field_validator("query", "user_id")
    @classmethod
    def search_text_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("options")
    @classmethod
    def options_must_not_contain_blank_values(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not option.strip() for option in value):
            raise ValueError("options must not contain blank values")
        return value


class SearchHit(ContractModel):
    id: str
    content: str
    score: float | None = None
    created_at: str | None = None


class SearchResponse(ContractModel):
    data: list[SearchHit]


class HealthResponse(ContractModel):
    status: Literal["ok"]

