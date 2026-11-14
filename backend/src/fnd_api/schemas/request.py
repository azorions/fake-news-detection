from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from fnd_api.schemas.enums import Depth


class AnalyzeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    depth: Depth = Depth.STANDARD
    max_claims: int = Field(5, ge=0, le=10)
    allow_external: bool = True  # False forces offline providers even when keys exist
    use_cache: bool = True
    store: bool = True  # persist to history


class AnalyzeRequest(BaseModel):
    """Exactly one of ``url`` and ``text``. The input type is inferred: a URL, a headline (text of
    at most 40 words) or longer text."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    url: HttpUrl | None = None
    text: str | None = Field(None, min_length=1, max_length=100_000)
    title: str | None = Field(None, max_length=500)
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)

    @model_validator(mode="after")
    def exactly_one_input(self) -> Self:
        if (self.url is None) == (self.text is None):
            raise ValueError("provide exactly one of 'url' or 'text'")
        return self
