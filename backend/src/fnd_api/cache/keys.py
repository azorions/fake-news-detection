"""Cache keys. A key is a hash, so the cache never holds submitted text or URLs in its key space."""

from __future__ import annotations

import hashlib
import json

from fnd_api.schemas import AnalyzeOptions

SCAN_TTL_S = 24 * 60 * 60

# Options that say how to handle the result, not what the result is.
_NOT_PART_OF_THE_RESULT = {"use_cache", "store"}


def scan_key(canonical_input: str, options: AnalyzeOptions, model_versions: dict[str, str]) -> str:
    """Key of a finished analysis: the same input, options and models give the same result."""
    payload = {
        "input": canonical_input,
        "options": options.model_dump(mode="json", exclude=_NOT_PART_OF_THE_RESULT),
        "models": model_versions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "scan:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
