"""Masked values must never leave the process.

Masking after ingestion is not masking — it is deletion, and by then the value
has been written to ClickHouse, backed up and possibly indexed. The mask is
installed as `Langfuse(mask=...)` at construction so it runs on every input,
output and metadata value the SDK is about to send.

These tests are cheap and they guard the one mistake that cannot be undone.
"""

from __future__ import annotations

import pytest

from ollama_langfuse_rag import mask_pii

SECRETS = [
    ("email", "reach me at aya.nasser@example.com please", "example.com"),
    ("egyptian mobile", "call 01012345678 after 5", "01012345678"),
    ("national id", "id 29901011234567 on file", "29901011234567"),
    ("card", "card 4111 1111 1111 1111 expires soon", "4111"),
]


@pytest.mark.parametrize("label,text,secret", SECRETS)
def test_secret_does_not_survive_masking(label: str, text: str, secret: str) -> None:
    """Each pattern class is redacted out of a plain string."""
    assert secret not in mask_pii(text), f"{label} survived the mask"


def test_masking_recurses_into_nested_payloads() -> None:
    """Trace payloads are nested, so a top-level-only mask would miss most of them."""
    payload = {
        "input": {"question": "mail aya.nasser@example.com"},
        "docs": [{"text": "id 29901011234567"}, {"text": "safe"}],
    }
    masked = str(mask_pii(payload))
    assert "example.com" not in masked
    assert "29901011234567" not in masked
    assert "safe" in masked, "masking must not eat non-PII content"


def test_masking_leaves_ordinary_text_alone() -> None:
    """An over-eager mask that eats the answer gets switched off within a week,
    which leaves you with no masking at all. Conservative beats thorough here."""
    text = "PSI above 0.25 is the conventional trigger to retrain, per the 2024 review."
    assert mask_pii(text) == text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
