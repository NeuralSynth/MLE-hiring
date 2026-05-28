import json
import pytest
from llm import llm, clean_json_response


def _call_or_skip(system: str, user: str) -> str:
    """Call llm.complete(), skip the test if the model is unavailable (e.g. OOM)."""
    try:
        return llm.complete(system=system, user=user)
    except Exception as e:
        msg = str(e).lower()
        if any(kw in msg for kw in ("memory", "oom", "unavailable", "connection", "500", "503")):
            pytest.skip(f"LLM unavailable: {e}")
        raise


def test_llm_returns_response():
    result = _call_or_skip(
        system="You are a test assistant.",
        user="Reply with the word: working"
    )
    assert result is not None
    assert len(result) > 0


def test_llm_returns_valid_json():
    result = _call_or_skip(
        system="Reply only with valid JSON. No preamble.",
        user='Return this exact JSON: {"status": "working"}'
    )
    cleaned = clean_json_response(result)
    parsed = json.loads(cleaned)
    assert parsed["status"] == "working"


def test_llm_temperature_is_zero():
    """Same prompt twice should return identical output."""
    prompt = "Reply with a single random number between 1 and 1000."
    result_1 = _call_or_skip(system="You are a test assistant.", user=prompt)
    result_2 = _call_or_skip(system="You are a test assistant.", user=prompt)
    assert result_1 == result_2


def test_clean_json_response_strips_think_tags():
    """HANDOFF §1 — Qwen3 chain-of-thought blocks must be stripped before json.loads()."""
    raw = '<think>\nLet me think about this...\nThe answer is clear.\n</think>\n{"result": "ok"}'
    cleaned = clean_json_response(raw)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    parsed = json.loads(cleaned)
    assert parsed["result"] == "ok"


def test_clean_json_response_strips_markdown_fences():
    """Markdown code fences around JSON should be stripped."""
    raw = '```json\n{"key": "value"}\n```'
    cleaned = clean_json_response(raw)
    assert "```" not in cleaned
    parsed = json.loads(cleaned)
    assert parsed["key"] == "value"


def test_clean_json_response_strips_preamble_and_suffix():
    """Prose around the JSON object should be stripped (weak-model preamble)."""
    raw = 'Sure, here is the result: {"key": "value"} hope that helps!'
    cleaned = clean_json_response(raw)
    assert json.loads(cleaned) == {"key": "value"}


def test_clean_json_response_ignores_braces_inside_strings():
    """Braces inside string values must not break the balanced extraction."""
    raw = '{"text": "use {curly} braces", "ok": true}'
    parsed = json.loads(clean_json_response(raw))
    assert parsed["text"] == "use {curly} braces"
    assert parsed["ok"] is True
