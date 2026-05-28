import pytest
from classifier import classify

def test_classifies_devplatform():
    result = classify(
        "I cannot access my HackerRank test results",
        subject="Test Access Issue",
        company="HackerRank"
    )
    assert result["product_area"] == "devplatform"

def test_classifies_claude():
    result = classify(
        "Claude is not responding to my API calls",
        subject="API Issue",
        company="Anthropic"
    )
    assert result["product_area"] == "claude"

def test_classifies_visa():
    result = classify(
        "My Visa card was declined at a merchant abroad",
        subject="Card Declined",
        company="Visa"
    )
    assert result["product_area"] == "visa"

def test_ignores_wrong_company_field():
    """Company field says Visa but content is clearly DevPlatform."""
    result = classify(
        "My coding interview test expired before I could finish",
        subject="Test Expired",
        company="Visa"   # deliberately wrong
    )
    assert result["product_area"] == "devplatform"

def test_out_of_scope():
    result = classify(
        "Who was the first person to walk on the moon?",
        subject="General Question",
        company=None
    )
    assert result["product_area"] == "none"

def test_risk_levels_are_valid():
    result = classify(
        "Someone accessed my account without my permission",
        subject="Unauthorized Access",
        company=None
    )
    assert result["risk_level"] in ["low", "medium", "high", "critical"]

def test_output_schema():
    result = classify("Help with my account", subject="Help", company=None)
    assert "product_area" in result
    assert "request_type" in result
    assert "risk_level" in result
    assert "language" in result


# --- Robustness & normalization (driven with a stubbed LLM) ---

def _stub(monkeypatch, response):
    from llm import llm
    monkeypatch.setattr(llm, "complete", lambda system, user: response)

def test_one_bad_field_does_not_discard_the_rest(monkeypatch):
    _stub(monkeypatch, '{"product_area": null, "request_type": "bug", "risk_level": "high", "language": "fr"}')
    result = classify("body", "subj", "comp")
    assert result["request_type"] == "bug"       # preserved
    assert result["risk_level"] == "high"         # preserved
    assert result["language"] == "fr"             # preserved
    assert result["product_area"] == "none"       # only the bad field defaulted

def test_survives_preamble_around_json(monkeypatch):
    _stub(monkeypatch, 'Sure! Here you go: {"product_area":"claude","request_type":"bug","risk_level":"high","language":"en"} hope this helps')
    result = classify("body", "subj", "comp")
    assert result["product_area"] == "claude"
    assert result["risk_level"] == "high"

def test_language_normalized_to_iso(monkeypatch):
    _stub(monkeypatch, '{"product_area":"claude","request_type":"bug","risk_level":"low","language":"English"}')
    assert classify("b", "s", "c")["language"] == "en"
    _stub(monkeypatch, '{"product_area":"claude","request_type":"bug","risk_level":"low","language":"es-ES"}')
    assert classify("b", "s", "c")["language"] == "es"

def test_fine_request_type_mapped_and_subtype_kept(monkeypatch):
    _stub(monkeypatch, '{"product_area":"visa","request_type":"billing","risk_level":"medium","language":"en"}')
    result = classify("b", "s", "c")
    assert result["request_type"] == "product_issue"    # collapsed for the output schema
    assert result["request_subtype"] == "billing"       # fine label retained

def test_feature_request_is_reachable(monkeypatch):
    _stub(monkeypatch, '{"product_area":"claude","request_type":"feature_request","risk_level":"low","language":"en"}')
    assert classify("b", "s", "c")["request_type"] == "feature_request"

def test_output_keys_whitelisted(monkeypatch):
    _stub(monkeypatch, '{"product_area":"claude","request_type":"bug","risk_level":"low","language":"en","reason":"injected"}')
    result = classify("b", "s", "c")
    assert "reason" not in result
    assert set(result) == {"product_area", "request_type", "request_subtype", "risk_level", "language"}

def test_total_garbage_falls_back(monkeypatch):
    _stub(monkeypatch, 'not json at all')
    result = classify("b", "s", "c")
    assert result == {
        "product_area": "none",
        "request_type": "product_issue",
        "request_subtype": "product_issue",
        "risk_level": "low",
        "language": "en",
    }
