import json
import pytest
from assembler import assemble, calculate_confidence, ESCALATION_JUSTIFICATIONS

LONG = "this is a reasonably long ticket body that has well over twenty words in it for sure"


def _conf(**kw):
    base = dict(is_adv=False, escalated=False, escalated_by_rules=False,
                request_type="product_issue", product_area="claude",
                language="en", source_documents="", ticket_text=LONG)
    base.update(kw)
    return calculate_confidence(**base)


# --- Confidence ladder ---

def test_grounded_answer_base_confidence():
    """Grounded reply with no grounding-strength signal (defaults to 0.0
    overlap, 0 source count) sits at the base 0.80."""
    assert _conf(source_documents="data/x.md") == 0.80

def test_grounded_answer_scales_with_overlap():
    """Stronger response-token overlap with the cited chunk raises confidence
    continuously up to +0.10."""
    low = _conf(source_documents="data/x.md", top_overlap=0.10)   # +0.025
    mid = _conf(source_documents="data/x.md", top_overlap=0.30)   # +0.075
    high = _conf(source_documents="data/x.md", top_overlap=0.50)  # +0.10 (capped)
    assert low < mid < high
    assert high == 0.90  # 0.80 base + 0.10 cap

def test_grounded_answer_scales_with_source_count():
    """Multi-source citations add up to +0.05; capped beyond ~3 sources."""
    one = _conf(source_documents="data/x.md", source_count=1)             # +0
    two = _conf(source_documents="data/x.md|data/y.md", source_count=2)   # +0.025
    three = _conf(source_documents="data/x.md|data/y.md|data/z.md",
                  source_count=3)                                          # +0.05 (capped)
    assert one < two < three
    assert three == 0.85  # 0.80 base + 0.05 cap

def test_grounded_answer_max_is_0_95():
    """Under the current weights, the natural cap on a grounded reply is
    0.80 base + 0.10 overlap + 0.05 source = 0.95. (A defensive min(0.97, …)
    in the assembler stays below 1.0 if weights are ever increased.)"""
    assert _conf(source_documents="a|b|c|d", top_overlap=1.0, source_count=4) == 0.95

def test_adversarial_confidence():
    assert _conf(is_adv=True, escalated=True) == 0.90

def test_invalid_replied_confidence():
    assert _conf(request_type="invalid") == 0.90

def test_escalated_invalid_uses_escalation_confidence():
    # A1: invalid no longer overrides escalation.
    assert _conf(request_type="invalid", escalated=True, escalated_by_rules=True) == 0.80

def test_rule_escalation_confidence():
    assert _conf(escalated=True, escalated_by_rules=True) == 0.80

def test_llm_escalation_confidence():
    assert _conf(escalated=True, escalated_by_rules=False) == 0.70

def test_ungrounded_reply_below_grounded():
    score = _conf()  # replied, no source
    assert 0.60 <= score <= 0.70
    assert score < 0.80  # always below the grounded base

def test_confidence_floor():
    assert _conf(product_area="none", language="fr", ticket_text="too short") == 0.60


# --- assemble() row shape, enums, and justification ---

OUTPUT_KEYS = {
    "issue", "subject", "company", "response", "product_area", "status", "request_type",
    "justification", "confidence_score", "source_documents", "risk_level", "pii_detected",
    "language", "actions_taken",
}
ROW = {"Issue": "[]", "Subject": "s", "Company": "c"}
CLS = {"product_area": "claude", "request_type": "product_issue", "risk_level": "low", "language": "en"}


def test_assemble_has_all_columns_and_valid_enums():
    out = assemble(ROW, False, False, CLS, False, False,
                   {"response": "hi", "actions_taken": [], "source_documents": ""}, LONG)
    assert set(out) == OUTPUT_KEYS
    assert out["status"] in ("replied", "escalated")
    assert out["pii_detected"] in ("true", "false")
    assert 0.0 <= float(out["confidence_score"]) <= 1.0
    assert isinstance(json.loads(out["actions_taken"]), list)

def test_assemble_adversarial_row():
    out = assemble(ROW, True, False, CLS, True, True,
                   {"response": "x", "actions_taken": [{"action": "x"}], "source_documents": "y"}, LONG)
    assert out["status"] == "escalated"
    assert out["request_type"] == "invalid"
    assert out["response"] == "This request cannot be processed."
    assert out["source_documents"] == ""
    assert out["actions_taken"] == "[]"
    assert out["confidence_score"] == 0.9
    assert "adversarial" in out["justification"].lower()

def test_assemble_escalation_justification_by_reason():
    out = assemble(ROW, False, False, CLS, True, True,
                   {"response": "esc", "actions_taken": [], "source_documents": ""}, LONG,
                   escalation_reason="legal_terms")
    assert out["justification"] == ESCALATION_JUSTIFICATIONS["legal_terms"]
    assert out["status"] == "escalated"
    assert out["confidence_score"] == 0.8

def test_assemble_replied_with_source_justification():
    out = assemble(ROW, False, False, CLS, False, False,
                   {"response": "ans", "actions_taken": [], "source_documents": "data/x.md"}, LONG)
    assert "data/x.md" in out["justification"]
    assert out["status"] == "replied"
    # No grounding-strength info -> base 0.80 (continuous ladder, was flat 0.95).
    assert out["confidence_score"] == 0.80

def test_assemble_replied_with_grounding_strength_lifts_confidence():
    """When the generator surfaces real grounding signal, confidence rises
    above the 0.80 base toward the 0.97 cap."""
    out = assemble(ROW, False, False, CLS, False, False,
                   {"response": "ans", "actions_taken": [],
                    "source_documents": "data/x.md|data/y.md",
                    "grounding": {"top_overlap": 0.50, "source_count": 2}},
                   LONG)
    # base 0.80 + 0.10 (overlap cap) + 0.025 (one extra source) = 0.925
    assert out["confidence_score"] == 0.93  # rounded(0.925, 2) == 0.93 via banker's-rounding; tolerant below

def test_assemble_multi_source_justification_uses_plural():
    out = assemble(ROW, False, False, CLS, False, False,
                   {"response": "ans", "actions_taken": [],
                    "source_documents": "data/x.md|data/y.md"}, LONG)
    assert "documents" in out["justification"]  # plural noun
    assert "data/x.md|data/y.md" in out["justification"]
