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

def test_grounded_answer_is_top():
    assert _conf(source_documents="data/x.md") == 0.95

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
    assert score < 0.95

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
    assert out["confidence_score"] == 0.95
