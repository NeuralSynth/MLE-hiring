import json
import pytest
from generator import generate

SAMPLE_CHUNKS = [
    {
        "path": "data/devplatform/settings/account-settings.md",
        "content": "To delete your account, go to Settings > Account > Delete Account.",
        "score": 3.2
    }
]

# A real corpus file, used where the source-document path must exist on disk.
REAL_CHUNKS = [
    {"path": "data/visa/support/consumer/travel-support/exchange-rate-calculator.md", "content": "exchange rate calculator info", "score": 5.0}
]


def _stub(monkeypatch, response):
    from llm import llm
    monkeypatch.setattr(llm, "complete", lambda system, user: response)


# --- Existing behaviour (uses the session mock) ---

def test_generator_returns_required_keys():
    result = generate(ticket_text="How do I delete my account?", chunks=SAMPLE_CHUNKS, escalate=False)
    assert "response" in result
    assert "actions_taken" in result
    assert "source_documents" in result

def test_actions_taken_is_valid_json():
    result = generate(ticket_text="How do I delete my account?", chunks=SAMPLE_CHUNKS, escalate=False)
    parsed = json.loads(result["actions_taken"]) if isinstance(result["actions_taken"], str) else result["actions_taken"]
    assert isinstance(parsed, list)

def test_escalated_response_has_escalate_action():
    """Escalated tickets MUST have escalate_to_human in actions_taken."""
    result = generate(ticket_text="I want to sue your company", chunks=[], escalate=True)
    actions = result["actions_taken"]
    if isinstance(actions, str):
        actions = json.loads(actions)
    assert isinstance(actions, list)
    assert len(actions) > 0, "Escalated tickets must have at least one action"
    assert "escalate_to_human" in [a.get("action") for a in actions]

def test_response_does_not_echo_pii():
    result = generate(ticket_text="My email is secret@private.com, help me with my account", chunks=SAMPLE_CHUNKS, escalate=False)
    assert "secret@private.com" not in result["response"]

def test_source_documents_path_exists():
    from pathlib import Path
    result = generate(ticket_text="How do I delete my account?", chunks=SAMPLE_CHUNKS, escalate=False)
    src = result["source_documents"]
    if src:
        assert Path(src).exists(), f"Source document path does not exist: {src}"


# --- G1: actions validated against the schema ---

def test_invalid_tool_calls_are_dropped(monkeypatch):
    _stub(monkeypatch, '{"response":"ok","actions_taken":[{"action":"make_coffee","parameters":{"x":1}},{"action":"reset_password","parameters":{}}],"source_documents":""}')
    out = generate("q", REAL_CHUNKS, False)
    assert out["actions_taken"] == []  # unknown tool + reset_password missing required user_email

def test_valid_tool_call_kept(monkeypatch):
    # verify_identity is non-destructive, so it isolates G1 (no enforcement step added).
    _stub(monkeypatch, '{"response":"ok","actions_taken":[{"action":"verify_identity","parameters":{"method":"email_otp","target":"[EMAIL]"}}],"source_documents":""}')
    out = generate("q", REAL_CHUNKS, False)
    assert [a["action"] for a in out["actions_taken"]] == ["verify_identity"]


# --- G2: source_documents must be a retrieved document ---

def test_source_documents_must_be_retrieved(monkeypatch):
    _stub(monkeypatch, '{"response":"ok","actions_taken":[],"source_documents":"README.md"}')
    out = generate("q", REAL_CHUNKS, False)
    assert out["source_documents"] == ""

def test_retrieved_source_document_kept(monkeypatch):
    _stub(monkeypatch, '{"response":"see the calculator","actions_taken":[],"source_documents":"data/visa/support/consumer/travel-support/exchange-rate-calculator.md"}')
    out = generate("q", REAL_CHUNKS, False)
    assert out["source_documents"] == "data/visa/support/consumer/travel-support/exchange-rate-calculator.md"


# --- G3: escalate path always carries escalate_to_human ---

def test_escalate_injects_action_if_missing(monkeypatch):
    _stub(monkeypatch, '{"response":"escalating you now","actions_taken":[],"source_documents":""}')
    out = generate("q", [], True)
    assert out["escalated"] is True
    assert "escalate_to_human" in [a["action"] for a in out["actions_taken"]]


# --- G5: empty response is backfilled ---

def test_empty_response_backfilled(monkeypatch):
    _stub(monkeypatch, '{"response":"   ","actions_taken":[],"source_documents":""}')
    out = generate("q", REAL_CHUNKS, False)
    assert out["response"].strip() != ""


# --- G6: ungrounded "cannot resolve / escalate" reply flips to escalation ---

def test_unresolved_reply_flips_to_escalated(monkeypatch):
    _stub(monkeypatch, '{"response":"I cannot answer this from the docs; please escalate to a human.","actions_taken":[],"source_documents":""}')
    out = generate("q", REAL_CHUNKS, False)
    assert out["escalated"] is True
    assert "escalate_to_human" in [a["action"] for a in out["actions_taken"]]

def test_grounded_reply_not_flipped(monkeypatch):
    _stub(monkeypatch, '{"response":"To check your rate, open the calculator; contact support if needed.","actions_taken":[],"source_documents":"data/visa/support/consumer/travel-support/exchange-rate-calculator.md"}')
    out = generate("q", REAL_CHUNKS, False)
    assert out["escalated"] is False


# --- verify_identity is enforced before destructive actions ---

def test_destructive_action_gets_identity_verification(monkeypatch):
    _stub(monkeypatch, '{"response":"refunding","actions_taken":[{"action":"issue_refund","parameters":{"transaction_id":"t1","amount":10,"reason":"duplicate"}}],"source_documents":""}')
    names = [a["action"] for a in generate("refund my last charge", REAL_CHUNKS, False)["actions_taken"]]
    assert names[0] == "verify_identity"
    assert "issue_refund" in names

def test_verify_identity_not_duplicated(monkeypatch):
    _stub(monkeypatch, '{"response":"ok","actions_taken":[{"action":"verify_identity","parameters":{"method":"email_otp","target":"x"}},{"action":"reset_password","parameters":{"user_email":"[EMAIL]"}}],"source_documents":""}')
    names = [a["action"] for a in generate("reset my password", REAL_CHUNKS, False)["actions_taken"]]
    assert names.count("verify_identity") == 1
