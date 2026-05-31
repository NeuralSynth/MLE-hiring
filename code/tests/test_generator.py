import json
import pytest
from generator import generate, rule_escalation, _RULE_ESCALATION_ROUTING

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

def test_generator_returns_actions_taken_as_list():
    """Unit check: generate() returns actions_taken as a list (parseable as
    JSON either way). Distinct from the pipeline-level same-name check in
    test_pipeline.py — see also: that test asserts every row across the
    end-to-end run is JSON-valid."""
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


# --- G2b: deterministic citation backfill when the LLM omits / mangles the path ---

# Two real chunks whose paths exist on disk, used to drive the backfill picker.
G2B_CHUNKS = [
    {
        "path": "data/visa/support/consumer/travel-support/exchange-rate-calculator.md",
        "content": "The exchange rate calculator lets you convert amounts between currencies for travel planning.",
        "score": 0.92,
    },
    {
        "path": "data/devplatform/settings/account-settings.md",
        "content": "To delete your account, go to Settings then Account then Delete Account.",
        "score": 0.55,
    },
]


def test_g2b_backfills_when_llm_omits_citation_and_response_overlaps_chunk(monkeypatch):
    """LLM emits empty source but the response is measurably grounded in one of
    the chunks — backfill attributes that chunk's path."""
    _stub(monkeypatch,
          '{"response":"Use the exchange rate calculator to convert currencies for travel planning.",'
          '"actions_taken":[],"source_documents":""}')
    out = generate("how do I convert money", G2B_CHUNKS, False)
    assert out["source_documents"] == "data/visa/support/consumer/travel-support/exchange-rate-calculator.md"


def test_g2b_stays_empty_when_response_does_not_overlap(monkeypatch):
    """Generic / templated response with no content overlap → no false attribution."""
    _stub(monkeypatch, '{"response":"Thanks for reaching out, we appreciate your patience.","actions_taken":[],"source_documents":""}')
    out = generate("any question", G2B_CHUNKS, False)
    assert out["source_documents"] == ""


def test_g2b_does_not_override_valid_llm_citation(monkeypatch):
    """LLM cited a valid path. Even if another chunk overlaps the response more,
    the LLM's choice is preserved (G2b only fires when source_doc is empty)."""
    _stub(monkeypatch,
          '{"response":"To delete your account go to settings then account then delete account.",'
          '"actions_taken":[],"source_documents":"data/devplatform/settings/account-settings.md"}')
    out = generate("delete my account", G2B_CHUNKS, False)
    assert out["source_documents"] == "data/devplatform/settings/account-settings.md"


def test_g2b_never_fires_on_escalate_path(monkeypatch):
    """Escalate path: source_documents must stay empty even if a chunk overlaps the response."""
    _stub(monkeypatch,
          '{"response":"Escalating to a human; exchange rate calculator info is not applicable here.",'
          '"actions_taken":[],"source_documents":""}')
    out = generate("anything", G2B_CHUNKS, True)
    assert out["source_documents"] == ""


def test_g2b_does_not_fire_after_g6_flip(monkeypatch):
    """G6: ungrounded 'I cannot answer; please escalate' reply that happens to
    share tokens with a chunk. G6 flips to escalated and G2b respects that —
    source stays empty."""
    _stub(monkeypatch,
          '{"response":"I cannot answer this question about the exchange rate calculator from the docs; please escalate.",'
          '"actions_taken":[],"source_documents":""}')
    out = generate("how do I convert money", G2B_CHUNKS, False)
    assert out["escalated"] is True
    assert out["source_documents"] == ""


def test_g2b_uses_retriever_score_as_tiebreaker(monkeypatch):
    """Two chunks with similar overlap: BOTH are attributed (multi-source per
    the schema), but the higher-scored chunk comes FIRST in the pipe-separated
    output."""
    chunks = [
        {"path": "data/visa/support/consumer/travel-support/exchange-rate-calculator.md",
         "content": "alpha beta gamma delta epsilon zeta eta theta", "score": 0.30},
        {"path": "data/devplatform/settings/account-settings.md",
         "content": "alpha beta gamma delta epsilon zeta eta theta", "score": 0.90},
    ]
    _stub(monkeypatch,
          '{"response":"alpha beta gamma delta epsilon zeta eta theta","actions_taken":[],"source_documents":""}')
    out = generate("q", chunks, False)
    # Same overlap (1.0 both) — both attributed; higher-scored chunk first.
    assert out["source_documents"] == (
        "data/devplatform/settings/account-settings.md"
        "|data/visa/support/consumer/travel-support/exchange-rate-calculator.md"
    )


def test_g2b_returns_empty_for_short_response(monkeypatch):
    """Responses with fewer than 3 content tokens are too short to attribute."""
    _stub(monkeypatch, '{"response":"ok","actions_taken":[],"source_documents":""}')
    out = generate("q", G2B_CHUNKS, False)
    assert out["source_documents"] == ""


# --- Gap A: pipe-separated multi-source attribution ---

def test_g2_keeps_multiple_valid_pipe_separated_paths(monkeypatch):
    """LLM cited two valid paths pipe-separated — both must be preserved. G2
    checks each path exists on disk, so this test uses two real corpus files."""
    chunks = [
        {"path": "data/visa/support/consumer/travel-support/exchange-rate-calculator.md",
         "content": "x", "score": 0.5},
        {"path": "data/claude/amazon-bedrock/7996918-what-is-amazon-bedrock.md",
         "content": "y", "score": 0.4},
    ]
    _stub(monkeypatch,
          '{"response":"see both docs","actions_taken":[],'
          '"source_documents":"data/visa/support/consumer/travel-support/exchange-rate-calculator.md|data/claude/amazon-bedrock/7996918-what-is-amazon-bedrock.md"}')
    out = generate("q", chunks, False)
    assert out["source_documents"] == (
        "data/visa/support/consumer/travel-support/exchange-rate-calculator.md"
        "|data/claude/amazon-bedrock/7996918-what-is-amazon-bedrock.md"
    )


def test_g2_drops_invalid_paths_keeps_valid_in_pipe_separated(monkeypatch):
    """LLM cited two paths, one invalid: only the valid one survives, the
    invalid one is dropped without collapsing the whole field to empty."""
    chunks = [
        {"path": "data/visa/support/consumer/travel-support/exchange-rate-calculator.md",
         "content": "x", "score": 0.5},
    ]
    _stub(monkeypatch,
          '{"response":"see the calculator","actions_taken":[],'
          '"source_documents":"README.md|data/visa/support/consumer/travel-support/exchange-rate-calculator.md"}')
    out = generate("q", chunks, False)
    assert out["source_documents"] == "data/visa/support/consumer/travel-support/exchange-rate-calculator.md"


def test_g2_collapses_to_empty_when_all_paths_invalid(monkeypatch):
    """Every cited path invalid → field collapses to "" → G2b may then backfill
    or stay empty. Here the response doesn't overlap, so it stays empty."""
    chunks = [
        {"path": "data/visa/support/consumer/travel-support/exchange-rate-calculator.md",
         "content": "exchange rate calculator", "score": 0.5},
    ]
    _stub(monkeypatch,
          '{"response":"general thanks for your patience","actions_taken":[],'
          '"source_documents":"README.md|nonexistent.md"}')
    out = generate("q", chunks, False)
    assert out["source_documents"] == ""


def test_g2b_backfills_all_overlapping_chunks_pipe_separated(monkeypatch):
    """LLM emits empty, response overlaps TWO chunks — both should be attributed
    pipe-separated, ranked by combined score."""
    chunks = [
        {"path": "data/visa/support/consumer/travel-support/exchange-rate-calculator.md",
         "content": "exchange rate calculator currencies travel", "score": 0.40},
        {"path": "data/devplatform/settings/account-settings.md",
         "content": "delete account settings exchange currencies", "score": 0.80},
    ]
    # Response uses tokens from BOTH chunks.
    _stub(monkeypatch,
          '{"response":"Use the exchange rate calculator and settings to manage currencies for travel.",'
          '"actions_taken":[],"source_documents":""}')
    out = generate("q", chunks, False)
    paths = out["source_documents"].split("|")
    assert len(paths) == 2
    # Both real corpus paths in the output (order is by combined score).
    assert "data/visa/support/consumer/travel-support/exchange-rate-calculator.md" in paths
    assert "data/devplatform/settings/account-settings.md" in paths


def test_g2b_caps_attribution_at_three_chunks(monkeypatch):
    """If more than _MAX_BACKFILL_CHUNKS chunks overlap, only the top 3 are kept."""
    # All four chunks share the exact response tokens, so each has overlap=1.0;
    # ranking falls to retriever_score; only the top 3 should be attributed.
    chunks = [
        {"path": "data/visa/support/consumer/travel-support/exchange-rate-calculator.md",
         "content": "alpha beta gamma delta", "score": 0.90},
        {"path": "data/devplatform/settings/account-settings.md",
         "content": "alpha beta gamma delta", "score": 0.80},
        {"path": "data/claude/amazon-bedrock/7996918-what-is-amazon-bedrock.md",
         "content": "alpha beta gamma delta", "score": 0.70},
        {"path": "data/claude/claude-api-and-console/troubleshooting/8114490-where-can-i-find-your-api-documentation.md",
         "content": "alpha beta gamma delta", "score": 0.60},
    ]
    _stub(monkeypatch,
          '{"response":"alpha beta gamma delta","actions_taken":[],"source_documents":""}')
    out = generate("q", chunks, False)
    paths = out["source_documents"].split("|")
    assert len(paths) == 3  # capped at _MAX_BACKFILL_CHUNKS
    # The lowest-scored chunk (0.60) must NOT appear.
    assert "data/claude/claude-api-and-console/troubleshooting/8114490-where-can-i-find-your-api-documentation.md" not in paths


# --- 4a: deterministic rule-based escalation (no LLM call) ---

# Mirrors the escalate_to_human schema in data/api_specs/internal_tools.json.
_VALID_DEPARTMENTS = {"billing", "technical", "security", "legal", "general"}
_VALID_PRIORITIES = {"low", "normal", "high", "urgent"}


def test_rule_escalation_shape_and_schema():
    """rule_escalation returns the generate() shape with a schema-valid
    escalate_to_human action, empty sources, and escalated=True."""
    out = rule_escalation("legal_terms")
    assert set(out) == {"response", "actions_taken", "source_documents", "escalated", "grounding"}
    assert out["escalated"] is True
    assert out["source_documents"] == ""
    assert out["response"]  # non-empty message
    actions = out["actions_taken"]
    assert len(actions) == 1 and actions[0]["action"] == "escalate_to_human"
    params = actions[0]["parameters"]
    assert set(params) >= {"priority", "department", "summary"}
    assert params["department"] in _VALID_DEPARTMENTS
    assert params["priority"] in _VALID_PRIORITIES
    assert params["summary"]


@pytest.mark.parametrize("reason", list(_RULE_ESCALATION_ROUTING))
def test_rule_escalation_routing_is_schema_valid(reason):
    """Every routed reason maps to enum-valid department/priority."""
    params = rule_escalation(reason)["actions_taken"][0]["parameters"]
    assert params["department"] in _VALID_DEPARTMENTS
    assert params["priority"] in _VALID_PRIORITIES


def test_rule_escalation_specific_routes():
    """Spot-check the reasons whose routing carries real meaning."""
    assert rule_escalation("legal_terms")["actions_taken"][0]["parameters"]["department"] == "legal"
    assert rule_escalation("critical_risk")["actions_taken"][0]["parameters"]["priority"] == "urgent"
    assert rule_escalation("pii_financial")["actions_taken"][0]["parameters"]["department"] == "billing"


def test_rule_escalation_unknown_reason_falls_back():
    """An unmapped reason still yields a valid, neutral escalation."""
    params = rule_escalation("some_future_reason")["actions_taken"][0]["parameters"]
    assert params["department"] == "general"
    assert params["priority"] == "normal"


def test_rule_escalation_makes_no_llm_call(monkeypatch):
    """The whole point of 4a: no LLM is invoked. Stub complete() to explode."""
    from llm import llm
    def _boom(system, user):
        raise AssertionError("rule_escalation must not call the LLM")
    monkeypatch.setattr(llm, "complete", _boom)
    out = rule_escalation("no_docs")  # must not raise
    assert out["escalated"] is True
