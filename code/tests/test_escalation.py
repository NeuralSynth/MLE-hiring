import pytest
from escalation import should_escalate, _LEGAL_RE, _FINANCIAL_RE

CHUNK = [{"content": "some doc", "path": "x.md", "score": 2.0}]


def test_critical_risk_always_escalates():
    esc, by_rules, reason = should_escalate("critical", False, "normal support question", "devplatform", CHUNK)
    assert esc is True and by_rules is True and reason == "critical_risk"

def test_legal_keyword_always_escalates():
    esc, by_rules, reason = should_escalate("low", False, "I am going to file a lawsuit against your company", "visa", CHUNK)
    assert esc is True and by_rules is True and reason == "legal_terms"

def test_human_request_escalates():
    esc, by_rules, reason = should_escalate("low", False, "This is useless, let me speak to a human", "visa", CHUNK)
    assert esc is True and by_rules is True and reason == "human_request"

def test_pii_plus_financial_escalates():
    esc, by_rules, reason = should_escalate("medium", True, "my card was charged incorrectly, I want a refund", "visa", CHUNK)
    assert esc is True and by_rules is True and reason == "pii_financial"

def test_no_corpus_match_escalates():
    esc, by_rules, reason = should_escalate("low", False, "something completely unrelated", "devplatform", [])
    assert esc is True and by_rules is True and reason == "no_docs"

def test_clean_faq_does_not_escalate():
    esc, by_rules, reason = should_escalate(
        "low", False, "How do I reset my password?", "devplatform",
        [{"content": "to reset your password go to account settings", "path": "x.md", "score": 2.5}])
    assert esc is False and by_rules is False


# --- B1: whole-word matching, no substring false positives ---

def test_legal_regex_ignores_substrings():
    assert not _LEGAL_RE.search("I have an issue with my account")
    assert not _LEGAL_RE.search("your courteous staff were helpful")
    assert not _LEGAL_RE.search("I would like to pursue a discount")
    assert _LEGAL_RE.search("I will file a lawsuit")
    assert _LEGAL_RE.search("this looks like fraud")

def test_financial_regex_ignores_feedback():
    assert not _FINANCIAL_RE.search("here is my feedback about the app")
    assert _FINANCIAL_RE.search("I was charged twice")

def test_issue_ticket_not_escalated_as_legal(monkeypatch):
    from llm import llm
    monkeypatch.setattr(llm, "complete", lambda system, user: "reply")
    esc, by_rules, reason = should_escalate(
        "low", False, "I have an issue logging in to my dashboard today",
        "devplatform", [{"content": "how to fix an issue logging in to your dashboard", "path": "x.md", "score": 2.0}])
    assert reason != "legal_terms"


# --- B2: supervisor verdict parsed by first word ---

def test_llm_verdict_uses_first_word(monkeypatch):
    from llm import llm
    monkeypatch.setattr(llm, "complete", lambda system, user: "Reply — there is no need to escalate this.")
    esc, by_rules, reason = should_escalate(
        "low", False, "how do I change my avatar image please", "claude",
        [{"content": "change your avatar in settings", "path": "x.md", "score": 2.0}])
    assert esc is False


# --- M1: high risk is deferred to the supervisor, with risk passed in ---

def test_high_risk_defers_to_supervisor(monkeypatch):
    from llm import llm
    captured = {}
    def fake(system, user):
        captured["user"] = user
        return "escalate"
    monkeypatch.setattr(llm, "complete", fake)
    esc, by_rules, reason = should_escalate(
        "high", False, "how do I change my avatar image please", "claude",
        [{"content": "change your avatar in settings", "path": "x.md", "score": 2.0}])
    assert esc is True and by_rules is False and reason == "supervisor_llm"
    assert "high" in captured["user"].lower()


# --- M3: weak retrieval escalates by rule ---

def test_weak_retrieval_escalates(monkeypatch):
    from llm import llm
    monkeypatch.setattr(llm, "complete", lambda system, user: "reply")
    esc, by_rules, reason = should_escalate(
        "low", False, "configure single sign on saml provider okta integration",
        "devplatform", [{"content": "how to bake a chocolate cake recipe", "path": "x.md", "score": 0.5}])
    assert esc is True and by_rules is True and reason == "weak_retrieval"


# --- request_subtype is surfaced to the supervisor LLM ---

def test_request_subtype_passed_to_supervisor(monkeypatch):
    from llm import llm
    captured = {}
    def fake(system, user):
        captured["user"] = user
        return "reply"
    monkeypatch.setattr(llm, "complete", fake)
    should_escalate("low", False, "how do I change my avatar image please", "claude",
                    [{"content": "change your avatar in settings", "path": "x.md", "score": 2.0}],
                    request_subtype="privacy")
    assert "privacy" in captured["user"].lower()


# --- Supervisor LLM defaults to REPLY when the docs cover the topic (anti-over-escalation) ---

def test_supervisor_prompt_defaults_to_reply():
    """Lock in the bias so the prompt can't silently drift back."""
    from escalation import ESCALATION_SYSTEM_PROMPT
    assert "Default to REPLY" in ESCALATION_SYSTEM_PROMPT

def test_supervisor_replies_when_docs_cover_a_how_to():
    """An answerable how-to with a matching doc -> reply, not escalate."""
    chunks = [{"path": "data/devplatform/rescheduling-an-interview.md",
               "content": ("Rescheduling an interview: DevPlatform candidates can request a reschedule of their "
                           "assessment from the dashboard. Reschedules due to unforeseen circumstances are typically approved."),
               "score": 0.85}]
    esc, by_rules, reason = should_escalate(
        "low", False,
        "I would like to request a rescheduling of my DevPlatform assessment due to unforeseen circumstances",
        "devplatform", chunks)
    assert esc is False and reason == ""

def test_supervisor_replies_even_when_risk_is_high_if_docs_cover_it():
    """High risk alone is not a standalone escalation trigger when the docs answer it."""
    chunks = [{"path": "data/claude/data-retention.md",
               "content": "Claude data retention: data used to improve models is retained for a limited period and can be opted out.",
               "score": 0.87}]
    esc, by_rules, reason = should_escalate(
        "high", False,
        "I am allowing Claude to use my data to improve the models, how long will the data be used for",
        "claude", chunks)
    assert esc is False

def test_supervisor_escalates_a_confirmed_outage(monkeypatch):
    """A confirmed service outage is one of the four explicit escalate criteria."""
    from llm import llm
    monkeypatch.setattr(llm, "complete", lambda system, user: "escalate")
    chunks = [{"path": "x.md",
               "content": "Claude API requests sometimes fail intermittently; if all requests are failing please contact support.",
               "score": 0.5}]
    esc, by_rules, reason = should_escalate(
        "high", False, "Claude has stopped working completely, all requests are failing",
        "claude", chunks)
    assert esc is True and by_rules is False and reason == "supervisor_llm"
