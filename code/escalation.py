import re
from config import LEGAL_KEYWORDS, HUMAN_REQUEST_KEYWORDS, FINANCIAL_WORDS
from llm import llm
from retriever import tokenize, format_context

# Weak-retrieval gate: if a substantive ticket's top chunk covers almost none of
# its content words, the documentation isn't really on-topic -> escalate.
MIN_TOKENS_FOR_COVERAGE = 4
WEAK_COVERAGE = 0.15

ESCALATION_SYSTEM_PROMPT = """You are a customer support supervisor. Your job is to decide whether a customer support ticket should be escalated to a human agent, or if it can be resolved automatically using the provided support documentation.

Escalate if:
- The customer is highly frustrated, angry, abusive, or explicitly demands to speak to a supervisor, manager, or human agent.
- The request requires account changes, settings, or database operations that are not described or permitted in the help documents.
- The ticket refers to a suspected bug, system failure, service outage, or technical error that cannot be resolved by standard customer troubleshooting.
- The request is out-of-scope or there is no clear documentation to resolve the issue.

You are also given a triage risk level. Treat a "high" or "critical" risk level as a strong reason to escalate unless the documentation clearly and completely resolves the issue.
You are also given the request type; treat sensitive types such as "privacy" or "billing" with extra caution before resolving automatically.

Otherwise, if the provided help documents cover the exact issue and can resolve it, output "reply".

Output only one word: escalate or reply.
"""


def _compile(words: list[str]) -> re.Pattern:
    """Whole-word (or whole-phrase) alternation, so 'sue' matches 'sue' but not
    'issue', and 'fee' matches 'fee' but not 'feedback'."""
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


_LEGAL_RE = _compile(LEGAL_KEYWORDS)
_HUMAN_RE = _compile(HUMAN_REQUEST_KEYWORDS)
_FINANCIAL_RE = _compile(FINANCIAL_WORDS)


def _llm_escalation_check(ticket_text: str, retrieved_chunks: list, risk_level: str,
                          pii_detected: bool, request_subtype: str = "") -> bool:
    """Ask the supervisor LLM to decide on tickets that clear every rule."""
    doc_context = format_context(retrieved_chunks)
    user_content = (
        f"Triage risk level: {risk_level}\n"
        f"Request type: {request_subtype or 'unknown'}\n"
        f"PII present: {pii_detected}\n\n"
        f"Retrieved Support Documentation:\n{doc_context}\n\n"
        f"Customer Ticket:\n{ticket_text}"
    )
    result = llm.complete(system=ESCALATION_SYSTEM_PROMPT, user=user_content)
    # First word only — so "reply, no need to escalate" is not read as escalate.
    return result.strip().lower().startswith("escalate")


def should_escalate(
    risk_level: str,
    pii_detected: bool,
    ticket_text: str,
    product_area: str,
    retrieved_chunks: list,
    request_subtype: str = "",
) -> tuple[bool, bool, str]:
    """Decide whether a ticket should be escalated.

    Returns (escalate, escalated_by_rules, reason). Deterministic compliance and
    safety rules run first and short-circuit; the LLM supervisor only judges
    tickets that clear every rule. `reason` drives the output justification and
    is "" when the ticket is not escalated.
    """
    # Rule 1: critical risk level.
    if risk_level == "critical":
        return True, True, "critical_risk"

    # Rule 2a: legal / compliance language.
    if _LEGAL_RE.search(ticket_text):
        return True, True, "legal_terms"

    # Rule 2b: explicit request to reach a human.
    if _HUMAN_RE.search(ticket_text):
        return True, True, "human_request"

    # Rule 3: PII detected together with a financial action.
    if pii_detected and _FINANCIAL_RE.search(ticket_text):
        return True, True, "pii_financial"

    # Rule 4: vague, out-of-scope request with no product area.
    if product_area == "none" and len(ticket_text.split()) < 20:
        return True, True, "vague_out_of_scope"

    # Rule 5: no corpus documents matched at all.
    if not retrieved_chunks:
        return True, True, "no_docs"

    # Rule 6: weak retrieval — the top chunk barely overlaps a substantive ticket.
    ticket_tokens = set(tokenize(ticket_text))
    if len(ticket_tokens) >= MIN_TOKENS_FOR_COVERAGE:
        top_tokens = set(tokenize(retrieved_chunks[0]["content"]))
        if len(ticket_tokens & top_tokens) / len(ticket_tokens) < WEAK_COVERAGE:
            return True, True, "weak_retrieval"

    # Otherwise, let the LLM supervisor judge the genuinely ambiguous remainder.
    if _llm_escalation_check(ticket_text, retrieved_chunks, risk_level, pii_detected, request_subtype):
        return True, False, "supervisor_llm"
    return False, False, ""
