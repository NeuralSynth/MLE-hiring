from config import ESCALATION_KEYWORDS
from llm import llm

ESCALATION_SYSTEM_PROMPT = """You are a customer support supervisor. Your job is to decide whether a customer support ticket should be escalated to a human agent, or if it can be resolved automatically using the provided support documentation.

Escalate if:
- The customer is highly frustrated, angry, abusive, or explicitly demands to speak to a supervisor, manager, or human agent.
- The request requires account changes, settings, or database operations that are not described or permitted in the help documents.
- The ticket refers to a suspected bug, system failure, service outage, or technical error that cannot be resolved by standard customer troubleshooting.
- The request is out-of-scope or there is no clear documentation to resolve the issue.

Otherwise, if the provided help documents cover the exact issue and can resolve it, output "reply".

Output only one word: escalate or reply.
"""

def _llm_escalation_check(ticket_text: str, retrieved_chunks: list) -> bool:
    if not retrieved_chunks:
        return True
        
    doc_context = "\n\n".join([f"Document {i+1} (Path: {doc['path']}):\n{doc['content']}" for i, doc in enumerate(retrieved_chunks)])
    user_content = f"Retrieved Support Documentation:\n{doc_context}\n\nCustomer Ticket:\n{ticket_text}"
    
    result = llm.complete(
        system=ESCALATION_SYSTEM_PROMPT,
        user=user_content
    )
    
    return "escalate" in result.strip().lower()

def should_escalate(
    risk_level: str,
    pii_detected: bool,
    ticket_text: str,
    product_area: str,
    retrieved_chunks: list
) -> tuple[bool, bool]:
    text_lower = ticket_text.lower()

    # Rule 1: critical risk level
    if risk_level == "critical":
        return True, True

    # Rule 2: legal keywords
    if any(kw in text_lower for kw in ESCALATION_KEYWORDS):
        return True, True

    # Rule 3: PII detected and a financial action word is present
    financial_words = ["refund", "charge", "transaction", "payment", "transfer", "purchase", "billing"]
    if pii_detected and any(w in text_lower for w in financial_words):
        return True, True

    # Rule 4: vague report with no company specified
    if product_area == "none" and len(ticket_text.split()) < 20:
        return True, True

    # Rule 5: no corpus documents matched at all
    if not retrieved_chunks:
        return True, True

    # If no hardcoded rules match, use the LLM check for ambiguous cases
    is_escalated = _llm_escalation_check(ticket_text, retrieved_chunks)
    return is_escalated, False
