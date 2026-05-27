import json
from config import ROOT

# Accurate, reason-specific justifications for escalated tickets (keyed by the
# `reason` returned from should_escalate).
ESCALATION_JUSTIFICATIONS = {
    "critical_risk": "Escalated automatically: the ticket was triaged as critical risk and needs human review.",
    "legal_terms": "Escalated automatically: legal or compliance language was detected, which must be handled by a human specialist.",
    "human_request": "Escalated automatically: the customer explicitly asked to reach a human or supervisor.",
    "pii_financial": "Escalated automatically: personal data combined with a financial request requires manual identity verification.",
    "vague_out_of_scope": "Escalated automatically: the request is too vague or out of scope to resolve from the support documentation.",
    "no_docs": "Escalated automatically: no matching support documentation was found for this request.",
    "weak_retrieval": "Escalated automatically: the available documentation does not sufficiently cover this request.",
    "supervisor_llm": "Escalated by supervisor assessment: the request exceeds automated capabilities or requires manual verification.",
    "generator_unresolved": "Escalated: the request could not be resolved from the available support documentation.",
}
_DEFAULT_ESCALATION = "Escalated by supervisor assessment because the request exceeds automated capabilities or requires manual verification."

def calculate_confidence(
    is_adv: bool,
    escalated: bool,
    escalated_by_rules: bool,
    request_type: str,
    product_area: str,
    language: str,
    source_documents: str,
    ticket_text: str
) -> float:
    # A correct, grounded answer is the most confident outcome.
    if not is_adv and not escalated and request_type != "invalid" and source_documents:
        return 0.95

    # Confident *decisions* that are rejections, not resolutions.
    if is_adv:
        return 0.90
    if request_type == "invalid" and not escalated:
        return 0.90

    # Escalations are a safe hand-off, not a resolution. (Checked before the
    # ungrounded-reply fallback, and after invalid, so an escalated 'invalid'
    # ticket uses the escalation confidence rather than reporting 1.0.)
    if escalated:
        return 0.80 if escalated_by_rules else 0.70

    # Replied without a grounded source — least certain.
    score = 0.70
    if product_area == "none":
        score -= 0.05
    if language != "en":
        score -= 0.05
    if len(ticket_text.split()) < 20:
        score -= 0.15
    return max(0.60, score)

def assemble(
    row: dict,
    is_adv: bool,
    pii_detected: bool,
    classification: dict,
    escalate: bool,
    escalated_by_rules: bool,
    generated: dict,
    ticket_text: str,
    escalation_reason: str = ""
) -> dict:
    # 1. Map status and request_type
    status = "escalated" if escalate else "replied"
    request_type = classification.get("request_type", "product_issue").lower()
    product_area = classification.get("product_area", "none")
    language = classification.get("language", "en")
    
    response = generated.get("response", "")
    actions = generated.get("actions_taken", [])
    source_docs = generated.get("source_documents", "")
    
    # 2. Determine justification
    if is_adv:
        justification = "Adversarial prompt injection attempt detected by the safety screener. Escalated immediately for safety."
        status = "escalated"
        request_type = "invalid"
        response = "This request cannot be processed."
        actions = []
        source_docs = ""
    elif escalate:
        justification = ESCALATION_JUSTIFICATIONS.get(escalation_reason, _DEFAULT_ESCALATION)
    else:
        if source_docs:
            justification = f"Answered from the support corpus using document: {source_docs}."
        else:
            justification = "Answered using general support guidance. No specific documentation matching all parameters was found."

    # 3. Calculate confidence score
    confidence = calculate_confidence(
        is_adv=is_adv,
        escalated=escalate,
        escalated_by_rules=escalated_by_rules,
        request_type=request_type,
        product_area=product_area,
        language=language,
        source_documents=source_docs,
        ticket_text=ticket_text
    )
    
    # 4. Serialize Actions Taken to valid JSON string
    try:
        actions_str = json.dumps(actions)
    except Exception:
        actions_str = "[]"
        
    # 5. Build output dictionary conforming exactly to validator's lowercase snake_case schema
    output_row = {
        "issue": row.get("Issue", ""),
        "subject": row.get("Subject", ""),
        "company": row.get("Company", ""),
        "response": response,
        "product_area": product_area,
        "status": status,
        "request_type": request_type,
        "justification": justification,
        "confidence_score": round(confidence, 2),
        "source_documents": source_docs,
        "risk_level": classification.get("risk_level", "low").lower(),
        "pii_detected": "true" if pii_detected else "false",
        "language": language,
        "actions_taken": actions_str
    }
    
    return output_row
