import json
from config import ROOT

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
    if is_adv:
        return 0.99
    if request_type == "invalid":
        return 1.0
        
    # Check if we have a clean FAQ match
    if not escalated and source_documents:
        return 0.95
        
    if escalated:
        if escalated_by_rules:
            return 0.80
        else:
            return 0.70
            
    # Fallback calculation
    base_score = 0.85
    modifiers = 0.0
    
    if product_area == "none":
        modifiers -= 0.05
    if language != "en":
        modifiers -= 0.05
    if len(ticket_text.split()) < 20:
        modifiers -= 0.15
        
    return max(0.60, min(1.0, base_score + modifiers))

def assemble(
    row: dict,
    is_adv: bool,
    pii_detected: bool,
    classification: dict,
    escalate: bool,
    escalated_by_rules: bool,
    generated: dict,
    ticket_text: str
) -> dict:
    # 1. Map status and request_type
    status = "escalated" if escalate else "replied"
    request_type = classification.get("request_type", "product_issue")
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
        if escalated_by_rules:
            justification = "Escalated by automated compliance rules (e.g. legal terms detected, critical risk, PII with financial request, or vague out-of-scope query)."
        else:
            justification = "Escalated by supervisor assessment because the request exceeds automated capabilities or requires manual verification."
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
        
    # 5. Build output dictionary conforming exactly to output.csv lowercase schema
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
        "risk_level": classification.get("risk_level", "low"),
        "pii_detected": "true" if pii_detected else "false",
        "language": language,
        "actions_taken": actions_str
    }
    
    return output_row
