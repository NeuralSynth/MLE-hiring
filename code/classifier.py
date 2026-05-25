import json
from llm import llm, clean_json_response

CLASSIFIER_SYSTEM_PROMPT = """You are a ticket classifier. Output only valid JSON matching this exact schema.
The Company field in the ticket is a hint only — infer product_area from ticket content, not from the Company field.

Schema:
{
  "product_area": "devplatform" | "claude" | "visa" | "none",
  "request_type": "product_issue" | "bug" | "invalid" | "billing" | "account" | "privacy" | "travel_support" | "general_support" | "conversation_management" | "community",
  "risk_level": "low" | "medium" | "high" | "critical",
  "language": "<ISO 639-1 code>"
}

Output only the JSON object. No preamble, no explanation.
"""

def classify(ticket_text: str, subject: str, company: str) -> dict:
    user_content = f"Subject: {subject}\nCompany Hint: {company}\nTicket Body:\n{ticket_text}"
    
    result = llm.complete(
        system=CLASSIFIER_SYSTEM_PROMPT,
        user=user_content
    )
    
    try:
        cleaned = clean_json_response(result)
        classification = json.loads(cleaned)
        
        # Ensure fallback defaults if LLM outputs invalid values
        product_area = classification.get("product_area", "none").strip().lower()
        if product_area not in ["devplatform", "claude", "visa", "none"]:
            classification["product_area"] = "none"
        else:
            classification["product_area"] = product_area
            
        request_type = classification.get("request_type", "product_issue").strip().lower()
        mapping = {
            "product_issue": "product_issue",
            "billing": "product_issue",
            "account": "product_issue",
            "general_support": "product_issue",
            "community": "product_issue",
            "travel_support": "product_issue",
            "privacy": "product_issue",
            "conversation_management": "product_issue",
            "bug": "bug",
            "invalid": "invalid",
            "feature_request": "feature_request"
        }
        classification["request_type"] = mapping.get(request_type, "product_issue")
        
        risk_level = classification.get("risk_level", "low").strip().lower()
        if risk_level not in ["low", "medium", "high", "critical"]:
            classification["risk_level"] = "low"
        else:
            classification["risk_level"] = risk_level
            
        language = classification.get("language", "en").strip().lower()
        classification["language"] = language
        
        return classification
    except Exception:
        # Robust fallback classification
        return {
            "product_area": "none",
            "request_type": "product_issue",
            "risk_level": "low",
            "language": "en"
        }
