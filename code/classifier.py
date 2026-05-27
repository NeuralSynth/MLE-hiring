import re
import json
from llm import llm, clean_json_response

PRODUCT_AREAS = ("devplatform", "claude", "visa", "none")
RISK_LEVELS = ("low", "medium", "high", "critical")

# Fine-grained request types the LLM may emit, each mapped to one of the four
# coarse values the output schema allows. The fine label is also returned as
# `request_subtype` so later stages (e.g. escalation) can use it if useful.
REQUEST_TYPE_MAP = {
    "product_issue": "product_issue",
    "feature_request": "feature_request",
    "bug": "bug",
    "invalid": "invalid",
    "billing": "product_issue",
    "account": "product_issue",
    "privacy": "product_issue",
    "travel_support": "product_issue",
    "general_support": "product_issue",
    "conversation_management": "product_issue",
    "community": "product_issue",
}

# Common language names -> ISO 639-1, for when the model ignores the "code" ask.
LANGUAGE_NAMES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "chinese": "zh", "mandarin": "zh",
    "hindi": "hi", "japanese": "ja", "korean": "ko", "russian": "ru",
    "arabic": "ar", "dutch": "nl",
}

CLASSIFIER_SYSTEM_PROMPT = """You are a ticket classifier. Output only valid JSON matching this exact schema.
The Company field in the ticket is a hint only — infer product_area from ticket content, not from the Company field.

Schema:
{
  "product_area": "devplatform" | "claude" | "visa" | "none",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid" | "billing" | "account" | "privacy" | "travel_support" | "general_support" | "conversation_management" | "community",
  "risk_level": "low" | "medium" | "high" | "critical",
  "language": "<ISO 639-1 code>"
}

Output only the JSON object. No preamble, no explanation.
"""


def _normalize_product_area(value) -> str:
    area = str(value or "none").strip().lower()
    return area if area in PRODUCT_AREAS else "none"


def _normalize_risk(value) -> str:
    risk = str(value or "low").strip().lower()
    return risk if risk in RISK_LEVELS else "low"


def _coarse_request_type(value) -> str:
    return REQUEST_TYPE_MAP.get(str(value or "").strip().lower(), "product_issue")


def _fine_request_type(value) -> str:
    fine = str(value or "").strip().lower()
    return fine if fine in REQUEST_TYPE_MAP else "product_issue"


def _normalize_language(value) -> str:
    lang = str(value or "").strip().lower()
    if not lang:
        return "en"
    code = re.split(r"[-_]", lang)[0]
    if len(code) == 2 and code.isalpha():
        return code
    return LANGUAGE_NAMES.get(lang, LANGUAGE_NAMES.get(code, "en"))


def classify(ticket_text: str, subject: str, company: str) -> dict:
    """Classify a ticket into product_area, request_type, risk_level, language.

    Each field is normalized independently, so one malformed value cannot
    discard the others. `request_subtype` carries the fine-grained category
    before it is collapsed to the four coarse request types of the output schema.
    """
    user_content = (
        f"Subject: {subject or ''}\n"
        f"Company Hint: {company or ''}\n"
        f"Ticket Body:\n{ticket_text}"
    )

    try:
        result = llm.complete(system=CLASSIFIER_SYSTEM_PROMPT, user=user_content)
        data = json.loads(clean_json_response(result))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    return {
        "product_area": _normalize_product_area(data.get("product_area")),
        "request_type": _coarse_request_type(data.get("request_type")),
        "request_subtype": _fine_request_type(data.get("request_type")),
        "risk_level": _normalize_risk(data.get("risk_level")),
        "language": _normalize_language(data.get("language")),
    }
