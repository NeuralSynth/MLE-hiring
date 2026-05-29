import re

# PII regex patterns

# Street-address building blocks. Matching is intentionally case-sensitive: a
# real address capitalizes the street name and type ("123 Main St"), so
# requiring a capitalized name token right after the house number prevents false
# positives like "3 items in St Louis" or "2 days at Dr Smith". The bounded
# {1,4} repetition also removes the backtracking risk of the old open-ended filler.
_STREET_TYPES = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Highway|Hwy|Boulevard|Blvd|Drive|Dr|"
    r"Lane|Ln|Court|Ct|Circle|Cir|Trail|Trl|Way|Plaza|Plz|Square|Sq|"
    r"Terrace|Ter|Suite|Apt)"
)
_STREET_NAME = r"(?:[A-Z][a-zA-Z'.-]*|\d{1,3}(?:st|nd|rd|th))"

PATTERNS = {
    "email": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Phone: 3-3-4 with mandatory separators (parens / country code also accepted),
    # so a bare 10-digit run (order or tracking IDs) is not treated as a phone.
    "phone": re.compile(r"\b(?:\+\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    # General credit card format
    "credit_card_raw": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    # Address: <house number> <capitalized street name(s)> <street type>
    "address_street": re.compile(
        r"\b\d{1,6}\s+(?:" + _STREET_NAME + r"\s+){1,4}" + _STREET_TYPES + r"\b\.?"
    ),
    # City, State ZIP (e.g. Springfield, IL 62704)
    "address_zip": re.compile(r"\b[A-Za-z\s\-.']+,\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"),
    # IPv4 with strict per-octet 0–255 validation, so "1.2.3.999" doesn't match.
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    ),
    # Undashed SSN ONLY when a context word ("SSN" / "social security") is
    # within 15 chars of the 9-digit run. Avoids the false positives a bare
    # 9-digit run would create on order IDs, tracking numbers, etc.
    # The bounded lazy filler `.{0,15}?` accepts short intervening text like
    # " is ", ": ", " number ", " was " between the context word and digits.
    "ssn_with_context": re.compile(
        r"\b(?:SSN|social\s*security(?:\s*number)?)\b.{0,15}?\b\d{9}\b",
        re.IGNORECASE
    ),
    # API keys / secret tokens. Format: sk- or pk- prefix + 20+ chars, which
    # covers OpenAI (sk-...), Anthropic (sk-ant-...), Stripe (sk_live_... /
    # pk_live_...). 20-char minimum prevents matching ordinary text starting
    # with "sk-" or "pk-".
    "api_key": re.compile(r"\b(?:sk|pk)[_-][A-Za-z0-9_-]{20,}\b"),
}

def check_luhn(card_str: str) -> bool:
    digits = [int(d) for d in card_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reversed_digits = digits[::-1]
    for i, digit in enumerate(reversed_digits):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _mask_card(match: re.Match) -> str:
    """Mask a candidate card, keeping the last 4 digits — but only if it passes
    Luhn (otherwise it is not a real card, so leave it untouched)."""
    if check_luhn(match.group(0)):
        last4 = "".join(c for c in match.group(0) if c.isdigit())[-4:]
        return f"[CARD ****{last4}]"
    return match.group(0)


def _mask_phone(match: re.Match) -> str:
    digits = "".join(c for c in match.group(0) if c.isdigit())
    return f"[PHONE ...{digits[-4:]}]" if len(digits) >= 4 else "[PHONE]"


def _scrub(text: str) -> str:
    """Replace every detected PII span with a placeholder. Cards are masked
    before phones so a card's digits are never read as a phone number. Shared by
    detect_pii and redact_pii so the flag and the masking can never disagree."""
    text = PATTERNS["email"].sub("[EMAIL]", text)
    text = PATTERNS["api_key"].sub("[API_KEY]", text)  # before card so sk_/pk_ tokens aren't reread as digits
    text = PATTERNS["credit_card_raw"].sub(_mask_card, text)
    text = PATTERNS["ssn_with_context"].sub("[SSN]", text)  # contextual undashed SSN before dashed (more specific)
    text = PATTERNS["ssn"].sub("[SSN]", text)
    text = PATTERNS["phone"].sub(_mask_phone, text)
    text = PATTERNS["address_street"].sub("[ADDRESS]", text)
    text = PATTERNS["address_zip"].sub("[ADDRESS]", text)
    text = PATTERNS["ip_address"].sub("[IP]", text)
    return text


def detect_pii(text: str) -> bool:
    """True if the text contains any detectable PII."""
    if not text:
        return False
    return _scrub(text) != text


def redact_pii(text: str) -> str:
    """Return the text with all detected PII replaced by placeholders.

    Used to mask the ticket before it reaches any LLM (safety, classifier,
    escalation, generator), so raw PII is never sent to the model or echoed into
    the output — a deterministic guarantee rather than trusting the LLM to
    self-censor. Credit cards and phones keep their last 4 digits.
    """
    if not text:
        return text
    return _scrub(text)
