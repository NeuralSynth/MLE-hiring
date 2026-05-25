import re

# PII regex patterns
PATTERNS = {
    "email": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    # General credit card format
    "credit_card_raw": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    # Address patterns: street followed by road types
    "address_street": re.compile(r"\b\d+\s+[A-Za-z0-9\s,.]+?\s+(?:Street|St|Avenue|Ave|Road|Rd|Highway|Hwy|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Circle|Cir|Trail|Trl|Way|Plaza|Plz|Suite|Apt)\b", re.IGNORECASE),
    # City, State ZIP (e.g. Springfield, IL 62704)
    "address_zip": re.compile(r"\b[A-Za-z\s\-.']+,\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")
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

def detect_pii(text: str) -> bool:
    if not text:
        return False
    
    # 1. Simple regex checks
    if PATTERNS["email"].search(text):
        return True
    if PATTERNS["ssn"].search(text):
        return True
    if PATTERNS["phone"].search(text):
        return True
    if PATTERNS["address_street"].search(text):
        return True
    if PATTERNS["address_zip"].search(text):
        return True
    
    # 2. Credit Card Check with Luhn validation to reduce false positives
    cc_matches = PATTERNS["credit_card_raw"].findall(text)
    for match in cc_matches:
        # Strip spaces and dashes
        cleaned_cc = match.replace(" ", "").replace("-", "")
        if check_luhn(cleaned_cc):
            return True
            
    return False
