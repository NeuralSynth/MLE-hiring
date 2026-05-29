import pytest
from pii import detect_pii, redact_pii

# Should detect PII
def test_detects_email():
    assert detect_pii("contact me at user@example.com please") == True

def test_detects_phone():
    assert detect_pii("call me at +1-800-555-0199") == True

def test_detects_ssn():
    assert detect_pii("my SSN is 123-45-6789") == True

def test_detects_credit_card():
    assert detect_pii("card number 4111111111111111") == True

def test_detects_street_address():
    assert detect_pii("ship it to 123 Main Street, Springfield") == True

def test_detects_parenthesized_phone():
    assert detect_pii("reach me at (800) 555-0199") == True

# Should NOT detect PII
def test_clean_ticket():
    assert detect_pii("I cannot log into my account") == False

def test_clean_technical_question():
    assert detect_pii("How do I reset my API rate limits?") == False

def test_product_name_not_pii():
    assert detect_pii("I use DevPlatform for my interviews") == False

# Should NOT false-positive on street-type abbreviations in ordinary prose
def test_street_abbrev_not_pii():
    assert detect_pii("I have 3 items in St Louis store") == False
    assert detect_pii("I waited 2 days at Dr Smith office") == False

# Should NOT treat a bare 10-digit ID as a phone number
def test_order_id_not_phone():
    assert detect_pii("Order 1234567890 has not arrived") == False


# --- Redaction (#6): mask PII before it reaches the LLM or the output ---

def test_redacts_email():
    out = redact_pii("email me at john.doe@acme.com please")
    assert "john.doe@acme.com" not in out
    assert "[EMAIL]" in out

def test_redacts_card_keeps_last4():
    out = redact_pii("my card 4111 1111 1111 1111 was charged")
    assert "4111 1111 1111 1111" not in out
    assert "****1111" in out

def test_redacts_ssn_and_phone():
    out = redact_pii("ssn 123-45-6789 phone +1-800-555-0199")
    assert "123-45-6789" not in out
    assert "[SSN]" in out
    assert "555-0199" not in out
    assert "0199" in out  # last 4 retained as [PHONE ...0199]

def test_redacts_address():
    out = redact_pii("ship to 123 Main Street, Springfield, IL 62704")
    assert "123 Main Street" not in out
    assert "[ADDRESS]" in out

def test_redact_preserves_clean_text():
    text = "I cannot log into my account, please help"
    assert redact_pii(text) == text

def test_redact_is_idempotent():
    text = "card 4111111111111111 email a@b.com phone 800-555-0199"
    once = redact_pii(text)
    assert redact_pii(once) == once

def test_detect_and_redact_agree():
    for text in [
        "email a@b.com",
        "card 4111111111111111",
        "I cannot log in",
        "phone +1-800-555-0199",
        "Order 1234567890 pending",
    ]:
        assert detect_pii(text) == (redact_pii(text) != text)


# --- Gap G extensions: IP, contextual undashed SSN, API keys ---

def test_detects_ipv4_address():
    assert detect_pii("server at 192.168.1.100 is unreachable") is True

def test_does_not_detect_invalid_ipv4():
    """1.2.3.999 has an out-of-range octet — must NOT be flagged as an IP."""
    assert detect_pii("version is 1.2.3.999") is False

def test_redacts_ipv4_address():
    out = redact_pii("login from 10.0.0.5 failed")
    assert "10.0.0.5" not in out
    assert "[IP]" in out

def test_detects_undashed_ssn_with_context():
    """Undashed 9-digit run with adjacent SSN/social-security context word."""
    assert detect_pii("my SSN is 123456789 please verify") is True
    assert detect_pii("Social Security Number: 987654321") is True

def test_does_not_detect_bare_9_digit_run():
    """A bare 9-digit run (order / tracking IDs) must NOT be flagged as SSN."""
    assert detect_pii("Order 123456789 is pending") is False
    assert detect_pii("Tracking number 987654321 shipped") is False

def test_redacts_undashed_ssn_with_context():
    out = redact_pii("my SSN is 123456789 thanks")
    assert "123456789" not in out
    assert "[SSN]" in out

def test_detects_openai_api_key():
    assert detect_pii("my key is sk-proj-abcdef1234567890ABCDEF leaked") is True

def test_detects_stripe_api_key():
    assert detect_pii("token sk_live_51AbCdEfGhIjKlMnOpQrSt was exposed") is True

def test_does_not_detect_short_sk_prefix():
    """A short 'sk-foo' or 'pk-x' is ordinary text, not a key — must not match."""
    assert detect_pii("sk-foo and pk-bar are short prefixes") is False

def test_redacts_api_key():
    out = redact_pii("my secret is sk-proj-abcdef1234567890ABCDEF1234")
    assert "sk-proj-abcdef" not in out
    assert "[API_KEY]" in out
