import pytest
import sys
import os
import json
import re
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load env vars before any test runs
load_dotenv()

# Keep the suite deterministic and model-independent: the semantic re-rank is
# exercised by its own unit tests, not the whole pipeline.
os.environ.setdefault("DISABLE_EMBEDDINGS", "1")

# Make code/ importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever import build_indexes

def mock_complete(system: str, user: str) -> str:
    system_lower = system.lower()
    user_lower = user.lower()

    # 0. Check for direct JSON echoing tests (e.g. test_llm_returns_valid_json)
    if "json" in system_lower and "{" in user_lower:
        match = re.search(r"(\{.*?\})", user, re.DOTALL)
        if match:
            return match.group(1)

    # 1. Safety check
    if "security classifier" in system_lower:
        if any(kw in user_lower for kw in [
            "ignore", "dan", "system prompt", "reveal", "maintenance",
            "lawyer", "lawsuit", "class action", "calc", "pwned", "pwn",
            "instruction", "authorized", "list", "files", "knowledge base",
            "marked", "replied", "confidence", "escalation",
            # mock-side stand-ins for the new (B) and (C) categories the real LLM
            # will recognize semantically via the strengthened safety prompt:
            "previous agent", "previous representative", "scrape", "exfiltrat",
        ]):
            return "adversarial"
        if "ignorez" in user_lower or "invite" in user_lower:
            return "adversarial"
        return "safe"

    # 2. Classifier check
    if "ticket classifier" in system_lower:
        product = "none"
        if "hackerrank" in user_lower or "devplatform" in user_lower or "test" in user_lower or "assessment" in user_lower:
            product = "devplatform"
        elif "claude" in user_lower or "anthropic" in user_lower:
            product = "claude"
        elif "visa" in user_lower:
            product = "visa"

        req_type = "product_issue"
        if "bug" in user_lower or "error" in user_lower or "crash" in user_lower:
            req_type = "bug"
        elif "feature" in user_lower or "request" in user_lower:
            req_type = "feature_request"
        elif "ignore" in user_lower or "dan" in user_lower or "override" in user_lower or "calc" in user_lower:
            req_type = "invalid"

        risk = "low"
        if any(w in user_lower for w in ["unauthorized", "stolen", "hack", "leak", "compromise", "breach"]):
            risk = "critical"
        elif any(w in user_lower for w in ["legal", "sue", "lawsuit", "complaint"]):
            risk = "high"

        return json.dumps({
            "product_area": product,
            "request_type": req_type,
            "risk_level": risk,
            "language": "en"
        })

    # 3. Escalation supervisor check
    if "customer support supervisor" in system_lower:
        if any(kw in user_lower for kw in ["supervisor", "manager", "escalate", "unresolved", "frustrated"]):
            return "escalate"
        return "reply"

    # 4. Generator check
    if "customer support agent" in system_lower:
        if "flagged for escalation" in system_lower or "escalate_to_human" in system_lower or "escalate" in user_lower:
            return json.dumps({
                "response": "Your request has been escalated to a human specialist. We will review your ticket and follow up shortly.",
                "actions_taken": [{
                    "action": "escalate_to_human",
                    "parameters": {
                        "priority": "normal",
                        "department": "general",
                        "summary": "Escalated by supervisor."
                    }
                }],
                "source_documents": ""
            })
        
        doc_path = ""
        match = re.search(r"path:\s*([^\s\)]+\.md)", user_lower)
        if match:
            doc_path = match.group(1).strip()
        
        return json.dumps({
            "response": "Here is the answer from support documents.",
            "actions_taken": [],
            "source_documents": doc_path
        })

    return "mock response"

@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()

@pytest.fixture(scope="session", autouse=True)
def initialize_indexes_and_mocks(monkeypatch_session):
    """Build BM25 indexes and mock all LLM complete calls for the session."""
    build_indexes()
    
    from llm import llm
    monkeypatch_session.setattr(llm, "complete", mock_complete)
