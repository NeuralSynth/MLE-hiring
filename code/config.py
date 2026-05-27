from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

CORPUS_PATHS = {
    "devplatform": DATA_DIR / "devplatform",
    "claude": DATA_DIR / "claude",
    "visa": DATA_DIR / "visa",
    "none": None,
}

API_SPECS_DIR = DATA_DIR / "api_specs"
TICKETS_PATH = ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_PATH = ROOT / "support_tickets" / "output.csv"

import os

LLM_TEMPERATURE = 0
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 5))
BM25_TOP_K = 5

# Legal / compliance language — always escalate (must be handled by a human).
LEGAL_KEYWORDS = [
    "lawsuit", "attorney", "lawyer", "legal action", "sue", "sued", "suing",
    "court", "gdpr", "data breach", "identity theft", "account compromise",
    "unauthorized access", "fraud", "fraudulent", "stolen", "police",
    "report crime", "regulatory", "compliance violation", "class action",
    "contract dispute",
]

# Explicit requests to reach a human — escalate, but for a different reason.
HUMAN_REQUEST_KEYWORDS = [
    "escalate", "supervisor", "manager", "human agent", "human specialist",
    "representative", "human support", "speak to a human", "speak to someone",
    "talk to a human", "real person", "live agent",
]

# Financial-action words (combined with detected PII -> escalate). Common
# inflections are listed explicitly because matching is whole-word.
FINANCIAL_WORDS = [
    "refund", "refunds", "refunded", "charge", "charges", "charged",
    "chargeback", "transaction", "transactions", "payment", "payments",
    "transfer", "debit", "credit", "invoice", "invoiced", "billing", "billed",
]
