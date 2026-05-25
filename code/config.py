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

LLM_TEMPERATURE = 0
MAX_WORKERS = 5
BM25_TOP_K = 5

ESCALATION_KEYWORDS = [
    "lawsuit", "attorney", "lawyer", "legal action",
    "sue", "court", "gdpr", "data breach", "identity theft",
    "account compromise", "unauthorized access", "fraud",
    "stolen", "police", "report crime", "regulatory",
    "compliance violation", "class action"
]
