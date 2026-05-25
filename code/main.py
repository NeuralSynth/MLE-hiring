import csv
import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add code folder to python path to avoid import resolution issues when running main.py
sys.path.append(os.path.dirname(__file__))

from config import TICKETS_PATH, OUTPUT_PATH, MAX_WORKERS
from retriever import build_indexes, retrieve
from safety import is_adversarial
from pii import detect_pii
from classifier import classify
from escalation import should_escalate
from generator import generate
from assembler import assemble

OUTPUT_COLUMNS = [
    "issue", "subject", "company", "response", "product_area", "status",
    "request_type", "justification", "confidence_score", "source_documents",
    "risk_level", "pii_detected", "language", "actions_taken"
]

def extract_text(issue_json: str) -> str:
    if not issue_json:
        return ""
    try:
        # Check if it looks like a JSON array
        turns = json.loads(issue_json)
        if isinstance(turns, list):
            formatted_turns = []
            for turn in turns:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "").strip()
                formatted_turns.append(f"{role}: {content}")
            return "\n".join(formatted_turns)
    except Exception:
        pass
    return str(issue_json).strip()

def fallback_row(row: dict, justification: str = "Escalated due to pipeline processing exception.") -> dict:
    return {
        "issue": row.get("Issue", ""),
        "subject": row.get("Subject", ""),
        "company": row.get("Company", ""),
        "response": "Your request has been escalated to a human support specialist.",
        "product_area": "none",
        "status": "escalated",
        "request_type": "product_issue",
        "justification": justification,
        "confidence_score": 0.60,
        "source_documents": "",
        "risk_level": "low",
        "pii_detected": "false",
        "language": "en",
        "actions_taken": "[]"
    }

def process_ticket(row: dict) -> dict:
    try:
        ticket_text = extract_text(row.get("Issue", ""))
        
        # Stage 1: Safety Screener
        if is_adversarial(ticket_text):
            # Assemble immediately as adversarial/escalated
            return assemble(
                row=row,
                is_adv=True,
                pii_detected=False,
                classification={"product_area": "none", "request_type": "invalid", "risk_level": "low", "language": "en"},
                escalate=True,
                escalated_by_rules=True,
                generated={"response": "This request cannot be processed.", "actions_taken": [], "source_documents": ""},
                ticket_text=ticket_text
            )
            
        # Stage 2: PII Detection
        pii = detect_pii(ticket_text)
        
        # Stage 3: Classification
        classification = classify(ticket_text, row.get("Subject", ""), row.get("Company", ""))
        
        # Stage 4: Document Retrieval
        product_area = classification.get("product_area", "none")
        chunks = retrieve(ticket_text, product_area)
        
        # Stage 5: Escalation Gate
        escalate, escalated_by_rules = should_escalate(
            classification.get("risk_level", "low"),
            pii,
            ticket_text,
            classification.get("product_area", "none"),
            chunks
        )
        
        # Stage 6: Response Generation
        generated = generate(ticket_text, chunks, escalate)
        
        # Stage 7: Output Assembly
        return assemble(
            row=row,
            is_adv=False,
            pii_detected=pii,
            classification=classification,
            escalate=escalate,
            escalated_by_rules=escalated_by_rules,
            generated=generated,
            ticket_text=ticket_text
        )
    except Exception as e:
        # Fallback to avoid crashes
        return fallback_row(row, f"Escalated due to pipeline processing exception: {str(e)}")

def main():
    print("Building BM25 retrieval indexes...")
    build_indexes()
    
    if not os.path.exists(TICKETS_PATH):
        print(f"Error: Input tickets file not found at {TICKETS_PATH}")
        sys.exit(1)
        
    print(f"Reading tickets from {TICKETS_PATH}...")
    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        # Read the file and handle cases with double quotes or multiple lines
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Processing {len(rows)} tickets with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_ticket, rows))
        
    print(f"Writing outputs to {OUTPUT_PATH}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
        
    print("Done!")

if __name__ == "__main__":
    main()
