from datetime import datetime

start = datetime.now()

print(f"Started at: {start}")

import csv
import json
import sys
import os
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add code folder to python path to avoid import resolution issues when running main.py
sys.path.append(os.path.dirname(__file__))

from config import TICKETS_PATH, OUTPUT_PATH, PARTIAL_OUTPUT_PATH, MAX_WORKERS
from retriever import build_indexes, retrieve
from safety import is_adversarial
from pii import redact_pii
from classifier import classify
from escalation import should_escalate
from generator import generate
from assembler import assemble

OUTPUT_COLUMNS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type",
    "justification", "confidence_score", "source_documents",
    "risk_level", "pii_detected", "language", "actions_taken"
]

# ---------------------------------------------------------------------------
# Checkpoint helpers
#
# The partial file is the checkpoint. It has one extra leading line:
#   # CHECKPOINT input_hash=<16hex> llm_provider=<str> llm_model=<str>
# followed by the normal CSV header and rows. On clean completion the partial
# is sorted to match input order, written as OUTPUT_PATH, and deleted.
# ---------------------------------------------------------------------------

def _ticket_key(row: dict) -> str:
    """Stable 16-char hash of a ticket's identifying fields. Accepts both
    input rows (Title Case keys) and output rows (lowercase keys) so the same
    key function works on either side of the pipeline."""
    issue = row.get("Issue") or row.get("issue") or ""
    subject = row.get("Subject") or row.get("subject") or ""
    company = row.get("Company") or row.get("company") or ""
    return hashlib.sha256(f"{issue}|{subject}|{company}".encode("utf-8")).hexdigest()[:16]


def _input_hash(path: Path) -> str:
    """Short sha256 of the input CSV's bytes — refuses to resume if it changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _write_partial_header(file, input_hash: str) -> None:
    """Write the metadata comment line then the CSV column header."""
    provider = os.getenv("LLM_PROVIDER", "")
    model = os.getenv("LLM_MODEL", "")
    file.write(f"# CHECKPOINT input_hash={input_hash} llm_provider={provider} llm_model={model}\n")
    csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS).writeheader()


def _load_partial(path: Path) -> tuple[dict, list[dict]]:
    """Return (metadata_dict, list_of_complete_rows). A torn final row (fewer
    columns than the header — possible on a hard crash mid-write) is dropped so
    its ticket is re-processed on resume."""
    metadata: dict = {}
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        first_line = f.readline()
        if first_line.startswith("# CHECKPOINT"):
            for part in first_line[len("# CHECKPOINT"):].strip().split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    metadata[k] = v
        else:
            f.seek(0)  # no metadata line; rewind so DictReader sees the CSV header
        for row in csv.DictReader(f):
            # csv.DictReader returns None for missing fields on a short row; a
            # complete row has a string (possibly "") for every column.
            if all(row.get(col) is not None for col in OUTPUT_COLUMNS):
                rows.append(row)
    return metadata, rows


def _finalize(input_rows: list[dict]) -> None:
    """Read the partial file, sort by input-row order, write OUTPUT_PATH,
    delete the partial. Defensive: any input row missing from the partial gets
    a fallback row so the final CSV always has exactly len(input_rows) rows."""
    _, partial_rows = _load_partial(PARTIAL_OUTPUT_PATH)
    by_key = {_ticket_key(r): r for r in partial_rows}
    final = []
    for row in input_rows:
        key = _ticket_key(row)
        if key in by_key:
            final.append(by_key[key])
        else:
            final.append(fallback_row(row, "Missing from checkpoint; regenerated as fallback."))
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(final)
    PARTIAL_OUTPUT_PATH.unlink()


# ---------------------------------------------------------------------------
# Per-ticket processing (unchanged from before — checkpointing wraps it)
# ---------------------------------------------------------------------------

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
        # Stage 2 (PII): detect + redact up front, before any LLM call, so raw
        # PII never reaches the model (safety/classifier/escalation/generator)
        # or the output. A redacted copy that differs from the original is the
        # PII signal itself.
        redacted_text = redact_pii(ticket_text)
        pii = redacted_text != ticket_text

        # Stage 1: Safety Screener
        if is_adversarial(redacted_text):
            # Assemble immediately as adversarial/escalated
            return assemble(
                row=row,
                is_adv=True,
                pii_detected=pii,
                classification={"product_area": "none", "request_type": "invalid", "risk_level": "low", "language": "en"},
                escalate=True,
                escalated_by_rules=True,
                generated={"response": "This request cannot be processed.", "actions_taken": [], "source_documents": ""},
                ticket_text=ticket_text
            )

        # Stages 3-6 below all run on redacted_text — raw PII is never sent out.

        # Stage 3: Classification
        classification = classify(redacted_text, row.get("Subject", ""), row.get("Company", ""))

        # Stage 4: Document Retrieval
        # Combine Subject + ticket_text for richer BM25 signal (HANDOFF §5 Fix B)
        product_area = classification.get("product_area", "none")
        query = f"{row.get('Subject', '')} {ticket_text}".strip()
        chunks = retrieve(query, product_area)

        # Stage 5: Escalation Gate
        escalate, escalated_by_rules, esc_reason = should_escalate(
            classification.get("risk_level", "low"),
            pii,
            redacted_text,
            classification.get("product_area", "none"),
            chunks,
            classification.get("request_subtype", ""),
        )

        # Stage 6: Response Generation
        generated = generate(redacted_text, chunks, escalate)

        # G6: the generator escalates a reply it could not resolve from the docs.
        if generated.get("escalated") and not escalate:
            escalate, escalated_by_rules, esc_reason = True, False, "generator_unresolved"

        # Stage 7: Output Assembly
        return assemble(
            row=row,
            is_adv=False,
            pii_detected=pii,
            classification=classification,
            escalate=escalate,
            escalated_by_rules=escalated_by_rules,
            generated=generated,
            ticket_text=ticket_text,
            escalation_reason=esc_reason,
        )
    except Exception as e:
        # Fallback to avoid crashes
        return fallback_row(row, f"Escalated due to pipeline processing exception: {str(e)}")


# ---------------------------------------------------------------------------
# Main loop — streams completed tickets to the partial file and resumes on
# restart from whatever the partial contains.
# ---------------------------------------------------------------------------

def main():
    print("Building BM25 retrieval indexes...")
    build_indexes()

    if not os.path.exists(TICKETS_PATH):
        print(f"Error: Input tickets file not found at {TICKETS_PATH}")
        sys.exit(1)

    print(f"Reading tickets from {TICKETS_PATH}...")
    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # --- Checkpoint: validate / load / decide what to skip ---
    input_hash = _input_hash(TICKETS_PATH)
    force_restart = os.getenv("FORCE_RESTART", "").lower() in ("1", "true", "yes")

    if force_restart and PARTIAL_OUTPUT_PATH.exists():
        print(f"FORCE_RESTART set — deleting existing partial file {PARTIAL_OUTPUT_PATH}")
        PARTIAL_OUTPUT_PATH.unlink()

    processed_keys: set[str] = set()
    if PARTIAL_OUTPUT_PATH.exists():
        partial_meta, partial_rows = _load_partial(PARTIAL_OUTPUT_PATH)
        # Hard fail on input mismatch — user must revert the input or FORCE_RESTART.
        if partial_meta.get("input_hash") != input_hash:
            print(f"ERROR: input CSV has changed since the checkpoint was created.")
            print(f"  Partial input hash : {partial_meta.get('input_hash')}")
            print(f"  Current input hash : {input_hash}")
            print(f"  Revert {TICKETS_PATH.name} or set FORCE_RESTART=1 to discard the checkpoint.")
            sys.exit(1)
        # Soft warn on provider/model mismatch — continue, but the warning makes
        # the mixed-model output visible.
        current_provider = os.getenv("LLM_PROVIDER", "")
        current_model = os.getenv("LLM_MODEL", "")
        if (partial_meta.get("llm_provider") != current_provider
                or partial_meta.get("llm_model") != current_model):
            print("WARNING: provider/model differs from the partial checkpoint.")
            print(f"  Partial : {partial_meta.get('llm_provider')} / {partial_meta.get('llm_model')}")
            print(f"  Current : {current_provider} / {current_model}")
            print("  Continuing — final output will contain rows generated by BOTH.")
        processed_keys = {_ticket_key(r) for r in partial_rows}
        print(f"Resuming from {len(processed_keys)}/{len(rows)} prior tickets. "
              f"Set FORCE_RESTART=1 to start fresh.")

    pending = [r for r in rows if _ticket_key(r) not in processed_keys]

    if not pending:
        print(f"All {len(rows)} tickets already processed — finalizing output.")
    else:
        print(f"Processing {len(pending)} tickets with {MAX_WORKERS} workers...")
        os.makedirs(os.path.dirname(PARTIAL_OUTPUT_PATH), exist_ok=True)

        is_fresh = not PARTIAL_OUTPUT_PATH.exists()
        write_lock = threading.Lock()

        with open(PARTIAL_OUTPUT_PATH, "a", encoding="utf-8", newline="") as pf:
            if is_fresh:
                _write_partial_header(pf, input_hash)
                pf.flush()
            writer = csv.DictWriter(pf, fieldnames=OUTPUT_COLUMNS)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_ticket, r): r for r in pending}
                for i, fut in enumerate(as_completed(futures), 1):
                    result = fut.result()
                    with write_lock:
                        writer.writerow(result)
                        pf.flush()
                    if i % 5 == 0 or i == len(pending):
                        print(f"  {i}/{len(pending)} done")

    print(f"Sorting and writing final output to {OUTPUT_PATH}...")
    _finalize(rows)
    print("Done!")

    end = datetime.now()

    print(f"Finished at: {end}")
    print(f"Total runtime: {end - start}")

if __name__ == "__main__":
    main()
