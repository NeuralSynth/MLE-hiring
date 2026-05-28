"""Tests for the checkpoint / resume mechanism in main.py.

The unit tests cover the helpers in isolation (ticket key, input hash, partial
read/write, torn-row recovery). The integration tests build a partial file by
hand and exercise the main() resume path against the live `sample_support_tickets.csv`.
"""

import csv
import hashlib
import os
from pathlib import Path

import pytest

from main import (
    OUTPUT_COLUMNS,
    _ticket_key,
    _input_hash,
    _load_partial,
    _write_partial_header,
    _finalize,
    fallback_row,
    main,
)
from config import TICKETS_PATH, OUTPUT_PATH, PARTIAL_OUTPUT_PATH


# ---------------------------------------------------------------------------
# Helper: a minimal output row dict so we don't depend on the full pipeline
# ---------------------------------------------------------------------------

def _row(issue: str, subject: str = "S", company: str = "C", response: str = "ok") -> dict:
    return {
        "issue": issue,
        "subject": subject,
        "company": company,
        "response": response,
        "product_area": "claude",
        "status": "replied",
        "request_type": "product_issue",
        "justification": "Answered.",
        "confidence_score": "0.95",
        "source_documents": "",
        "risk_level": "low",
        "pii_detected": "false",
        "language": "en",
        "actions_taken": "[]",
    }


# ---------------------------------------------------------------------------
# _ticket_key
# ---------------------------------------------------------------------------

def test_ticket_key_is_stable():
    row = {"Issue": "x", "Subject": "y", "Company": "z"}
    assert _ticket_key(row) == _ticket_key(row)
    assert len(_ticket_key(row)) == 16


def test_ticket_key_works_on_both_input_and_output_row_shapes():
    """Input rows use Title Case keys; output rows use lowercase. The same
    key function must produce the same hash for the same ticket on either side."""
    input_row = {"Issue": "foo", "Subject": "bar", "Company": "baz"}
    output_row = {"issue": "foo", "subject": "bar", "company": "baz"}
    assert _ticket_key(input_row) == _ticket_key(output_row)


def test_ticket_key_distinguishes_different_tickets():
    a = _ticket_key({"Issue": "foo", "Subject": "x", "Company": "y"})
    b = _ticket_key({"Issue": "bar", "Subject": "x", "Company": "y"})
    assert a != b


# ---------------------------------------------------------------------------
# _input_hash
# ---------------------------------------------------------------------------

def test_input_hash_same_for_same_file(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text("Issue,Subject,Company\nfoo,bar,baz\n", encoding="utf-8")
    assert _input_hash(p) == _input_hash(p)


def test_input_hash_changes_when_file_changes(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text("Issue,Subject,Company\nfoo,bar,baz\n", encoding="utf-8")
    h1 = _input_hash(p)
    p.write_text("Issue,Subject,Company\nfoo,bar,baz\nextra,row,here\n", encoding="utf-8")
    h2 = _input_hash(p)
    assert h1 != h2


# ---------------------------------------------------------------------------
# _write_partial_header + _load_partial round-trip
# ---------------------------------------------------------------------------

def test_partial_round_trip_preserves_metadata_and_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3:latest")
    p = tmp_path / "partial.csv"
    rows = [_row("t1"), _row("t2"), _row("t3")]

    with open(p, "w", encoding="utf-8", newline="") as f:
        _write_partial_header(f, input_hash="deadbeefcafef00d")
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        for r in rows:
            writer.writerow(r)

    metadata, loaded = _load_partial(p)
    assert metadata["input_hash"] == "deadbeefcafef00d"
    assert metadata["llm_provider"] == "ollama"
    assert metadata["llm_model"] == "llama3:latest"
    assert len(loaded) == 3
    assert [r["issue"] for r in loaded] == ["t1", "t2", "t3"]


def test_partial_load_drops_torn_last_row(tmp_path):
    """A hard crash mid-write can leave a final row with fewer fields than
    the header. _load_partial must drop it so the ticket is reprocessed."""
    p = tmp_path / "partial.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        _write_partial_header(f, input_hash="abc")
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writerow(_row("complete"))
        # Now emit a torn row by writing fewer fields than the header expects.
        f.write("torn,row,only,three,cols\n")

    _, rows = _load_partial(p)
    assert len(rows) == 1
    assert rows[0]["issue"] == "complete"


def test_partial_load_handles_missing_metadata_line(tmp_path):
    """A partial file written by an older version (no # CHECKPOINT line) still
    parses — we just get an empty metadata dict."""
    p = tmp_path / "partial.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerow(_row("legacy"))

    metadata, rows = _load_partial(p)
    assert metadata == {}
    assert len(rows) == 1
    assert rows[0]["issue"] == "legacy"


# ---------------------------------------------------------------------------
# _finalize: sort by input order; reconstruct missing rows
# ---------------------------------------------------------------------------

def test_finalize_sorts_output_by_input_order(tmp_path, monkeypatch):
    """Workers complete in arbitrary order. The final output.csv must match
    the input row order regardless of which thread finished first."""
    # Redirect OUTPUT_PATH and PARTIAL_OUTPUT_PATH into tmp_path.
    output = tmp_path / "output.csv"
    partial = tmp_path / "output.partial.csv"
    monkeypatch.setattr("main.OUTPUT_PATH", output)
    monkeypatch.setattr("main.PARTIAL_OUTPUT_PATH", partial)

    # Input order: A, B, C
    input_rows = [
        {"Issue": "A", "Subject": "s", "Company": "c"},
        {"Issue": "B", "Subject": "s", "Company": "c"},
        {"Issue": "C", "Subject": "s", "Company": "c"},
    ]

    # Partial rows written in completion order: C, A, B
    with open(partial, "w", encoding="utf-8", newline="") as f:
        _write_partial_header(f, input_hash="x")
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        for issue in ("C", "A", "B"):
            writer.writerow(_row(issue))

    _finalize(input_rows)

    with open(output, "r", encoding="utf-8") as f:
        out_rows = list(csv.DictReader(f))
    assert [r["issue"] for r in out_rows] == ["A", "B", "C"]
    assert not partial.exists(), "partial file should be deleted after finalize"


def test_finalize_backfills_missing_ticket_with_fallback(tmp_path, monkeypatch):
    """If a row is missing from the partial (shouldn't happen, but defensive),
    finalize must still emit a valid row so the CSV has len(input) rows."""
    output = tmp_path / "output.csv"
    partial = tmp_path / "output.partial.csv"
    monkeypatch.setattr("main.OUTPUT_PATH", output)
    monkeypatch.setattr("main.PARTIAL_OUTPUT_PATH", partial)

    input_rows = [
        {"Issue": "A", "Subject": "s", "Company": "c"},
        {"Issue": "MISSING", "Subject": "s", "Company": "c"},
    ]

    with open(partial, "w", encoding="utf-8", newline="") as f:
        _write_partial_header(f, input_hash="x")
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writerow(_row("A"))  # only A is in the partial

    _finalize(input_rows)

    with open(output, "r", encoding="utf-8") as f:
        out_rows = list(csv.DictReader(f))
    assert len(out_rows) == 2
    assert out_rows[0]["issue"] == "A"
    assert out_rows[1]["issue"] == "MISSING"
    # The missing-ticket row must be a valid escalation fallback.
    assert out_rows[1]["status"] == "escalated"


# ---------------------------------------------------------------------------
# main() — integration: input-hash mismatch hard-fails
# ---------------------------------------------------------------------------

SAMPLE_CSV = Path(__file__).parent.parent.parent / "support_tickets" / "sample_support_tickets.csv"


def test_main_terminates_on_input_hash_mismatch(tmp_path, monkeypatch, capsys):
    """A partial file with a stale input_hash must cause main() to sys.exit(1)
    with a clear message — no pipeline work happens."""
    # Point TICKETS_PATH and the checkpoint paths at tmp_path copies.
    tickets = tmp_path / "support_tickets.csv"
    tickets.write_text(SAMPLE_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "output.csv"
    partial = tmp_path / "output.partial.csv"

    monkeypatch.setattr("main.TICKETS_PATH", tickets)
    monkeypatch.setattr("main.OUTPUT_PATH", output)
    monkeypatch.setattr("main.PARTIAL_OUTPUT_PATH", partial)

    # Write a partial whose recorded input_hash deliberately doesn't match.
    with open(partial, "w", encoding="utf-8", newline="") as f:
        _write_partial_header(f, input_hash="0000000000000000")
        csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS).writeheader()

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "input CSV has changed" in out
    assert "FORCE_RESTART=1" in out


def test_main_resumes_and_skips_processed_tickets(tmp_path, monkeypatch, capsys):
    """A partial file containing one already-processed ticket: main() should
    skip it, process only the remainder, then emit a sorted output.csv."""
    tickets = tmp_path / "support_tickets.csv"
    tickets.write_text(SAMPLE_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "output.csv"
    partial = tmp_path / "output.partial.csv"

    monkeypatch.setattr("main.TICKETS_PATH", tickets)
    monkeypatch.setattr("main.OUTPUT_PATH", output)
    monkeypatch.setattr("main.PARTIAL_OUTPUT_PATH", partial)

    # Read the input to learn the first ticket's identifying fields and
    # build a matching partial row.
    with open(tickets, "r", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))
    first = input_rows[0]
    pre_processed = _row(first["Issue"], first["Subject"], first["Company"], response="PRESEEDED")

    # Compute the input_hash so the partial is considered valid for resume.
    current_input_hash = _input_hash(tickets)
    with open(partial, "w", encoding="utf-8", newline="") as f:
        _write_partial_header(f, input_hash=current_input_hash)
        csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS).writerow(pre_processed)

    main()
    out = capsys.readouterr().out
    assert "Resuming from 1/" in out

    # Final output should preserve the pre-seeded row (proving it was NOT
    # re-processed) and be in input order.
    with open(output, "r", encoding="utf-8") as f:
        out_rows = list(csv.DictReader(f))
    assert len(out_rows) == len(input_rows)
    assert out_rows[0]["response"] == "PRESEEDED"
    # And the partial file is gone after successful finalize.
    assert not partial.exists()


def test_force_restart_discards_existing_partial(tmp_path, monkeypatch, capsys):
    """FORCE_RESTART=1 must delete the partial before resume logic runs, so
    even a valid checkpoint is discarded."""
    tickets = tmp_path / "support_tickets.csv"
    tickets.write_text(SAMPLE_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "output.csv"
    partial = tmp_path / "output.partial.csv"

    monkeypatch.setattr("main.TICKETS_PATH", tickets)
    monkeypatch.setattr("main.OUTPUT_PATH", output)
    monkeypatch.setattr("main.PARTIAL_OUTPUT_PATH", partial)
    monkeypatch.setenv("FORCE_RESTART", "1")

    # Write a (valid-hash) partial that would otherwise be honored.
    with open(partial, "w", encoding="utf-8", newline="") as f:
        _write_partial_header(f, input_hash=_input_hash(tickets))
        csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS).writerow(_row("would-be-skipped"))

    main()
    out = capsys.readouterr().out
    assert "FORCE_RESTART set" in out
    assert "Resuming from" not in out
