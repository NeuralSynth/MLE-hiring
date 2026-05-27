# Support Triage Agent Architecture

This document details the design, pipeline flow, and technical implementation of the multi-domain support triage agent.

---

## 1. High-Level Architecture Flow

The agent uses a **Sequential Pipeline** design rather than an agent loop. Tickets pass through a sequence of deterministic stages with exactly 3 LLM calls per ticket maximum.

```
support_tickets.csv
        │
        ▼
   [PARSE TICKET]
   Extract Issue JSON → conversation turns (single and multi-turn)
   Extract Subject, Company — Company treated as hint only
        │
        ▼
   [STAGE 1: SAFETY SCREENER]                           safety.py
   De-obfuscate: NFKC + strip zero-width, decode base64/hex,
     flatten homoglyphs (rule-matching copy only)
   Deterministic injection rules (override / role-change /
     exfiltration / formula / multilingual) → short-circuit
   Else hardened LLM call → "safe" | "adversarial"
   If adversarial → STOP: write hardcoded escalation row
        │
        ▼
   [STAGE 2: PII DETECTOR + REDACTION]                  pii.py
   Regex only — no LLM, deterministic
   Detects: emails, phones, credit cards (Luhn), SSNs, addresses
   Redacts PII to placeholders ([EMAIL], [CARD ****1234], ...)
     before any LLM call; card/phone keep last 4 digits
   Output: pii_detected (bool) + redacted text for the LLM stages
        │
        ▼
   [STAGE 3: CLASSIFIER]                                classifier.py
   Single LLM call → JSON
   Output: product_area, request_type, risk_level, language
   Infers product_area from content, NOT from Company field
        │
        ▼
   [STAGE 4: RETRIEVER]                                 retriever.py
   Chunk-level BM25 on product_area subfolder ONLY
   Tokenizer: alphanumeric split + stopword removal (query+index)
   Docs cleaned (frontmatter / URLs / Related Articles stripped),
     header-aware chunked, enriched with title+breadcrumb+slug
   Relevance floor (relative) + max chunks/doc
   Output: top-k chunks (path, content, score)
        │
        ▼
   [STAGE 5: ESCALATION GATE]                           escalation.py
   Rules-first — LLM cannot override hardcoded rules:
     - risk_level = critical
     - legal keyword in ticket text
     - pii_detected=True AND financial word in ticket text
     - product_area="none" AND ticket < 20 words
     - retrieved_chunks is empty
   LLM only for ambiguous cases after rules pass
        │
        ▼
   [STAGE 6: RESPONSE GENERATOR]                        generator.py
   If escalate=True  → escalation message + escalate_to_human action
   If escalate=False → grounded response from cleaned chunks only
   Output: response, actions_taken (list), source_documents (path str)
        │
        ▼
   [STAGE 7: OUTPUT ASSEMBLER]                          assembler.py
   Validates: json.loads(actions_taken) succeeds
   Validates: source_documents path exists on disk or is empty string
   Calculates: confidence_score deterministically (no LLM)
   Writes: one complete row with all OUTPUT_COLUMNS in correct case
```

---

## 2. Component Breakdown

### Safety Screener (`safety.py`)
- **Purpose**: Prevent prompt injection and system-prompt / document leakage, including obfuscated and non-English attacks.
- **Defense in depth (de-obfuscate → rules → LLM):**
  1. **Normalize** — Unicode NFKC plus stripping of zero-width / control characters, defeating full-width look-alikes and invisible-character keyword splitting (e.g. zero-width spaces inserted mid-word).
  2. **Decode** — base64 and hex segments that decode to readable text are appended to the screened text, so payloads hidden in encodings/binaries become visible. Segments that decode to noise (ordinary words, IDs, card numbers) are discarded.
  3. **Deterministic rules** — high-precision regexes for instruction override, role/persona change, data exfiltration, output manipulation, and spreadsheet-formula injection, plus a curated multilingual phrase backstop. Matching runs on a homoglyph-flattened, de-accented copy (Cyrillic/Greek look-alikes → Latin) so `Ignоre` (Cyrillic `о`) is caught. A rule match short-circuits to `adversarial` — the model cannot override it.
  4. **LLM screener** — runs on the de-obfuscated text for novel / nuanced / multilingual cases the rules don't cover; it is the primary multilingual detector. Temperature `0`.
- **Why before retrieval**: screening first eliminates the risk of adversarial prompts manipulating downstream logic. If flagged as `adversarial`, the pipeline immediately stops and writes a canned escalation row — the failure mode is safe, so over-detection cannot leak.
- **Homoglyph flattening is applied only to the rule-matching copy**, never to the text sent to the LLM, so genuine non-Latin (e.g. Russian, Chinese) tickets are screened intact.

### PII Detector (`pii.py`)
- **Purpose**: Detect personal data and prevent it from reaching the model or the output.
- **Approach**: Pure regex — emails, SSNs, phone numbers, credit card numbers (Luhn checksum), physical addresses.
- **Why no LLM**: Guarantees 100% deterministic, zero-latency execution. LLMs can hallucinate PII or miss it; regex cannot be prompted into changing its behavior.
- **Redaction (preventive, not just detective)**: `redact_pii()` masks every detected span before the ticket reaches any LLM (safety, classifier, escalation, generator) and therefore before anything is written to the output. Raw PII is never sent to the model or echoed back — a deterministic guarantee instead of trusting the LLM's "don't echo PII" instruction. Credit cards and phones keep their last 4 digits (`[CARD ****1234]`) so the agent can still reference "your card ending 1234"; emails, SSNs and addresses are fully masked. The local BM25 retrieval query still uses the raw text (no network), and the output `issue` column preserves the original ticket.
- **One engine**: `detect_pii()` and `redact_pii()` share `_scrub()`, so the `pii_detected` flag can never disagree with what was masked.

### Classifier (`classifier.py`)
- **Purpose**: Identify ticket metadata — product area, request type, risk level, language.
- **Approach**: Structured JSON classification via LLM at `temperature=0`.
- **Key constraint**: Ignores the `Company` field hint if it contradicts ticket text content.

### BM25 Retriever (`retriever.py`)
- **Purpose**: Retrieve relevant support documentation from the local corpus.
- **Why BM25 over vector DB**:
  1. **Determinism**: BM25 uses exact term statistics; the index is fully reproducible across runs.
  2. **Speed**: Index built once at startup; all searches are in-process with zero network calls.
  3. **No dependencies**: No embedding API keys, rate limits, or network calls required.
- **Tokenization**: alphanumeric split (`[a-z0-9]+`) plus stopword removal, applied symmetrically to the index and the query. Fixes punctuation gluing (`api.` ≠ `api`) and stops common words (how/do/i/my) from dominating BM25 scoring.
- **Content cleaning** (`clean_content`): strips YAML frontmatter, link URLs (keeps anchor text), bare URLs, `Related Articles` blocks, and the `_Last updated_` line — so a document is not credited for topics it only *links* to, and YAML no longer leaks into responses.
- **Header-aware chunking**: each file is split by markdown headers into passages (small sections merged, sections over ~320 words window-split with overlap, headings kept inline). 85% of corpus docs exceed the ~190-word embedding/scoring sweet spot, so whole-doc indexing previously truncated/diluted them; chunking gives focused units and a smaller generator context.
- **Index enrichment**: the document title, frontmatter breadcrumbs, and the filename slug are added to a chunk's BM25 tokens (not the displayed content), surfacing the right article for question-style queries.
- **Relevance gate**: hits below a relative floor (a fraction of the top score) are dropped and chunks-per-document are capped, so the long tail of marginal matches never reaches the generator.
- **Query construction**: `Subject + ticket_text` combined for BM25 — more signal than ticket content alone.
- **Excluded files**: a `EXCLUDED_FILES` set blocks trap/deprecated/off-topic documents from every area's index.
- **Known limitation**: retrieval still searches only the classified `product_area` (hard gate); a soft cross-area prior and an optional local semantic re-rank are deferred.

### Escalation Gate (`escalation.py`)
- **Rules-first rationale**: Hardcoded rules cannot be bypassed by adversarial prompts or LLM hallucination. Legal, GDPR, and fraud cases must *always* escalate regardless of LLM opinion.
- **LLM as second pass**: Used only on genuinely ambiguous tickets that clear all rules — where escalation is a judgment call, not a compliance requirement.

### Response Generator (`generator.py`)
- **Purpose**: Compose the customer response and trigger tool calls.
- **Approach**: Grounds the answer strictly in retrieved documents. Injects `internal_tools.json` schemas.
- **`verify_identity` guardrail**: Only triggered when ALL of: (1) account-level action requested, (2) involves money/access changes, (3) no prior verification in conversation. Specifically NOT triggered for general information questions, status inquiries, or non-modifying configuration queries.
- **Escalation action preservation**: `escalate_to_human` actions generated by the LLM are passed through as-is; they are never cleared after generation.

### Assembler (`assembler.py`)
- **Purpose**: Serialize outputs and compute confidence score.
- **Confidence is deterministic** — never asked from the LLM (LLMs routinely self-report `0.99` regardless of evidence quality):
  - `invalid` request type → `1.0`
  - Adversarial detected → `0.99`
  - Clean FAQ match with source doc → `0.95`
  - Rule-based escalation → `0.80`
  - LLM-based escalation → `0.70`

---

## 3. LLM Configuration

The agent supports multiple providers via environment variables, requiring no code changes to switch:

| Provider       | Config                   | Notes                                                                                       |
|----------------|--------------------------|---------------------------------------------------------------------------------------------|
| Ollama (local) | `LLM_PROVIDER=ollama`    | Requires Ollama running at `localhost:11434`. Uses `qwen3.6` or any tag from `ollama list`. |
| Groq           | `LLM_PROVIDER=groq`      | Fast remote inference. Use `llama-3.3-70b-versatile`.                                       |
| Anthropic      | `LLM_PROVIDER=anthropic` | Requires `ANTHROPIC_API_KEY`.                                                               |
| OpenAI         | `LLM_PROVIDER=openai`    | Requires `OPENAI_API_KEY`.                                                                  |

**Why Ollama + Qwen3**: Local inference avoids API costs and rate limits during development. `temperature=0` on all calls maximizes output determinism. Qwen3's chain-of-thought `<think>...</think>` blocks are stripped by `clean_json_response()` before any `json.loads()` call.

---

## 4. Output Schema

```python
OUTPUT_COLUMNS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type",
    "justification", "confidence_score", "source_documents",
    "risk_level", "pii_detected", "language", "actions_taken"
]
```

Constraints:
- `status`: `replied` or `escalated` (lowercase)
- `product_area`: `devplatform`, `claude`, `visa`, or `none`
- `request_type`: `product_issue`, `feature_request`, `bug`, or `invalid`
- `risk_level`: `low`, `medium`, `high`, or `critical`
- `pii_detected`: `true` or `false` (lowercase string)
- `source_documents`: single relative path string or empty string — never a list
- `actions_taken`: JSON array string — `[]` when empty, never null
- `confidence_score`: float 0.60–1.0

---

## 5. Known Limitations and Observed Failure Modes

These were observed during actual pipeline runs on `sample_support_tickets.csv`:

1. **YAML frontmatter leaking into responses** — Fixed by stripping at index build time. Root cause: Markdown corpus files include `---` YAML headers that BM25 indexed and LLM echoed verbatim.

2. **Generic `support.md` returned for all Visa queries** — Fixed by: (a) adding `support.md` at root level to `EXCLUDED_FILES`, and (b) combining Subject + ticket text as BM25 query. Root cause: The root-level `support.md` had the most tokens overall and superficially matched everything.

3. **Escalation actions silently cleared** — Was `if escalate: actions = []` immediately after generation. Fixed by removing those 3 lines. The post-validation `isinstance(actions, list)` check handles malformed action lists without clearing valid ones.

4. **`verify_identity` on irrelevant tickets** — Was triggered on billing address questions and corporate card config requests. Fixed by tightening the system prompt conditions.

5. **BM25 semantic gap**: BM25 misses relevant articles when the ticket uses different vocabulary (e.g. "terminate" vs "cancel"). Mitigated by retrieving top-5 documents.

6. **Model memory constraints**: The `qwen3.6:latest` model (36B parameters) requires ~20 GiB RAM and may fail to load if system memory is occupied by other processes. Configure `LLM_MODEL` to a smaller model (e.g. `llama3:latest` at 4.7 GB) or switch to `LLM_PROVIDER=groq` in `.env` if this occurs.

---

## 6. Self-Assessment — Where the Hidden Test Set Will Challenge This Agent

- **Novel adversarial patterns**: The safety screener uses a fixed LLM prompt; creative adversarial inputs not in the training distribution may evade detection.
- **Cross-domain tickets**: A ticket that legitimately spans multiple product areas (e.g. a Visa payment via the DevPlatform API) will be classified into one area and may miss relevant docs from the other.
- **Non-English tickets**: The BM25 corpus is English-only; non-English tickets get classified and potentially retrieved against English documents, reducing response quality.
- **Highly specialized Visa/DevPlatform terminology**: BM25 keyword matching fails on domain jargon not present in the corpus (e.g. specific Visa program codes or undocumented API error codes).
