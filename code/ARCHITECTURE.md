# Support Triage Agent — Architecture

A multi-domain customer-support triage agent. It reads support tickets, decides
whether each can be **answered automatically** (grounded in a local
documentation corpus) or must be **escalated to a human**, performs any required
**tool calls**, and writes a fully-structured row per ticket for evaluation.

This document is the single source of truth for the design: **what** the system
does, **how** it does it, **why** each architecture/library/model was chosen,
the **trade-offs** taken, and the **operational benchmarks** measured on the
shipped corpus.

---

## 1. What it does (the contract)

|                 |                                                                                                                                                       |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Input**       | `support_tickets/support_tickets.csv` — columns `Issue` (a JSON array of conversation turns, or plain text), `Subject`, `Company`.                    |
| **Knowledge**   | `data/{devplatform,claude,visa}/**/*.md` — 780 markdown support articles.                                                                             |
| **Tools**       | `data/api_specs/internal_tools.json` — 6 callable tools (refund, password reset, account lock, escalate, subscription change, identity verification). |
| **Output**      | `support_tickets/output.csv` — one row per ticket with all 14 columns below.                                                                          |
| **Entry point** | `python code/main.py` (validate with `python code/validate_output.py`).                                                                               |

Output columns (lowercase snake_case, validated by `validate_output.py`):

`issue, subject, company, response, product_area, status, request_type,
justification, confidence_score, source_documents, risk_level, pii_detected,
language, actions_taken`

---

## 2. Design philosophy

Five principles drive every decision below.

| Principle                            | What it means here                                                                                                                                                          | Why                                                                                                                                    |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **Deterministic where it matters**   | Safety rules, PII detection, escalation rules, retrieval scoring, and the confidence score are all deterministic code, not LLM opinions. `temperature=0` on every LLM call. | Reproducible runs; the parts that protect users (safety, PII, compliance) cannot be argued away by a clever prompt or a hallucination. |
| **Rules-first, LLM-for-nuance**      | Hard rules run first and *short-circuit*; the LLM only judges what the rules leave ambiguous.                                                                               | A compliance/safety decision must never depend solely on an LLM's mood.                                                                |
| **Defense in depth against leakage** | De-obfuscation → PII redaction → grounded generation → output validation are independent layers.                                                                            | No single layer is trusted to be perfect; a miss in one is caught by another.                                                          |
| **Graceful degradation**             | Every LLM/JSON/embedding/IO failure has a safe fallback (escalate, mask, or fall back to BM25).                                                                             | A batch of 89 tickets must never crash, and the system must run with or without optional dependencies.                                 |
| **Fail safe, not open**              | When unsure (parse error, no docs, weak retrieval, adversarial), the system **escalates** rather than guessing.                                                             | Over-escalation costs accuracy points; a wrong confident answer or a leak costs trust.                                                 |

---

## 3. High-level pipeline

A **sequential pipeline**, not an agent loop. Each ticket flows through
deterministic stages; LLMs are bounded sub-steps. Up to **4 LLM calls** per
ticket (safety, classification, escalation supervisor, generation), several of
which are skipped by deterministic short-circuits.

```
support_tickets.csv
        │
        ▼
   [PARSE]                      main.py
   Issue JSON -> conversation turns ("User: ... / Agent: ...")
        │
        ▼
   [PII DETECT + REDACT]        pii.py        (deterministic, no LLM)
   Mask emails/phones/SSNs/cards/addresses BEFORE any LLM sees the text.
   redacted_text feeds every LLM stage; pii_detected = (redacted != raw).
        │
        ▼
   [STAGE 1: SAFETY SCREENER]   safety.py
   De-obfuscate (NFKC + strip zero-width, decode base64/hex, flatten
     homoglyphs on the rule-matching copy)
   Deterministic injection rules (override/role-change/exfil/formula/
     multilingual) -> short-circuit to adversarial
   Else hardened LLM screen: prompt injection, social engineering
     (fake or fabricated-prior-agent authority / coercion / urgency for
     elevated access), and out-of-policy assistance (scrape/exfil tooling)
     -> safe|adversarial
   If adversarial -> STOP: write a canned escalation row
        │
        ▼
   [STAGE 3: CLASSIFIER]        classifier.py (1 LLM call -> JSON)
   product_area, request_type (+ fine request_subtype), risk_level, language
   Per-field coercion; output keys whitelisted
        │
        ▼
   [STAGE 4: RETRIEVER]         retriever.py  (deterministic)
   Chunk-level BM25 over the classified area (L3: falls back to all areas
     when area is none/unknown/empty)
   Optional model2vec semantic re-rank, fused 0.6*BM25 + 0.4*cosine
   Relative relevance floor + max chunks/doc
        │
        ▼
   [STAGE 5: ESCALATION GATE]   escalation.py
   Rules-first (whole-word): critical risk | legal terms | human request |
     PII+financial | vague/none | no docs | weak retrieval (coverage)
   Else LLM supervisor (given risk + PII + request subtype; first-word verdict)
        │
        ▼
   [STAGE 6: RESPONSE GENERATOR] generator.py (1 LLM call -> JSON)
   Grounded answer from chunks OR escalation message; tool calls.
   Validates actions vs schema; enforces verify_identity before destructive
     actions; validates citation; backfills empty response; can flip an
     ungrounded "I can't resolve this" reply to escalated.
        │
        ▼
   [STAGE 7: OUTPUT ASSEMBLER]  assembler.py  (deterministic)
   status, reason-specific justification, deterministic confidence,
   JSON-serialized actions -> one validated 14-column row
        │
        ▼
   output.csv   (written concurrently across MAX_WORKERS threads)
```

### Redacted vs. raw text — a key data-flow detail
`main.py` computes `redacted_text = redact_pii(ticket_text)` once, up front, and
feeds **that** to every LLM-facing stage (safety, classifier, escalation
supervisor, generator). The **raw** text is used only for:
- the local **BM25 query** (`Subject + Issue`) — never leaves the machine; raw text preserves full lexical signal; and
- the **`issue` output column** — faithful input passthrough (echoing the user's own ticket is not a new leak).

So raw PII is never sent to an LLM provider and can never appear in the
generated `response`.

---

## 4. Component breakdown

Each stage below: **what** it does, **how**, **why** that approach, and the
**trade-off** accepted.

### 4.1 Multi-provider LLM client — `llm.py`
- **What**: a single `llm.complete(system, user)` over a pluggable provider.
- **How**: provider selected by `LLM_PROVIDER` env var; `temperature=0`
  everywhere; `clean_json_response()` strips `<think>` blocks and Markdown
  fences, then extracts the **first balanced `{…}` object** (string-aware, so
  braces inside values don't break it) — tolerating preamble that weaker models
  add.
- **Providers**: `ollama` (local, default), `groq`, `anthropic`, `openai`.
- **Why**: provider-agnostic so the agent runs locally for free during dev and
  on a fast hosted model for evaluation, with no code change.
- **Trade-off**: a thin abstraction over each SDK rather than a heavy framework
  (e.g. LangChain) — fewer moving parts and dependencies, at the cost of not
  getting framework features we don't need.

### 4.2 PII detection + redaction — `pii.py`
- **What**: detect personal data and **mask it before it reaches any model or
  the output**.
- **How**: pure regex — email, separated phone (3-3-4, optional country code),
  dashed SSN, credit card (with **Luhn** validation to cut false positives),
  street/zip addresses. A single `_scrub()` engine powers both `detect_pii()`
  (bool) and `redact_pii()` (masked text) so the flag can never disagree with
  the masking. Cards/phones keep their **last 4** (`[CARD ****1234]`) so the
  agent can still say "card ending 1234"; emails/SSNs/addresses are fully masked.
- **Why no LLM**: deterministic, zero-latency, and impossible to prompt-inject
  into changing behavior; an LLM can hallucinate or miss PII.
- **Trade-off**: regex recall gaps (undashed SSNs, non-US phone groupings,
  names) are not masked; we accept lower recall to avoid the over-redaction /
  over-escalation that looser patterns cause. Tool parameters that legitimately
  need an identifier receive the **masked** value — intentional (an action
  records intent; a real executor resolves the identifier from the verified
  account).

### 4.3 Safety screener — `safety.py`
- **What**: detect adversarial tickets in three categories — **prompt injection**
  (hijacking instructions), **social engineering** (manipulating the agent into
  granting access / refunds / exceptions via unverifiable authority — *including
  a fabricated prior agent, supervisor, or company representative* — coercion,
  threats, or manufactured urgency), and **out-of-policy assistance** (writing
  scripts to scrape/exfiltrate documentation, or other tasks that violate
  policy) — including obfuscated and non-English ones.
- **How** (defense in depth):
  1. **Normalize** — Unicode NFKC + strip zero-width/control chars (defeats
     full-width look-alikes and `ig⁠no⁠re`-style splitting).
  2. **Decode** — base64 and hex segments that decode to readable text are
     appended for screening (payloads hidden in encodings become visible);
     noise (IDs, ordinary words, card numbers) is discarded.
  3. **Deterministic rules** — high-precision regexes (instruction override,
     role/persona change, data exfiltration, output manipulation, spreadsheet
     formula injection) + a curated multilingual phrase backstop, matched on a
     **homoglyph-flattened, de-accented** copy. A match short-circuits to
     `adversarial`.
  4. **LLM screener** — runs on the de-obfuscated text for novel/nuanced cases;
     it is the primary multilingual detector.
- **Why before retrieval**: screening first prevents an adversarial prompt from
  steering classification/retrieval. Failure mode is safe — a flagged ticket is
  escalated with a canned response, never answered.
- **Trade-off**: deterministic rules are tuned for **precision** (a false
  positive escalates a real ticket, hurting accuracy), so subtle novel attacks
  rely on the LLM. Homoglyph flattening is applied **only** to the rule-matching
  copy, never to the LLM input, so real non-Latin tickets stay intact.

**Adversarial categories at a glance:**

| Category                         | What it covers                                                             | Example signals                                                                                                                                                                                                         |
|----------------------------------|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| (A) **Prompt injection**         | Hijacking the agent's instructions                                         | "ignore previous instructions", role overrides ("you are now DAN"), reveal-system-prompt, output manipulation, exfiltration verbs, spreadsheet-formula injections, encoded payloads (base64/hex/homoglyph/zero-width)   |
| (B) **Social engineering**       | Manipulating the agent to obtain unentitled access, refunds, or exceptions | Unverifiable authority — *including a fabricated prior agent / manager / representative* — coercion or threats, manufactured urgency / "budget is not a constraint", requests to bypass limits or grant elevated access |
| (C) **Out-of-policy assistance** | Misusing the agent's capabilities to violate policy or extract data        | Requests for scripts/code that scrape, exfiltrate, or bulk-extract documentation, content, or user data from this or another service                                                                                    |

### 4.4 Classifier — `classifier.py`
- **What**: assign `product_area`, `request_type`, `risk_level`, `language`.
- **How**: one structured-JSON LLM call; then **each field is normalized
  independently** (`str(value or default)` + allow-list) so one malformed field
  can't discard the rest. A fine `request_subtype` (e.g. `billing`, `privacy`)
  is kept for downstream use, then mapped to the 4 coarse output `request_type`
  values. `language` is normalized to ISO-639-1. Output keys are whitelisted.
- **Why**: the LLM is good at semantic classification; deterministic
  post-processing makes its output safe to consume.
- **Trade-off**: 10 fine categories collapse to 4 coarse ones (the output
  schema) — granularity is preserved internally (`request_subtype`) but not in
  the CSV.

### 4.5 Retriever — `retriever.py`  *(the most load-bearing stage; see §6)*
- **What**: fetch the documentation chunks most relevant to the ticket.
- **How**: chunk-level **BM25** (lexical) over the classified area, optionally
  **re-ranked** by a local `model2vec` embedding (semantic), fused
  `0.6·BM25 + 0.4·cosine`; a relative relevance floor and per-document cap trim
  results; **L3** falls back to all areas when the area is `none`/unknown or
  empty.
- **Why hybrid**: BM25 nails exact terms (error codes, product names); the
  embedder closes pure vocabulary gaps (`terminate`≈`cancel`). See §6 for the
  full rationale and trade-offs.

### 4.6 Escalation gate — `escalation.py`
- **What**: decide escalate vs. reply, and *why*.
- **How**: rules-first, each returning a **reason code**:
  `critical_risk` → `legal_terms` → `human_request` → `pii_financial` →
  `vague_out_of_scope` → `no_docs` → `weak_retrieval`; otherwise an LLM
  supervisor (given risk level, PII flag, and request subtype) decides — its
  prompt **defaults to REPLY whenever the documentation covers the topic**
  (even if not a literal match) and only escalates on four explicit triggers:
  frustration / explicit human request, an action the agent can't perform from
  docs (refund/restore/modify), confirmed bug/outage, or docs that don't
  address the question at all. Risk level and subtype are **context only —
  not standalone escalation triggers**. The one-word verdict is parsed by the
  **first word** (so "reply, no need to escalate" is not misread). Keyword
  matching is **whole-word** (`sue` ≠ `issue`,
  `fee` ≠ `feedback`); legal and human-request keywords are separate lists for
  accurate justifications. **Weak retrieval** = the top chunk covers
  &lt;15% of the ticket's content terms (term coverage, not raw BM25 score,
  which isn't comparable across queries).
- **Why rules-first**: legal/GDPR/fraud and explicit human requests must always
  escalate regardless of LLM opinion.
- **Trade-off**: keyword rules can't read intent ("I'm *not* suing") and may
  over-escalate; safe-failure mode makes that acceptable.

**Escalation reason codes:** each rule (and the LLM/G6 paths) emits a `reason` that `assembler.py` maps to a precise justification.

| Reason                 | Trigger                                                              | Confidence | Justification (excerpt)                                          |
|------------------------|----------------------------------------------------------------------|:----------:|------------------------------------------------------------------|
| *(adversarial)*        | Safety screener flagged the ticket                                   |    0.90    | "Adversarial input detected by the safety screener…"             |
| `critical_risk`        | classifier `risk_level == "critical"`                                |    0.80    | "triaged as critical risk and needs human review"                |
| `legal_terms`          | legal / compliance keyword (whole-word)                              |    0.80    | "legal or compliance language was detected"                      |
| `human_request`        | explicit human / supervisor / manager request                        |    0.80    | "customer explicitly asked to reach a human"                     |
| `pii_financial`        | PII detected together with a financial keyword                       |    0.80    | "personal data combined with a financial request"                |
| `vague_out_of_scope`   | `product_area == "none"` AND <20 words                               |    0.80    | "too vague or out of scope to resolve"                           |
| `no_docs`              | retrieval returns no chunks                                          |    0.80    | "no matching support documentation was found"                    |
| `weak_retrieval`       | top chunk covers <15% of ticket content terms                        |    0.80    | "documentation does not sufficiently cover this request"         |
| `supervisor_llm`       | LLM supervisor returns "escalate" on an ambiguous case               |    0.70    | "exceeds automated capabilities or requires manual verification" |
| `generator_unresolved` | normal-path reply is ungrounded and says it cannot resolve (G6 flip) |    0.70    | "could not be resolved from the available support documentation" |

### 4.7 Response generator — `generator.py`
- **What**: write the customer reply and any tool calls.
- **How**: grounded answer strictly from retrieved chunks (or an escalation
  message); the **full tool schema** is injected. Post-validation:
  **actions** are checked against the schema (unknown tools / missing required
  params dropped); a **destructive** action (refund/reset/lock/subscription)
  without a preceding `verify_identity` gets one **injected**; `source_documents`
  is kept only if it's a **retrieved** doc that exists on disk; an empty
  response is backfilled; an **ungrounded** "I can't resolve this / escalate"
  reply flips the ticket to escalated so `status` matches the text.
- **Why**: the LLM writes fluent grounded prose; deterministic post-validation
  makes its tool use and citations trustworthy.
- **Trade-off**: the full schema in the prompt costs tokens (kept full because
  the descriptions/enums materially improve tool-call accuracy); whether
  `verify_identity`'s *conditions* are met still relies on the LLM, but its
  *presence* before a destructive action is enforced in code.

**Tools at a glance** (from `data/api_specs/internal_tools.json`):

| Tool                  | Destructive | Required parameters                          | Auto-handled                                                    |
|-----------------------|:-----------:|----------------------------------------------|-----------------------------------------------------------------|
| `escalate_to_human`   |      —      | `priority`, `department`, `summary`          | Injected on the escalate path / G6 flip if the model omitted it |
| `verify_identity`     |      —      | `method`, `target`                           | Injected *before* any destructive action if missing             |
| `issue_refund`        |      ✓      | `transaction_id`, `amount`, `reason`         | Identity gate enforced                                          |
| `reset_password`      |      ✓      | `user_email`                                 | Identity gate enforced                                          |
| `lock_account`        |      ✓      | `user_identifier`, `lock_reason`             | Identity gate enforced                                          |
| `modify_subscription` |      ✓      | `user_id`, `action` (optional `target_plan`) | Identity gate enforced                                          |

### 4.8 Output assembler — `assembler.py`
- **What**: produce the final validated row + deterministic confidence.
- **How**: maps status, picks a **reason-specific justification**, computes
  confidence from a fixed ladder, JSON-serializes actions, emits the exact
  lowercase schema. See §8 for the confidence model.
- **Why deterministic confidence**: LLMs self-report `0.99` regardless of
  evidence; a fixed ladder is honest and reproducible.

---

## 5. Techniques used (and why)

| Technique                                                             | Where                        | Why chosen                                                                                                    |
|-----------------------------------------------------------------------|------------------------------|---------------------------------------------------------------------------------------------------------------|
| **BM25 (Okapi)** lexical ranking                                      | retriever                    | Exact-term precision, deterministic, no network/keys.                                                         |
| **Header-aware chunking**                                             | retriever                    | 85% of docs exceed an embedder's window; chunking prevents truncation/dilution and shrinks generator context. |
| **Static embeddings (model2vec) + score fusion**                      | retriever                    | Closes vocabulary-mismatch gaps BM25 can't, while staying CPU-only and deterministic.                         |
| **Index enrichment** (title + breadcrumb + filename slug into tokens) | retriever                    | The filename slug is essentially the question; boosts the right article without polluting displayed content.  |
| **Term-coverage gate**                                                | retriever + escalation       | A query-relative relevance signal (BM25 scores aren't comparable across queries).                             |
| **Unicode NFKC + homoglyph flattening + base64/hex decode**           | safety                       | De-obfuscate before screening so hidden/obfuscated injections are visible.                                    |
| **Whole-word regex matching**                                         | safety, escalation, PII      | Precision — avoids `sue`∈`issue`, `fee`∈`feedback`, `St`∈`St Louis`.                                          |
| **Luhn checksum**                                                     | PII                          | Validates credit-card candidates, cutting false positives.                                                    |
| **PII redaction (format-preserving)**                                 | pii + main                   | Deterministic prevention of leakage to the model and the output.                                              |
| **Rules-first guardrails + LLM tiebreaker**                           | safety, escalation           | Compliance can't be overridden by the LLM; the LLM only handles ambiguity.                                    |
| **Schema-validated tool calls + identity-gate enforcement**           | generator                    | Malformed/hallucinated/destructive-without-verify calls never reach output.                                   |
| **Balanced-brace JSON extraction + per-field coercion**               | llm + classifier + generator | Robust to weak-model preamble and partial/garbled JSON.                                                       |
| **Deterministic confidence ladder**                                   | assembler                    | Reproducible, honest confidence (not LLM self-report).                                                        |
| **Thread-pool concurrency**                                           | main                         | LLM calls are I/O-bound; overlap them across `MAX_WORKERS`.                                                   |
| **Graceful degradation everywhere**                                   | all                          | Never crash a batch; run with or without optional deps.                                                       |

---

## 6. Retrieval deep-dive

Retrieval feeds **both** grounding (generator) and the escalation decision
(no/weak docs → escalate), so its quality propagates everywhere — hence the most
engineering attention.

### 6.1 Why BM25 (over a vector DB) as the base
|                                               | BM25 (`rank_bm25`)                        | Vector DB / pure dense                            |
|-----------------------------------------------|-------------------------------------------|---------------------------------------------------|
| Determinism                                   | Exact term statistics, fully reproducible | Depends on model + ANN index; harder to reproduce |
| Dependencies                                  | One small pure-Python lib                 | Embedding model + (often) a DB/ANN engine         |
| Exact terms (error codes, product names, IDs) | **Strong**                                | Often "smoothed over"                             |
| Vocabulary mismatch (synonyms/paraphrase)     | **Weak**                                  | Strong                                            |
| Cost/latency                                  | In-process, ~ms, no network               | Model load + (sometimes) network                  |

BM25 is the deterministic, dependency-light base; its one real weakness
(vocabulary mismatch) is addressed by the optional semantic re-rank rather than
by replacing it.

### 6.2 Chunking
Docs are split by Markdown headers into passages (headings kept inline so they
survive merges); small sections merge up to ~220 words, sections over ~320 words
window-split with ~30-word overlap. **Why it's necessary here**: median doc is
465 words and **85% exceed ~190 words** — embedding or scoring a whole doc
truncates/dilutes it. Chunking yields focused units and a far smaller generator
context. Each chunk's BM25 tokens are enriched with the doc title, frontmatter
breadcrumbs, and filename slug (not shown to the LLM).

### 6.3 Content cleaning
`clean_content()` strips YAML frontmatter, `## Related Articles` blocks, link
URLs (keeping anchor text), bare URLs, and `_Last updated_` lines — so a doc is
not credited for topics it merely *links to* (a real precision leak we observed),
and YAML never reaches a response.

### 6.4 Semantic re-rank (optional, `model2vec`)
BM25 supplies a top-20 candidate pool; if embeddings are available the query and
candidates are embedded and fused `0.6·BM25_norm + 0.4·max(0, cosine)`, then a
relative floor (drop &lt;0.25× top) and per-doc cap (≤2) apply.

**Why model2vec / `potion-base-8M`** — and why it is *sufficient, not ideal*:

| Option                              | Quality                        | Footprint                  | Latency                       | Determinism       |
|-------------------------------------|--------------------------------|----------------------------|-------------------------------|-------------------|
| **model2vec (chosen)**              | Good (static, distilled)       | ~30 MB, **no torch**       | ~ms, CPU                      | Deterministic     |
| fastembed + bge-small (ONNX)        | Better (contextual bi-encoder) | ~90 MB, no torch           | tens of ms                    | Deterministic     |
| sentence-transformers cross-encoder | Best (pairwise)                | hundreds of MB + **torch** | ~0.3–0.8 s/ticket             | Deterministic     |
| LLM re-rank                         | High                           | none extra                 | a full LLM call + **network** | Non-deterministic |

model2vec is a **static, distilled** embedding (token vectors mean-pooled, no
attention) — that's what makes it tiny, fast, CPU-only and deterministic, and
also why it is *below* contextual models in raw quality. But it is only a
**re-ranker over an already-good BM25 candidate set**, so "sufficient" is
genuinely fine: it just needs to lift synonym matches (`terminate`≈`cancel`,
measured cosine **0.58**) that BM25 can't see. The integration is **pluggable**,
so swapping to fastembed/bge-small or a cross-encoder is a one-function change.
If `model2vec` (or its model) is unavailable, or `DISABLE_EMBEDDINGS` is set, the
retriever runs **pure BM25** — no failure.

**Concrete semantic signals** (model2vec `potion-base-8M`, L2-normalized cosine):

| Query A                           | Query B                |   Cosine | Observation                                                   |
|-----------------------------------|------------------------|---------:|---------------------------------------------------------------|
| "how do I cancel my subscription" | "terminate my account" | **0.58** | Synonymy BM25 cannot see — exactly the gap the re-rank closes |
| "how do I cancel my subscription" | "reset password"       | **0.27** | Unrelated topic — correctly low                               |

### 6.5 L3 — cross-area fallback
Retrieval searches the classified `product_area` first; it falls back to **all
areas** when the area is `none`/unknown or returns nothing, so a misclassified
or `none` ticket isn't blinded. Trap files are excluded from every index at
build, so cross-area search can't resurface them. **Residual**: if a *wrong*
area still has a weak lexical match, the fallback doesn't trigger — the Stage-5
weak-retrieval rule is the backstop. (The heavier "always search all areas with
an in-area boost" prior was deliberately not taken, to avoid cross-area noise on
every query.)

---

## 7. Safety & leakage defense-in-depth

| Threat                                                                                                        | Layer(s) that defend it                                                                                                 |
|---------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Prompt injection / role hijack                                                                                | Safety de-obfuscation + deterministic rules + LLM screener                                                              |
| Encoded/obfuscated injection (base64/hex/homoglyph/zero-width)                                                | Safety normalize + decode + homoglyph flatten                                                                           |
| System-prompt / document exfiltration                                                                         | Safety exfiltration rules; generator answers only from retrieved chunks                                                 |
| Social engineering (fake authority, fabricated prior agent/manager, coercion, urgency for elevated access)    | Safety LLM screener (manipulation category); generator never grants, promises, or validates elevated access / authority |
| Out-of-policy assistance (scraping / exfiltration tooling, "write a script that pulls down all support docs") | Safety LLM screener (out-of-policy category)                                                                            |
| PII sent to a 3rd-party model                                                                                 | PII redaction before every LLM call                                                                                     |
| PII echoed into the output                                                                                    | PII redaction + grounded generation (and the `response` is built from redacted input)                                   |
| Unauthorized destructive action                                                                               | `verify_identity` enforced in code before refund/reset/lock/subscription                                                |
| Hallucinated tool calls / citations                                                                           | Schema validation of actions; citation must be a retrieved doc                                                          |

The failure mode of every safety layer is **escalate / mask / drop**, never
"answer anyway".

---

## 8. Confidence model (`assembler.py`)

Deterministic ladder; reflects confidence in the *correctness of the agent's
output*, with a grounded resolution highest. (`invalid` no longer overrides an
escalation — an escalated ticket always uses the escalation confidence.)

| Outcome                                            | Score                                                                |
|----------------------------------------------------|----------------------------------------------------------------------|
| Clean answer grounded in a retrieved doc           | **0.95**                                                             |
| Adversarial rejected, or `invalid` (not escalated) | **0.90**                                                             |
| Rule-based escalation                              | **0.80**                                                             |
| LLM- or generator-driven escalation                | **0.70**                                                             |
| Replied without a grounded source                  | **0.60–0.70** (penalized for `none` area / non-English / very short) |

---

## 9. Output schema & contract

| Column                          | Source                       | Constraint                                              |
|---------------------------------|------------------------------|---------------------------------------------------------|
| `issue` / `subject` / `company` | Input passthrough            | verbatim                                                |
| `response`                      | Generator (never empty)      | string                                                  |
| `product_area`                  | Classifier                   | `devplatform` / `claude` / `visa` / `none`              |
| `status`                        | Escalation decision          | `replied` / `escalated`                                 |
| `request_type`                  | Classifier (coarse)          | `product_issue` / `feature_request` / `bug` / `invalid` |
| `justification`                 | Assembler (reason-keyed)     | string                                                  |
| `confidence_score`              | Assembler ladder             | float `0.60`–`0.95`                                     |
| `source_documents`              | Generator (validated)        | a retrieved path, or `""`                               |
| `risk_level`                    | Classifier                   | `low` / `medium` / `high` / `critical`                  |
| `pii_detected`                  | `redact_pii` diff            | `true` / `false`                                        |
| `language`                      | Classifier                   | ISO-639-1                                               |
| `actions_taken`                 | Generator (schema-validated) | JSON array string                                       |

---

## 10. Determinism & reproducibility

- `temperature=0` on all LLM calls.
- All gates (safety rules, PII, escalation rules, retrieval scoring, confidence)
  are deterministic code.
- BM25 and model2vec are deterministic given pinned inputs; ranking ties broken
  by stable secondary keys.
- `ThreadPoolExecutor.map` preserves input order, so `output.csv` row order is
  stable.
- Residual nondeterminism comes only from the hosted LLM's own variability on
  the free-text `response` and the rare ambiguous escalation/classification —
  the *structured* decisions around it are fixed.

---

## 11. Benchmarks & operational characteristics

Measured on the shipped corpus and dev machine (CPU). These are **operational**
metrics; a formal accuracy evaluation on the hidden set is pending.

| Metric                        | Value                                                                      |
|-------------------------------|----------------------------------------------------------------------------|
| Corpus                        | 780 docs — devplatform 440, claude 326, visa 14                            |
| Doc length                    | median 465 words, mean 674, p90 1,269, max 21,896; **85% > 190 words**     |
| Chunks indexed                | **3,081** — devplatform 1,855, claude 1,193, visa 33                       |
| Embedding model               | `minishlab/potion-base-8M`, 256-dim, static                                |
| Embedding build (full corpus) | **~1.7 s** on CPU                                                          |
| Embedding matrix size         | ~3,081 × 256 × 4 B ≈ **~3 MB**                                             |
| Semantic signal example       | cosine(`cancel`, `terminate`) ≈ **0.58**; cosine(`cancel`, `reset`) ≈ 0.27 |
| Per-ticket local work         | sub-millisecond BM25 + ~ms embedding (LLM latency dominates)               |
| LLM calls / ticket            | up to 4 (safety, classify, escalation, generate); fewer via short-circuits |
| Tickets                       | 89 (`support_tickets.csv`), processed across `MAX_WORKERS` (default 5)     |
| Test suite                    | **147 tests**, ~0.7 s (LLM mocked; embeddings disabled for determinism)    |

### Observed outcomes on the 89-ticket shipped run

A snapshot of how the pipeline classified the `support_tickets.csv` corpus end-to-end.
Counts sum to **89**; **68 escalated / 21 replied**.

| Outcome bucket                              |  Count | Confidence | Where it comes from                                                   |
|---------------------------------------------|-------:|:----------:|-----------------------------------------------------------------------|
| Replied with cited source (grounded answer) |     19 |    0.95    | Generator normal path; `source_documents` validated against retrieval |
| Replied without source                      |      2 | 0.60–0.70  | Compliments / non-issues with no corpus citation                      |
| **Adversarial** (safety screener)           | **23** |    0.90    | (A) injection + (B) social engineering + (C) out-of-policy            |
| `critical_risk` (Rule 1)                    |      8 |    0.80    | classifier `risk_level == "critical"`                                 |
| `legal_terms` (Rule 2a)                     |      5 |    0.80    | legal / compliance whole-word keyword                                 |
| `human_request` (Rule 2b)                   |      1 |    0.80    | explicit ask for a human/supervisor/manager                           |
| `pii_financial` (Rule 3)                    |      1 |    0.80    | PII detected + financial word                                         |
| `vague_out_of_scope` (Rule 4)               |      3 |    0.80    | `product_area == "none"` AND <20 words                                |
| `weak_retrieval` (Rule 6)                   |      2 |    0.80    | top chunk covers <15% of ticket content terms                         |
| `supervisor_llm` (ambiguous case)           |     23 |    0.70    | LLM supervisor decided escalate after all rules passed                |
| `generator_unresolved` (G6 flip)            |      2 |    0.70    | normal-path reply was ungrounded and said it could not resolve        |
| **Total**                                   | **89** |            |                                                                       |

### Pipeline behaviour across iterations

The supervisor prompt and safety screener were tightened iteratively during
development. Each row is a full run on the same 89-ticket `support_tickets.csv`.

| Iteration                                | Change introduced                                                                                                                             | Adversarial | Supervisor-LLM esc. | Grounded replies (0.95) | Total replied | Total escalated |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------:|--------------------:|------------------------:|--------------:|----------------:|
| 1. Initial hardened pipeline             | Stages 1–7 hardened (chunking, embeddings, L3, PII redaction, escalation rules, generator guards) — supervisor still biased toward escalation |          20 |                  35 |                      13 |            14 |              75 |
| 2. Supervisor "default to reply" rewrite | §4.6 prompt tightened: default to reply when docs cover the topic; four narrow escalate triggers; risk / subtype demoted to context           |          20 |                  26 |                      20 |            21 |              68 |
| 3. Safety (B) + (C) extensions           | Safety screener catches *fabricated prior-agent* claims and *out-of-policy / scrape-tooling* requests                                         |          23 |                  23 |                      19 |            21 |              68 |

Net effect across iterations: **+7 grounded replies**, **−12 supervisor-LLM escalations**, **+3 adversarial flags** (all on tickets that should be flagged — `#2` / `#45` / `#54`), with the escalated/replied totals stable.

---

## 12. Configuration

| Env var                                                 | Default                    | Purpose                                    |
|---------------------------------------------------------|----------------------------|--------------------------------------------|
| `LLM_PROVIDER`                                          | `ollama`                   | `ollama` / `groq` / `anthropic` / `openai` |
| `LLM_MODEL`                                             | —                          | model id for the provider                  |
| `GROQ_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | —                          | provider keys                              |
| `LOCAL_LLM_URL` / `LOCAL_LLM_KEY`                       | `localhost:11434/v1`       | Ollama endpoint                            |
| `MAX_WORKERS`                                           | `5`                        | concurrency for ticket processing          |
| `EMBED_MODEL`                                           | `minishlab/potion-base-8M` | semantic re-rank model                     |
| `DISABLE_EMBEDDINGS`                                    | unset                      | set to force pure-BM25 retrieval           |

---

## 13. Testing strategy

- **147 tests** across the seven pipeline stages plus LLM/pipeline glue.
- The LLM is **mocked** (`conftest.py`) so tests are fast, offline, and
  deterministic; embeddings are disabled in the suite and exercised by their own
  unit tests.

**Tests by stage:**

| File                 |   Tests | Coverage focus                                                                                                                                                             |
|----------------------|--------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `test_safety.py`     |      31 | Obfuscation (zero-width / full-width / homoglyph / base64 / hex); injection rules; multilingual backstop; social-engineering (B); out-of-policy (C); legit-input negatives |
| `test_retriever.py`  |      24 | Tokenizer, content cleaning, header-aware chunking (merge / split / hierarchy), enrichment, relative floor, L3 cross-area, fusion + embedding wrapper                      |
| `test_pii.py`        |      18 | Detection + redaction (email / SSN / phone / card + Luhn / address), false-positive negatives, idempotence, detect ↔ redact agreement                                      |
| `test_escalation.py` |      17 | Whole-word rules, first-word verdict, reason codes, weak-retrieval, request-subtype context, supervisor "default to reply"                                                 |
| `test_generator.py`  |      15 | Action validation vs schema, citation membership + disk, escalate-action injection, empty-response backfill, G6 flip, `verify_identity` enforcement                        |
| `test_classifier.py` |      14 | Per-field coercion, preamble robustness, ISO language normalization, fine → coarse mapping, `request_subtype`, garbage fallback                                            |
| `test_assembler.py`  |      12 | Confidence ladder (incl. escalated-`invalid` A1 fix), reason-keyed justification, row shape + enums, adversarial overrides                                                 |
| `test_pipeline.py`   |       9 | End-to-end on `sample_support_tickets.csv`, adversarial routing, out-of-scope routing                                                                                      |
| `test_llm.py`        |       7 | `clean_json_response` (think-tag / fences / preamble / braces-in-strings), provider initialisation paths                                                                   |
| **Total**            | **147** | ~0.7 s; LLM mocked; embeddings disabled                                                                                                                                    |

---

## 14. Known limitations & failure modes

| # | Limitation                                                                                                | Impact                                                                                                                                                                 | Mitigation / status                                                                            |
|--:|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | No end-to-end accuracy benchmark yet                                                                      | Confidence in real-world accuracy is unverified beyond the 147 unit/integration tests + structural validation                                                          | Re-run on hidden test set and iterate on observed misses                                       |
| 2 | Provider gap — Google/Gemini is referenced in `.env.example` & README but **not wired** in `llm.py`       | `LLM_PROVIDER=google` would raise `ValueError`                                                                                                                         | Documented; trivial to add a branch when/if needed                                             |
| 3 | `model2vec` is *sufficient, not ideal* — static embeddings trail contextual bi-/cross-encoders            | Some synonyms still slip past the re-rank                                                                                                                              | Pluggable: one-function swap to `fastembed`+`bge-small` (ONNX, no torch) or a cross-encoder    |
| 4 | **L3 residual** — a wrong-area ticket with a *weak* lexical match doesn't trigger the cross-area fallback | Misclassified-area tickets may return marginal in-area docs                                                                                                            | Stage-5 weak-retrieval rule (top chunk covers <15% of ticket terms) is the backstop            |
| 5 | **PII recall gaps** — undashed SSNs, non-US phone groupings, names                                        | Those values aren't redacted; can reach the LLM and the output                                                                                                         | Deliberate precision/recall trade-off (loose patterns over-redact); opt-in extensions possible |
| 6 | Keyword rules can't read intent ("I'm *not* going to sue" still trips the legal rule)                     | Some legitimate tickets over-escalate                                                                                                                                  | Safe-failure (escalation, not refusal) keeps it harmless                                       |
| 7 | Cross-domain tickets — a ticket spanning two product areas is classified into one                         | May miss relevant docs from the other corpus                                                                                                                           | L3 fallback + weak-retrieval rule partially compensate                                         |
| 8 | Non-English answers — the corpus is English-only                                                          | Lower answer quality for non-English tickets even when classified/retrieved correctly                                                                                  | A multilingual embedder could close some of the gap (deferred)                                 |
| 9 | LLM non-determinism on the supervisor's borderline cases                                                  | A few borderline tickets can flip reply↔escalate between runs (we observed `#8` "none of the submissions working" flip between troubleshoot-reply and outage-escalate) | `temperature=0` minimizes the variance; structural decisions around the LLM call are fixed     |
