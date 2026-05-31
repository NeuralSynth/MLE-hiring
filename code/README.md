# Multi-Domain Support Triage Agent

A terminal agent for the MLE hiring challenge. It reads support tickets
from `support_tickets/support_tickets.csv` and writes a fully-structured
row per ticket to `support_tickets/output.csv`, grounded in the local
documentation corpus under `data/`.

For each ticket, the pipeline runs seven stages:

- **PII redaction** — regex masking before any LLM sees the text
- **Safety screen** — prompt-injection / social-engineering / out-of-policy detection
- **Classification** — `product_area`, `request_type`, `risk_level`, `language`
- **Retrieval** — chunk-level BM25 over the classified area, with an optional `model2vec` semantic re-rank
- **Escalation gate** — seven deterministic rules, then an LLM supervisor for the rest
- **Response generation** — grounded reply or neutral escalation message, with schema-validated tool calls
- **Output assembly** — deterministic confidence, validated citations, schema-compliant CSV row

Everything that has to be right is deterministic code. The LLM is used
for what it's good at: classification, fluent grounded prose, and
borderline triage. Up to 4 LLM calls per ticket, with `temperature=0`
everywhere.

See **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** for the design, rationale,
trade-offs, benchmarks, and self-assessment.

---

## 1. Requirements

- **Python 3.10+**
- Install dependencies:

```bash
pip install -r code/requirements.txt
```

`model2vec` is **optional** — it powers a local semantic re-rank in the
retriever. If it (or its model) is unavailable, retrieval falls back to
pure BM25. The pipeline always runs.

---

## 2. Configure the LLM provider

Copy the example env file (at the **repo root**) to `.env`:

```bash
cp .env.example .env          # macOS / Linux
copy .env.example .env        # Windows
```

Set **one** provider:

| Provider | `.env` settings | Example `LLM_MODEL` | Notes |
|---|---|---|---|
| **Ollama** (local, default) | `LLM_PROVIDER=ollama`<br>`LOCAL_LLM_URL=http://localhost:11434/v1` | `llama3:latest` (must match `ollama list`) | No API key; runs fully offline; slow. |
| **Groq** | `LLM_PROVIDER=groq`<br>`GROQ_API_KEY=...` | `llama-3.3-70b-versatile` | Fast hosted inference — recommended for a full run. |
| **Anthropic** | `LLM_PROVIDER=anthropic`<br>`ANTHROPIC_API_KEY=...` | `claude-haiku-4-5` | Explicit `cache_control` prompt caching enabled. |
| **OpenAI** | `LLM_PROVIDER=openai`<br>`OPENAI_API_KEY=...` | `gpt-4o-mini` | Automatic prompt caching + `prompt_cache_key` routing. |
| **Gemini** | `LLM_PROVIDER=gemini`<br>`GOOGLE_API_KEY=...` | `gemini-2.5-flash` | Uses `google-genai` SDK; implicit caching on 2.5+ models. |

Example `.env` for Groq:

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=your-groq-api-key-here
```

> Set `LLM_PROVIDER` explicitly — leaving it blank doesn't fall back to
> a default. Read keys from `.env`; never hardcode them.

---

## 3. Run the agent

From the **repository root**:

```bash
python code/main.py
```

This will:

1. Build the chunk-level BM25 index over `data/` (and, if `model2vec` is installed, the embedding matrix for the semantic re-rank).
2. Read and parse tickets from `support_tickets/support_tickets.csv`.
3. For each ticket: PII redaction → safety screen → classification → retrieval → escalation decision → response/tool generation → output assembly.
4. Process tickets concurrently (`MAX_WORKERS`, default 8 — see [Performance](#8-performance--worker-count)).
5. Stream each completed row to `support_tickets/output.partial.csv` as a checkpoint; on clean completion, sort by input order and write `support_tickets/output.csv`.

> **Resume on restart.** If the run is interrupted (Ctrl+C, OOM,
> provider 5xx, OS kill), re-running `python code/main.py` reads the
> partial file and continues from the next unprocessed ticket. Use
> `FORCE_RESTART=1 python code/main.py` to discard the partial and
> start fresh. See [`ARCHITECTURE.md` §18](./ARCHITECTURE.md#18-checkpoint--resume)
> for details.

---

## 4. Validate the output format

```bash
python code/validate_output.py
```

Checks `output.csv` for column / row / enum compliance (structure only,
not correctness). Passing validation is necessary but not sufficient.

---

## 5. Run the tests

```bash
python -m pytest code/tests -q
```

**203 tests** covering every stage, ~3 s. The LLM is mocked and
embeddings are disabled in the suite (via `conftest.py`), so the suite
runs **offline, fast, and deterministically** — no API key or model
download required.

---

## 6. Optional: local semantic re-rank (`model2vec`)

The retriever combines BM25 with an optional local embedding re-rank
(`0.6·BM25 + 0.4·cosine`) to close vocabulary-mismatch gaps (e.g.
`terminate` ≈ `cancel`).

- **Auto-enabled** when `model2vec` is installed. The model (`minishlab/potion-base-8M`, ~30 MB) downloads from Hugging Face on first run, then caches. CPU-only, deterministic, adds ~1.7 s of one-time index build for the full corpus.
- **Disable it** (force pure BM25 — e.g. for fully offline runs):

```bash
DISABLE_EMBEDDINGS=1 python code/main.py          # macOS / Linux
$env:DISABLE_EMBEDDINGS=1; python code/main.py    # Windows (PowerShell)
```

- **Swap the model** via `EMBED_MODEL=<hf-model-id>`.

If the library or model can't load, retrieval silently falls back to
BM25 — the agent always runs.

---

## 7. Configuration reference

| Env var | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` / `groq` / `anthropic` / `openai` / `gemini` |
| `LLM_MODEL` | — | model id for the chosen provider |
| `GROQ_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` | — | provider API keys |
| `LOCAL_LLM_URL` / `LOCAL_LLM_KEY` | `http://localhost:11434/v1` | Ollama endpoint / dummy key |
| `MAX_WORKERS` | `8` | concurrent ticket workers (recommended minimum — see [Performance](#8-performance--worker-count)) |
| `EMBED_MODEL` | `minishlab/potion-base-8M` | semantic re-rank model |
| `DISABLE_EMBEDDINGS` | unset | set to force pure-BM25 retrieval |
| `FORCE_RESTART` | unset | set to `1` to discard `output.partial.csv` and re-process every ticket |

---

## 8. Performance & worker count

The pipeline must process the full ticket set within a **3-minute** limit.
Per-ticket local work is sub-millisecond, so wall-clock is dominated by LLM
round-trips and the right `MAX_WORKERS` is whatever keeps the LLM backend
busy without overloading it.

We benchmarked eight worker counts against the **89-ticket**
`support_tickets.csv` on the evaluated LLM backend:

| `MAX_WORKERS` | 89-ticket wall-clock | Within 3-min limit? |
|--------------:|----------------------|:-------------------:|
| 2 / 3 / 5     | ≥ 5 min              | ✗                   |
| **8**         | **0:02:32.461436**   | **✓**               |
| 10 / 12 / 15 / 18 | ≥ 5 min          | ✗                   |

**`MAX_WORKERS=8` is the recommended minimum and the default.** It is the
only value that finished within the limit — **0:02:32.461436** for all 89
tickets, ~27 s of headroom under 3 minutes. Fewer workers under-utilize
concurrency; more than 8 degrade throughput because the LLM backend
serializes concurrent requests. Every other count we tested was ≥ 5 minutes.
These numbers are specific to the benchmarked provider/model — re-sweep if
you change backends. See `ARCHITECTURE.md` §13 for the full table.

---

## 9. Layout

```
code/
├── main.py            # entry point / orchestration / checkpoint
├── llm.py             # multi-provider LLM client + prompt caching + JSON cleaning
├── pii.py             # PII detection + redaction (regex + Luhn)
├── safety.py          # prompt-injection screener (de-obfuscate + rules + LLM)
├── classifier.py      # product_area / request_type / risk / language
├── retriever.py       # chunked BM25 + optional model2vec re-rank + L3 fallback
├── escalation.py      # rules-first escalation gate + LLM supervisor
├── generator.py       # grounded response + schema-validated tool calls
├── assembler.py       # final row + continuous confidence ladder
├── validate_output.py # output format validator
├── config.py          # paths, constants, keyword lists
├── tests/             # 203 tests (LLM mocked, offline)
├── ARCHITECTURE.md    # full design, trade-offs, benchmarks, self-assessment
└── README.md          # this file
```
