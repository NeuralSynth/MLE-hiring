# Multi-Domain Support Triage Agent

A terminal AI agent that reads support tickets from
`support_tickets/support_tickets.csv` and writes fully-structured answers /
routing decisions to `support_tickets/output.csv`, grounded in a local corpus of
support documentation (`data/`).

For each ticket it screens for prompt injection, redacts PII, classifies the
request, retrieves the most relevant documentation, decides **reply vs.
escalate**, performs any required tool calls, and computes a confidence score —
all deterministically where it matters (`temperature=0`, rule-based guardrails).

See **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** for the full design, rationale,
trade-offs, and benchmarks.

---

## 1. Requirements

- **Python 3.10+**
- Install dependencies:

```bash
pip install -r code/requirements.txt
```

> `model2vec` is **optional** — it powers a local semantic re-rank in the
> retriever. If it (or its model) is unavailable, retrieval gracefully falls
> back to pure BM25. See §6.

---

## 2. Configure the LLM provider

Copy the example env file (at the **repo root**) to `.env`:

```bash
cp .env.example .env          # macOS / Linux
copy .env.example .env        # Windows
```

Then set **one** provider in `.env`. Supported providers:

| Provider | `.env` settings | Example `LLM_MODEL` | Notes |
|---|---|---|---|
| **Ollama** (local, default) | `LLM_PROVIDER=ollama`<br>`LOCAL_LLM_URL=http://localhost:11434/v1` | `llama3:latest` (must match `ollama list`) | No API key; runs fully offline. |
| **Groq** | `LLM_PROVIDER=groq`<br>`GROQ_API_KEY=...` | `llama-3.3-70b-versatile` | Fast hosted inference — recommended for a quick full run. |
| **Anthropic** | `LLM_PROVIDER=anthropic`<br>`ANTHROPIC_API_KEY=...` | `claude-3-5-sonnet-20241022` | |
| **OpenAI** | `LLM_PROVIDER=openai`<br>`OPENAI_API_KEY=...` | `gpt-4o-mini` | |
| **Gemini** | `LLM_PROVIDER=gemini`<br>`GOOGLE_API_KEY=...` | `gemini-2.5-flash` | Uses `google-genai` SDK; implicit prompt caching on 2.5+ models. |

Example `.env` for Groq:

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=your-groq-api-key-here
```

> **Note:** set `LLM_PROVIDER` explicitly — leaving it blank is treated as an
> empty provider, not the default. Read keys from `.env` only; never hardcode.

---

## 3. Run the agent

From the **repository root**:

```bash
python code/main.py
```

This will:
1. Build the chunk-level BM25 index over `data/` (and, if `model2vec` is
   installed, an embedding matrix for the semantic re-rank).
2. Read and parse tickets from `support_tickets/support_tickets.csv`.
3. For each ticket: PII redaction → safety screen → classification → retrieval →
   escalation decision → response/tool generation → output assembly.
4. Process tickets concurrently (`MAX_WORKERS`, default 5).
5. Stream each completed row to `support_tickets/output.partial.csv` as a
   checkpoint; on clean completion, sort by input order and write
   `support_tickets/output.csv`.

> **Resume on restart.** If the run is interrupted (`Ctrl+C`, OOM, provider
> 5xx, OS kill), re-running `python code/main.py` reads the partial file
> and continues from the next unprocessed ticket. Use `FORCE_RESTART=1
> python code/main.py` to discard the partial and start fresh. See
> [`ARCHITECTURE.md` §17](./ARCHITECTURE.md#17-checkpoint--resume) for details.

---

## 4. Validate the output format

```bash
python code/validate_output.py
```

This checks `output.csv` for column/row/enum compliance (structure only, not
correctness).

---

## 5. Run the tests

```bash
python -m pytest code/tests -q
```

138 tests covering every stage. The LLM is mocked and embeddings are disabled in
the suite, so it runs **offline, fast, and deterministically** — no API key or
model download required.

---

## 6. Optional: local semantic re-rank (`model2vec`)

The retriever combines BM25 with an optional local embedding re-rank
(`0.6·BM25 + 0.4·cosine`) to close vocabulary-mismatch gaps (e.g.
`terminate` ≈ `cancel`).

- **Auto-enabled** when `model2vec` is installed. The model
  (`minishlab/potion-base-8M`, ~30 MB) is downloaded from Hugging Face on the
  first run, then cached. It is CPU-only, deterministic, and adds ~1.7 s of
  one-time index build for the full corpus.
- **Disable it** (force pure BM25 — e.g. for fully offline/no-download runs):

```bash
DISABLE_EMBEDDINGS=1 python code/main.py          # macOS / Linux
$env:DISABLE_EMBEDDINGS=1; python code/main.py    # Windows (PowerShell)
```

- **Swap the model** via `EMBED_MODEL=<hf-model-id>`.

If the library or model can't load, retrieval silently falls back to BM25 — the
agent always runs.

---

## 7. Configuration reference

| Env var | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` / `groq` / `anthropic` / `openai` / `gemini` |
| `LLM_MODEL` | — | model id for the chosen provider |
| `GROQ_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` | — | provider API keys |
| `LOCAL_LLM_URL` / `LOCAL_LLM_KEY` | `http://localhost:11434/v1` | Ollama endpoint / dummy key |
| `MAX_WORKERS` | `5` | concurrent ticket workers |
| `EMBED_MODEL` | `minishlab/potion-base-8M` | semantic re-rank model |
| `DISABLE_EMBEDDINGS` | unset | set to force pure-BM25 retrieval |
| `FORCE_RESTART` | unset | set to `1` to discard `output.partial.csv` and re-process every ticket |

---

## 8. Layout

```
code/
├── main.py            # entry point / orchestration
├── llm.py             # multi-provider LLM client + JSON cleaning
├── pii.py             # PII detection + redaction (regex + Luhn)
├── safety.py          # prompt-injection screener (de-obfuscate + rules + LLM)
├── classifier.py      # product_area / request_type / risk / language
├── retriever.py       # chunked BM25 + optional model2vec re-rank + L3 fallback
├── escalation.py      # rules-first escalation gate + LLM supervisor
├── generator.py       # grounded response + schema-validated tool calls
├── assembler.py       # final row + deterministic confidence
├── validate_output.py # output format validator
├── config.py          # paths, constants, keyword lists
├── tests/             # 138 tests (LLM mocked, offline)
├── ARCHITECTURE.md    # full design, trade-offs, benchmarks
└── README.md          # this file
```
