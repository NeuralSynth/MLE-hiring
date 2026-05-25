# Multi-Domain Support Triage Agent

This is a terminal-based AI agent that reads support tickets from `support_tickets/support_tickets.csv` and produces answers/routing in `support_tickets/output.csv` grounded in a local corpus of support documentation.

---

## 1. Setup

### Install Dependencies
First, ensure you have Python 3.10+ installed. Install the third-party libraries required:

```bash
pip install -r code/requirements.txt
```

### Environment Configuration
Copy the `.env.example` file in the root to `.env`:

```bash
cp .env.example .env
```

Open `.env` and fill in your API key for your chosen provider. For example, to run the agent with OpenAI:

```env
OPENAI_API_KEY=your-openai-api-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

Other supported providers are:
- **Groq** (`LLM_PROVIDER=groq`, `LLM_MODEL=llama-3.3-70b-versatile`, `GROQ_API_KEY`)
- **Anthropic** (`LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-3-5-sonnet-20241022`, `ANTHROPIC_API_KEY`)
- **Google** (`LLM_PROVIDER=google`, `LLM_MODEL=gemini-2.5-flash`, `GOOGLE_API_KEY`)

---

## 2. Run the Agent

Execute the primary driver script from the repository root:

```bash
python code/main.py
```

This will:
1. Index all local help documents inside `data/` using BM25.
2. Read and parse tickets from `support_tickets/support_tickets.csv`.
3. Sequentially run safety screening, PII checks, metadata classification, retrieval, escalation gating, response generation, and output assembly.
4. Concurrently process tickets using a worker thread pool.
5. Write the final outputs conforming to the required schema to `support_tickets/output.csv`.

---

## 3. Formats Validation

To check that the generated `output.csv` complies structurally with the required column headers, row counts, and enums:

```bash
python code/validate_output.py
```
