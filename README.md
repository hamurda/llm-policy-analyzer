# GDPR Policy Analyzer

A Gradio app that evaluates privacy policies against 12 key GDPR articles and returns a color-coded compliance table with verdicts, rationale, and citations.

## How It Works

```
Privacy Policy Text
        │
        ▼
┌─────────────────────┐
│  Semantic Matching  │  sentence-transformers + pre-computed embeddings
│  Policy ↔ GDPR Arts │  cosine similarity against 12 articles
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  GPT-5-Mini (×12)   │  one structured call per article
│  Pydantic schema    │  → Covered / Partial / Not Observed
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Gradio UI          │  color-coded table + summary score
└─────────────────────┘
```

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/gdpr-policy-analyzer.git
cd gdpr-policy-analyzer
pip install -r requirements.txt

cp .env.example .env # Set your OpenAI API key

python app.py
```

Open `http://localhost:7860` in your browser. Click **Notion Privacy Policy** to load an example, then hit **Analyze**.

![GDPR Policy Analyzer](docs/screenshot.png)

## Rebuilding the Knowledge Base

The `*_embedding.json` files are checked in so the app works out of the box. If you modify the raw GDPR rules or checklist, regenerate them:

```bash
python scripts/build_embeddings.py
```

This encodes each rule/clause with `all-MiniLM-L6-v2` and writes the embeddings alongside the original JSON.

## Why it's built this way

- **No vector DB.** First version used ChromaDB + LangChain. Replaced with pre-computed embeddings and in-memory cosine similarity, same retrieval quality, zero infrastructure.
- **One LLM call per article.** Original made ~50+ calls (per-subpoint + rollup). Single prompt with all checklist questions inlined is faster, cheaper, and more consistent.
- **Structured output** with Pydantic schema guarantees parseable JSON. Fallback dict on failure, exponential backoff on retries.
- **Retry with backoff** — exponential backoff on LLM failures (rate limits, transient errors).


Tuning: `config.py` controls which articles to evaluate, similarity threshold, and model choice.