# UroAgent — Autonomous Clinical AI Orchestrator

[![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-ReAct%20Agent-1C3C3C)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A ReAct-based multi-agent system that autonomously answers clinical questions
about urology patients by reasoning over **structured EHR data** and
**unstructured clinical guidelines** in a single pipeline — built to
demonstrate agentic AI orchestration, tool-calling, and RAG grounding for
healthcare applications.

## What is UroAgent?

UroAgent is a **ReAct (Reason + Act) AI agent** that decides, on its own,
which of two specialized tools to call, in what order, and how many times,
in order to answer a natural-language clinical question:

| Tool | Data Type | Technology |
|---|---|---|
| `query_ehr_database` | Structured patient records (PSA, staging, treatment) | SQLite + LangChain SQL tool |
| `query_aua_guidelines` | Unstructured AUA clinical guidelines | FAISS vector store + RAG |

## Architecture

```
User Clinical Query
        |
        v
+-----------------------------------+
|   LangChain ReAct Agent           |
|   Model: Llama 3.3 70B (Groq)     |
|                                    |
|  Thought -> Action -> Observation |   <- ReAct loop
|       ^______________|            |
+------------+-----------------------+
             |
      +------+-------+
      v              v
+----------+   +----------------+
| SQLite   |   |  FAISS Index   |
| EHR DB   |   |  AUA Guidelines|
+----------+   +----------------+
      |              |
      +------+-------+
             v
   Synthesized Clinical Response
```

## Key Features

- **Agentic workflow** — autonomous multi-step reasoning via the ReAct loop; no hand-coded if/else routing.
- **Multi-modal data fusion** — combines structured SQL queries with unstructured RAG retrieval in a single agent.
- **Evidence-based answers** — every recommendation is grounded in mock AUA 2022/2023 clinical guidelines (PSA screening, staging, BPH, RCC, bladder/testicular cancer).
- **Tool-calling framework** — LangChain `@tool`-decorated functions with typed signatures and docstrings that guide LLM tool selection.
- **Streamlit UI** — interactive chat interface with visible agent reasoning steps for interpretability.
- **Safety guardrails** — the SQL tool rejects any non-`SELECT` statement, and the RAG index is built entirely from local, synthetic data.

## Project Structure

```
uroagent/
├── setup_db.py          # Creates & seeds the mock SQLite EHR database
├── setup_rag.py          # Embeds AUA guideline chunks -> FAISS index
├── agent.py              # Core ReAct agent: LLM + tool orchestration
├── app.py                # Streamlit web UI
├── tools/
│   ├── __init__.py
│   ├── sql_tool.py       # LangChain Tool: SQL EHR queries
│   └── rag_tool.py       # LangChain Tool: AUA guideline RAG
├── data/                 # Auto-generated, gitignored
│   ├── patient_records.db
│   └── aua_guidelines_index/
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/bsevakstark/UROagent.git
cd UROagent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your Groq API key

UroAgent runs on Groq's hosted Llama 3.3 70B (fast, free-tier friendly).
Get a key at [console.groq.com/keys](https://console.groq.com/keys).

```bash
cp .env.example .env
# Edit .env and add your key:  GROQ_API_KEY=gsk-...
```

### 3. Build the data assets (one-time)

```bash
python setup_db.py     # Creates data/patient_records.db
python setup_rag.py    # Creates data/aua_guidelines_index/
```

### 4. Launch

```bash
streamlit run app.py
```

Or run the agent directly in your terminal:

```bash
python agent.py
```

## Example Queries

| Query | Tools Used |
|---|---|
| "Which patients have a PSA level above 10 ng/mL?" | `query_ehr_database` |
| "What are the AUA recommendations for active surveillance?" | `query_aua_guidelines` |
| "Patient 3 has T3b / Gleason 4+4=8 — what does the guideline recommend?" | Both tools |
| "List all prostate cancer patients and their current treatments." | `query_ehr_database` |
| "What is the standard of care for muscle-invasive bladder cancer?" | `query_aua_guidelines` |

## Clinical Data Coverage

**EHR database (10 mock patients).** Diagnoses: prostate cancer, BPH, renal
cell carcinoma, bladder cancer, testicular cancer. Fields: `patient_id`,
`age`, `diagnosis`, `psa_level`, `tumor_stage`, `biopsy_gleason_score`,
`treatment`, `follow_up_months`.

**AUA guideline knowledge base (12 indexed chunks):** PSA screening &
thresholds, prostate cancer staging (AJCC 8th ed.), Gleason score / grade
groups, active surveillance, radical prostatectomy & radiation therapy,
androgen deprivation therapy, renal cell carcinoma, bladder cancer, BPH
management, testicular cancer.

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangChain (ReAct) |
| LLM | Groq — Llama 3.3 70B Versatile |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, no API key needed) |
| Vector store | FAISS |
| Structured DB | SQLite3 |
| UI | Streamlit |
| Language | Python 3.11+ |

## Disclaimer

This is a **portfolio / research project**. All patient data is entirely
synthetic and randomly generated. Clinical guideline content is simplified
for demonstration purposes. This tool is **not** intended for real clinical
decision-making.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built by [Bhavya Sevak](https://github.com/bsevakstark) — Biomedical Informatics M.S. candidate, Arizona State University.
