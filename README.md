# 🏥 UroAgent — Autonomous Clinical AI for Urology

> **A multi-modal agentic AI system** that supports clinical operations at the intersection of structured EHR data and evidence-based medical guidelines — built as a portfolio demonstration for the *Associate Data Scientist* role at the **USC Institute of Urology**.

---

## 🧠 What Is UroAgent?

UroAgent is a **ReAct (Reason + Act) AI agent** that autonomously answers clinical questions about urology patients by orchestrating two specialised tools:

| Tool | Data Type | Technology |
|------|-----------|------------|
| `query_ehr_database` | Structured patient records (PSA, staging, treatment) | SQLite + LangChain SQL Tool |
| `query_aua_guidelines` | Unstructured AUA clinical guidelines | FAISS vector store + RAG |

The agent decides **which tool(s) to call**, **in what order**, and **how many times** — all from a single natural-language clinical question.

---

## 🏗️ Architecture

```
User Clinical Query
        │
        ▼
┌──────────────────────────────────┐
│   LangChain ReAct Agent          │
│   Model: gpt-4o-mini (temp=0)    │
│                                  │
│  Thought → Action → Observation  │  ← ReAct Loop
│       └──────────────┘           │
└──────────┬───────────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────┐
│ SQLite  │  │  FAISS Index │
│  EHR   │  │ AUA Guidelines│
│   DB    │  │  (RAG / NLP) │
└─────────┘  └──────────────┘
     │            │
     └─────┬──────┘
           ▼
   Synthesised Clinical Response
```

---

## ✨ Key Features

- **Agentic Workflow** — Autonomous multi-step reasoning with the ReAct loop; no hand-coded if/else logic.
- **Multi-Modal Data Fusion** — Combines structured SQL queries with unstructured RAG retrieval in a single pipeline.
- **Evidence-Based Answers** — Every recommendation is grounded in mock AUA 2022/2023 clinical guidelines (PSA screening, staging, BPH, RCC, bladder/testicular cancer).
- **Tool-Calling Framework** — LangChain `@tool` decorators with typed signatures and rich docstrings that guide LLM tool selection.
- **Streamlit UI** — Interactive chat interface with visible agent reasoning steps for interpretability.
- **Safety First** — SQL tool rejects non-SELECT statements; RAG index is locally persisted and version-controlled independently of patient data.

---

## 🗂️ Project Structure

```
uroagent/
├── setup_db.py          # Creates & seeds the mock SQLite EHR database
├── setup_rag.py         # Embeds AUA guideline chunks → FAISS index
├── agent.py             # Core ReAct agent: LLM + tool orchestration
├── app.py               # Streamlit web UI
├── tools/
│   ├── __init__.py
│   ├── sql_tool.py      # LangChain Tool: SQL EHR queries
│   └── rag_tool.py      # LangChain Tool: AUA guideline RAG
├── data/                # Auto-generated (gitignored)
│   ├── patient_records.db
│   └── aua_guidelines_index/
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/uroagent.git
cd uroagent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your key:  OPENAI_API_KEY=sk-...
```

### 3. Build the data assets (one-time)

```bash
python setup_db.py     # Creates data/patient_records.db
python setup_rag.py    # Creates data/aua_guidelines_index/
```

### 4. Launch the Streamlit UI

```bash
streamlit run app.py
```

Or run the agent directly in your terminal:

```bash
python agent.py
```

---

## 💬 Example Queries

| Query | Tools Used |
|-------|------------|
| *"Which patients have a PSA level above 10 ng/mL?"* | `query_ehr_database` |
| *"What are the AUA recommendations for active surveillance?"* | `query_aua_guidelines` |
| *"Patient 3 has T3b / Gleason 4+4=8 — what does the guideline recommend?"* | Both tools |
| *"List all prostate cancer patients and their current treatments."* | `query_ehr_database` |
| *"What is the standard of care for muscle-invasive bladder cancer?"* | `query_aua_guidelines` |

---

## 🔬 Clinical Data Coverage

### EHR Database (10 mock patients)
Diagnoses: Prostate Cancer · BPH · Renal Cell Carcinoma · Bladder Cancer · Testicular Cancer  
Fields: `patient_id`, `age`, `diagnosis`, `psa_level`, `tumor_stage`, `biopsy_gleason_score`, `treatment`, `follow_up_months`

### AUA Guideline Knowledge Base (12 indexed chunks)
- PSA Screening & Thresholds (AUA 2023)
- Prostate Cancer Staging (AJCC 8th Ed.)
- Gleason Score / Grade Groups
- Active Surveillance (AUA 2022)
- Radical Prostatectomy & Radiation Therapy
- Androgen Deprivation Therapy
- Renal Cell Carcinoma (AUA 2022)
- Bladder Cancer (AUA/SUO 2023)
- BPH Management (AUA 2022)
- Testicular Cancer

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Agent Framework | LangChain (ReAct) |
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Structured DB | SQLite3 |
| UI | Streamlit |
| Language | Python 3.11+ |

---

## 📈 Relevance to the USC Institute of Urology

This project directly addresses the core technical requirements of the **Associate Data Scientist** role:

| Job Requirement | UroAgent Implementation |
|-----------------|------------------------|
| Agentic AI workflows | LangChain ReAct loop with autonomous tool selection |
| Multi-modal data (structured + unstructured) | SQLite EHR + FAISS RAG in a unified agent |
| RAG with tool-calling | `@tool`-decorated functions with semantic retrieval |
| Clinical EHR context | Mock patient records with PSA, staging, Gleason, treatment |
| LangChain framework | Core agent, prompt, tools, and executor all in LangChain |

---

## ⚠️ Disclaimer

This is a **portfolio / research project**. All patient data is entirely synthetic and randomly generated. Clinical guideline content is simplified for demonstration purposes. This tool is **not** intended for real clinical decision-making.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Your Name] · Targeting the Associate Data Scientist role at USC Institute of Urology*
