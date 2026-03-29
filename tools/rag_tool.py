"""
tools/rag_tool.py
-----------------
LangChain Tool — retrieves relevant AUA clinical guideline passages
using FAISS similarity search (Retrieval-Augmented Generation).
"""

import os
from functools import lru_cache
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = "data/aua_guidelines_index"
TOP_K = 3


@lru_cache(maxsize=1)
def _load_vectorstore() -> FAISS:
    """Load the FAISS index once and cache it for the process lifetime."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{INDEX_PATH}'. "
            "Please run `python setup_rag.py` first."
        )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


@tool
def query_aua_guidelines(clinical_question: str) -> str:
    """
    Search the AUA (American Urological Association) clinical guidelines
    for evidence-based recommendations relevant to a clinical question.

    Use this tool when the question involves:
    - Diagnostic criteria or PSA thresholds
    - Treatment recommendations (surgery, radiation, ADT, surveillance)
    - Cancer staging definitions or Gleason score interpretation
    - Screening protocols or follow-up schedules
    - Disease-specific management (prostate, bladder, kidney, testicular cancer, BPH)

    Input: a natural-language clinical question or keyword phrase.
    Output: the most relevant guideline passages with source citations.

    Examples:
      "PSA threshold for biopsy referral"
      "treatment options for T3b prostate cancer"
      "active surveillance criteria Gleason 6"
    """
    try:
        vectorstore = _load_vectorstore()
        docs = vectorstore.similarity_search(clinical_question, k=TOP_K)

        if not docs:
            return "No relevant guideline passages found for this query."

        sections = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            topic = doc.metadata.get("topic", "")
            sections.append(
                f"[Guideline {i}] Source: {source} | Topic: {topic}\n{doc.page_content}"
            )

        return (
            f"Retrieved {len(docs)} relevant AUA guideline passage(s):\n\n"
            + "\n\n".join(sections)
        )

    except FileNotFoundError as e:
        return f" RAG Error: {str(e)}"
    except Exception as e:
        return f" Unexpected RAG Error: {str(e)}"
