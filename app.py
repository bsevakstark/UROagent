"""
app.py
------
Streamlit web interface for UroAgent.
Run with: `streamlit run app.py`
"""

import streamlit as st
from agent import build_agent, run_query

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UroAgent — USC Institute of Urology",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=80)
    st.title("UroAgent")
    st.markdown("**USC Institute of Urology**")
    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "UroAgent is an autonomous clinical AI assistant built on a "
        "ReAct (Reason + Act) agentic loop.\n\n"
        "It can query:\n"
        "- 🗃️ **EHR Database** (structured SQL)\n"
        "- 📖 **AUA Guidelines** (RAG / FAISS)"
    )
    st.markdown("---")
    st.markdown("**Example Queries**")
    examples = [
        "Which patients have PSA > 10 ng/mL?",
        "What is the recommended treatment for T3b prostate cancer?",
        "Show all patients on active surveillance and explain the AUA criteria.",
        "Compare the Gleason scores of prostate cancer patients.",
        "What does the AUA say about BPH treatment options?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("🏥 UroAgent — Clinical AI Assistant")
st.markdown(
    "Ask a clinical question about **patient records** or **AUA guidelines**. "
    "UroAgent will autonomously decide which tools to use."
)

# Initialise session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agent" not in st.session_state:
    with st.spinner("Initialising UroAgent…"):
        st.session_state["agent"] = build_agent()

# Chat history display
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "steps" in msg:
            with st.expander("🔍 Agent reasoning steps"):
                for i, (action, obs) in enumerate(msg["steps"], 1):
                    st.markdown(f"**Step {i} — Tool:** `{action.tool}`")
                    st.markdown(f"**Input:** {action.tool_input}")
                    st.code(str(obs), language="text")

# Chat input
query = st.chat_input(
    "e.g. Which patients have T3 or T4 prostate cancer and what does the guideline recommend?"
)

# Also accept sidebar button clicks
if "query_input" in st.session_state:
    query = st.session_state.pop("query_input")

if query:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("UroAgent is reasoning…"):
            result = run_query(query, st.session_state["agent"])

        answer = result["output"]
        steps  = result["steps"]

        st.markdown(answer)

        if steps:
            with st.expander("🔍 Agent reasoning steps"):
                for i, (action, obs) in enumerate(steps, 1):
                    st.markdown(f"**Step {i} — Tool:** `{action.tool}`")
                    st.markdown(f"**Input:** {action.tool_input}")
                    st.code(str(obs), language="text")

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "steps": steps}
    )
