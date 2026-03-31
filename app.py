import os
if not os.path.exists("data/patient_records.db"):
    import setup_db
    setup_db.main()

if not os.path.exists("data/aua_guidelines_index"):
    import setup_rag
    setup_rag.main()
python - << 'PYEOF'
with open('app.py', 'w') as f:
    f.write('''import os
import streamlit as st
from agent import build_agent, run_query

st.set_page_config(page_title="UroAgent", page_icon="🏥", layout="wide")

with st.sidebar:
    st.title("🏥 UroAgent")
    st.markdown("**USC Institute of Urology**")
    st.markdown("---")
    st.markdown("**Tools:**\\n- 🗃️ EHR Database (SQL)\\n- 📖 AUA Guidelines (RAG)")
    st.markdown("---")
    st.markdown("**Try asking:**")
    examples = [
        "Which patients have PSA > 10 ng/mL?",
        "What does AUA recommend for T3b prostate cancer?",
        "Show all prostate cancer patients and their treatments",
        "What are the criteria for active surveillance?",
        "What is the treatment for Gleason 8 prostate cancer?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query_input"] = ex

st.title("🏥 UroAgent — Clinical AI Assistant")
st.markdown("Ask a clinical question about **patient records** or **AUA guidelines**.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agent" not in st.session_state:
    with st.spinner("Initialising UroAgent..."):
        st.session_state["agent"] = build_agent()

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a clinical question...")

if "query_input" in st.session_state:
    query = st.session_state.pop("query_input")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("UroAgent is reasoning..."):
            result = run_query(query, st.session_state["agent"])
        answer = result["output"]
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
''')
print("app.py rewritten!")
PYEOF
streamlit run app.py
