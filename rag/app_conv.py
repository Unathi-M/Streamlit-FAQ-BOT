# rag/app_conv.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import time
from rag.qa_pinecone import answer_retrieval_only

st.set_page_config(page_title="FAQ Bot (retrieval)", layout="centered")
st.title("🤖 FAQ Bot — Retrieval + Summarizer (fast)")

# UI controls
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K retrieval", 1, 8, 4)
    summarize = st.checkbox("Use FLAN-T5 summarizer (optional)", value=True)
    st.write("Note: summarizer improves fluency but adds a small CPU cost.")

# conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# display history
for i, turn in enumerate(st.session_state.history):
    role = turn["role"]
    if role == "user":
        with st.chat_message("user"):
            st.markdown(turn["text"])
    else:
        with st.chat_message("assistant"):
            st.markdown(turn["text"])
            if turn.get("sources"):
                with st.expander("Sources used"):
                    for s in turn["sources"]:
                        st.write(f"- **{s.get('source')}** (chunk {s.get('chunk_index')}):")
                        st.write(s.get("text", "")[:400] + "…")

# input
prompt = st.chat_input("Ask a question about the company's policies, support, etc.")
if prompt:
    # show user message
    st.session_state.history.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # call retrieval pipeline
    start = time.time()
    res = answer_retrieval_only(prompt, top_k=top_k, summarize=summarize)
    latency = time.time() - start

    # show answer
    answer_text = res.get("answer", "Sorry — no answer.")
    with st.chat_message("assistant"):
        st.markdown(answer_text)
        if res.get("sources"):
            with st.expander("Sources used"):
                for s in res["sources"]:
                    st.write(f"- **{s.get('source')}** (chunk {s.get('chunk_index')})")
    # store assistant message
    st.session_state.history.append({"role": "assistant", "text": answer_text, "sources": res.get("sources", [])})

    # quick debug info
    st.sidebar.write(f"Last query latency: {latency:.2f}s")

    
