# app.py
import streamlit as st
from data_prep import load_documents
from qa_chain import make_qa_pipeline

st.set_page_config(page_title="FAQ Bot — Hugging Face", page_icon="🤖")
st.title("FAQ Bot — Hugging Face (Free Model)")

# Load docs and QA model once
if "context" not in st.session_state:
    st.session_state.context = load_documents("docs")

if "qa" not in st.session_state:
    st.session_state.qa = make_qa_pipeline()

st.write("Ask a question about the FAQ data in /docs")

query = st.text_input("Your question:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        res = st.session_state.qa(question=query, context=st.session_state.context)
    st.markdown("### Answer")
    st.write(res["answer"])
    st.markdown("### Confidence")
    st.write(f"{res['score']:.2f}")
