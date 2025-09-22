# rag/app_rag.py
import streamlit as st
from logger_db import init_db, log_conversation
from utils_redact import redact_text

st.set_page_config(page_title="FAQ Bot (RAG)", page_icon="🤖")
st.title("FAQ Bot — Multilingual RAG (PoC)")

# Initialize logging DB
init_db()

# Initialize RAG once
if "rag" not in st.session_state:
    try:
        from qa_rag import FAQ_RAG
        st.session_state.rag = FAQ_RAG()
        st.success("✅ RAG initialized successfully!")
    except Exception as e:
        st.error(
            "RAG initialization failed — collection may not be built.\n\n"
            "Run (from project root):\n\n"
            "python rag/build_embeddings.py\n\n"
            f"Error details: {e}"
        )
        st.stop()

# Display Chroma collection info
try:
    coll = st.session_state.rag.collection
    vectors = coll.get()
    st.info(f"Collection '{coll.name}' loaded with {len(vectors['ids'])} vector(s).")
except Exception as e:
    st.warning(f"Could not fetch collection info: {e}")

# Sidebar settings
top_k = st.sidebar.slider("Top-K retrieval", min_value=1, max_value=8, value=4)
user_id = st.text_input("User ID (demo):", value="demo_user")
channel = st.selectbox("Channel:", ["web", "whatsapp", "email"])

# Query input
query = st.text_input("Ask a question:")

if st.button("Ask") and query.strip():
    # Redact query (optional)
    redacted_q = redact_text(query)
    
    # Get RAG answer
    res = st.session_state.rag.answer(query, top_k=top_k)
    
    # Display results
    st.markdown("### Answer")
    st.write(res["answer"])
    
    st.markdown("**Confidence:**")
    st.write(f"{res['score']:.3f}")
    
    st.markdown("**Sources used:**")
    for s in res["sources"]:
        st.write(f"- `{s['source']}` (chunk {s['chunk_index']})")
    
    # Log conversation (optional)
    log_conversation(channel, user_id, redacted_q, res["answer"], res["score"], res["sources"], escalated=0)
