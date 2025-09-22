# rag/qa_pinecone.py
import os
import json
import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone

# Prometheus metrics
from prometheus_client import Counter, Histogram, start_http_server

# --- Config: read secrets.json fallback to env ---
SECRETS_PATH = os.path.join(os.path.dirname(__file__), "..", "secrets.json")
secrets = {}
if os.path.exists(SECRETS_PATH):
    with open(SECRETS_PATH, "r", encoding="utf-8") as f:
        secrets = json.load(f)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or secrets.get("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or secrets.get("PINECONE_INDEX_NAME", "faq-index")

if not PINECONE_API_KEY or not INDEX_NAME:
    raise ValueError("Set PINECONE_API_KEY and PINECONE_INDEX_NAME in env or secrets.json")

# --- Models & clients ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "google/flan-t5-small")  # optional summarizer

# initialize Pinecone client (new SDK)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# sentence transformer embedder
embedder = SentenceTransformer(EMBED_MODEL)

# optional summarizer (lazy init)
_SUMMARIZER = None
def get_summarizer():
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = pipeline("text2text-generation", model=SUMMARIZER_MODEL, device=-1, max_new_tokens=128)
    return _SUMMARIZER

# simple in-memory caches
_embed_cache = {}

def embed_text(text: str):
    # very small cache to speed repeated queries
    key = text.strip().lower()
    if key in _embed_cache:
        return _embed_cache[key]
    vec = embedder.encode(text).tolist()
    _embed_cache[key] = vec
    return vec

# --- Metrics ---
REQUESTS = Counter("faq_requests_total", "Total FAQ queries")
LATENCY = Histogram("faq_request_latency_seconds", "Latency for FAQ queries")

# start prometheus metrics endpoint if enabled
if os.getenv("ENABLE_METRICS", "true").lower() in ("1", "true", "yes"):
    start_http_server(int(os.getenv("METRICS_PORT", 8000)))  # expose metrics on port 8000

# --- Retrieval + summarization functions ---
def query_pinecone(query: str, top_k: int = 4) -> List[Dict]:
    qv = embed_text(query)
    # new Pinecone index.query signature - try vector= first, fallback to queries=
    try:
        res = index.query(vector=qv, top_k=top_k, include_metadata=True)
    except TypeError:
        res = index.query(queries=[qv], top_k=top_k, include_metadata=True)
    matches = res.get("matches", []) or []
    docs = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        docs.append({
            "id": m.get("id"),
            "score": m.get("score"),
            "source": md.get("source", md.get("id")),
            "chunk_index": md.get("chunk_index"),
            "text": md.get("text", "")  # we stored snippet/text in metadata when building
        })
    return docs

def answer_retrieval_only(query: str, top_k: int = 4, summarize: bool = True) -> Dict:
    REQUESTS.inc()
    t0 = time.time()
    docs = query_pinecone(query, top_k=top_k)
    if not docs:
        LATENCY.observe(time.time() - t0)
        return {"answer": "Sorry — I don't have that info. Please contact support.", "sources": []}

    # quick retrieval-only answer: join top snippets
    snippets = [d["text"] for d in docs if d.get("text")]
    context = "\n\n".join(snippets)

    if not summarize:
        LATENCY.observe(time.time() - t0)
        return {"answer": context[:1200], "sources": docs}

    # summarizer
    try:
        summarizer = get_summarizer()
        prompt = f"Use the following context to answer the question succinctly.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        out = summarizer(prompt, max_new_tokens=128)[0]["generated_text"]
        LATENCY.observe(time.time() - t0)
        return {"answer": out.strip(), "sources": docs}
    except Exception as e:
        # if summarizer fails, fallback to raw context
        LATENCY.observe(time.time() - t0)
        return {"answer": context[:1200], "sources": docs}
