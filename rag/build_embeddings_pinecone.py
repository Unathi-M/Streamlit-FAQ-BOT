# rag/build_embeddings_pinecone.py
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from data_prep_rag import create_chunks_from_docs

# ---- Load secrets.json ----
with open("secrets.json") as f:
    secrets = json.load(f)

PINECONE_API_KEY = secrets["PINECONE_API_KEY"]
INDEX_NAME = secrets.get("PINECONE_INDEX_NAME", "faq-index")

# ---- Embedding model ----
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384  # model dimension

# ---- Initialize Pinecone client ----
pc = Pinecone(api_key=PINECONE_API_KEY)

def init_index():
    """Create index if it doesn't exist."""
    existing = pc.list_indexes().names()
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

def build_index(batch_size=32):
    index = init_index()
    encoder = SentenceTransformer(EMBED_MODEL)
    chunks = create_chunks_from_docs("docs")
    if not chunks:
        raise ValueError("No chunks found in docs/")

    to_upsert = []
    for i, c in enumerate(chunks):
        vec = encoder.encode(c["text"]).tolist()
        metadata = {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "id": c["id"],
            "text": c["text"][:1000]  # truncate long text
        }
        to_upsert.append({"id": c["id"], "values": vec, "metadata": metadata})

        if len(to_upsert) >= batch_size:
            index.upsert(vectors=to_upsert)
            to_upsert = []

    if to_upsert:
        index.upsert(vectors=to_upsert)

    print(f"✅ Upserted {len(chunks)} chunks into Pinecone index '{INDEX_NAME}'")

if __name__ == "__main__":
    build_index()
