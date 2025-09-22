# rag/build_embeddings.py
import os
import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
from data_prep_rag import create_chunks_from_docs

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PERSIST_DIR = "chromadb_store"
COLLECTION_NAME = "faq_collection"

def build_chroma_collection(collection_name: str = COLLECTION_NAME, persist_dir: str = PERSIST_DIR):
    print("Initializing Chroma PersistentClient at", persist_dir)
    client = chromadb.PersistentClient(path=persist_dir)

    # If the collection exists, delete and recreate (clean state for PoC)
    try:
        existing = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists. Deleting and recreating for clean PoC.")
        client.delete_collection(collection_name)
    except Exception:
        # Not found -> we'll create one below
        pass

    collection = client.create_collection(name=collection_name)
    print("Loading chunks from docs/ ...")
    chunks = create_chunks_from_docs("docs")
    if not chunks:
        raise ValueError("No document chunks found in docs/. Add some text/pdf files and try again.")

    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"], "id": c["id"]} for c in chunks]
    ids = [c["id"] for c in chunks]

    print(f"Encoding {len(texts)} chunks with {EMBED_MODEL_NAME} (this may take a minute)...")
    encoder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    emb_list = [e.tolist() for e in embeddings]

    print("Adding vectors to Chroma collection...")
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=emb_list)

    # try to persist (some clients support persist)
    try:
        client.persist()
    except Exception:
        pass

    print(f"Done. Added {len(ids)} vectors to collection '{collection_name}' (persist_dir={persist_dir})")
    return collection

if __name__ == "__main__":
    build_chroma_collection()
