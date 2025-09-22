# list_vectors.py
import chromadb

PERSIST_DIR = "chromadb_store"
COLLECTION_NAME = "faq_collection"

client = chromadb.PersistentClient(path=PERSIST_DIR)

try:
    collection = client.get_collection(name=COLLECTION_NAME)
    vectors = collection.get()
    print(f"Collection '{COLLECTION_NAME}' has {len(vectors['ids'])} vector(s).")
except Exception as e:
    print(f"Failed to fetch collection '{COLLECTION_NAME}': {e}")
