# rag/list_collections.py
# list_vectors.py
import chromadb

client = chromadb.PersistentClient(path="chromadb_store")
collection = client.get_collection("faq_collection")
print("Number of vectors in collection:", len(collection.get()["ids"]))

