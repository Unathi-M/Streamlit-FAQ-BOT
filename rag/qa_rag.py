# rag/qa_rag.py
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Paths & constants
PERSIST_DIR = os.path.join(os.getcwd(), "chromadb_store")
COLLECTION_NAME = "faq_collection"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QA_MODEL_NAME = "distilbert-base-uncased-distilled-squad"  # CPU-friendly

class FAQ_RAG:
    def __init__(self):
        # 1️⃣ Load embedding model
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 2️⃣ Initialize Chroma client
        self.client = chromadb.PersistentClient(path=PERSIST_DIR)
        
        # 3️⃣ Load collection or raise error
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
            print(f"Loaded existing collection '{COLLECTION_NAME}'")
        except:
            raise ValueError(f"Collection '{COLLECTION_NAME}' not found. Run build_embeddings.py first.")
        
        # 4️⃣ Initialize QA pipeline
        self.reader = pipeline(
            "question-answering",
            model=QA_MODEL_NAME,
            tokenizer=QA_MODEL_NAME,
            device=-1  # CPU
        )

    def answer(self, query, top_k=4):
        # Embed query
        query_vector = self.embedder.encode(query).tolist()
        
        # Retrieve top-k similar chunks
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        # Prepare sources text
        context_texts = []
        sources_info = []
        for chunk_text, chunk_meta in zip(results["documents"][0], results["metadatas"][0]):
            context_texts.append(chunk_text)
            sources_info.append({
                "id": chunk_meta.get("id", ""),
                "chunk_index": chunk_meta.get("chunk_index", 0),
                "source": chunk_meta.get("source", "")
            })
        
        # Combine retrieved chunks as context
        context = "\n".join(context_texts)
        
        # Run QA
        qa_output = self.reader(question=query, context=context)
        
        return {
            "answer": qa_output["answer"],
            "score": qa_output["score"],
            "sources": sources_info
        }
