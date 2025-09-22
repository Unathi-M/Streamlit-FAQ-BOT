# rag/langchain_rag.py
"""
LangChain-powered RAG assistant with memory + retrieval + source display.
CPU-friendly (HuggingFace pipeline).
"""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


class LangChainRAG:
    def __init__(self,
                 persist_dir="chromadb_store",
                 collection_name="faq_collection",
                 memory_k=5):

        print(f"🔍 Loading Chroma collection: {collection_name}")

        # 1) Embeddings
        embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2) Vectorstore
        self.vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name
        )

        # 3) LLM pipeline
        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=-1
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # 4) Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        )

        # 5) Conversational retrieval chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectordb.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )

        print("✅ LangChain RAG initialized successfully.")

    def ask(self, question: str):
        """
        Ask a question and return structured result.
        """
        try:
            result = self.chain({"question": question})
            answer = result.get("answer", "No answer found.")
            sources = []

            for doc in result.get("source_documents", []):
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "content": (doc.page_content or "")[:200]
                })

            return {"answer": answer, "sources": sources}

        except Exception as e:
            return {"answer": f"❌ Error: {str(e)}", "sources": []}
