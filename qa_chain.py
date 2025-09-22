# qa_chain.py
from transformers import pipeline
from data_prep import load_documents

def make_qa_pipeline():
    # Hugging Face free model
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return qa_pipeline

def ask_question(question: str, context: str):
    qa = make_qa_pipeline()
    result = qa(question=question, context=context)
    return result["answer"]

if __name__ == "__main__":
    context = load_documents("docs")
    q = "What is your return policy?"
    ans = ask_question(q, context)
    print("Q:", q)
    print("A:", ans)
