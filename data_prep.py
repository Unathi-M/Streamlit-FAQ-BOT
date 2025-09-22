# data_prep.py
import os

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_documents(folder: str = "docs") -> str:
    """Load all .txt files and join into one context string."""
    docs = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".txt"):
            path = os.path.join(folder, fname)
            docs.append(load_txt(path))
    return "\n".join(docs)

if __name__ == "__main__":
    context = load_documents("docs")
    print("Loaded context:")
    print(context[:300], "...")
