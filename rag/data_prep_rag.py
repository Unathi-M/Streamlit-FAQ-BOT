# rag/data_prep_rag.py
"""
Load TXT/PDF docs and chunk them for RAG.
This version robustly supports 'pypdf' or 'PyPDF2' installs.
"""

import os
from typing import List, Dict, Optional

# Try imports for PDF readers (support pypdf and PyPDF2)
PdfReader = None
_pdf_lib_name = None
try:
    from pypdf import PdfReader
    _pdf_lib_name = "pypdf"
except Exception:
    try:
        from PyPDF2 import PdfReader
        _pdf_lib_name = "PyPDF2"
    except Exception:
        PdfReader = None
        _pdf_lib_name = None

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf_text(path: str) -> str:
    """
    Extract text from each page of a PDF using PdfReader (if available).
    Returns concatenated text or empty string on failure.
    """
    if PdfReader is None:
        print(f"⚠️ PDF support not available (no pypdf/PyPDF2 installed). Skipping PDF: {path}")
        return ""

    try:
        reader = PdfReader(path)
        pages_text = []
        # support both old and new reader.page interfaces
        for page in getattr(reader, "pages", getattr(reader, "pages", [])):
            try:
                text = page.extract_text() or ""
            except Exception:
                # some PDF pages might fail extract_text; skip them gracefully
                text = ""
            pages_text.append(text)
        return "\n".join(pages_text)
    except Exception as e:
        print(f"⚠️ Failed to read PDF {path}: {e}")
        return ""

def load_documents(folder: str = "docs") -> List[Dict]:
    """
    Load TXT and PDF documents from a folder.
    Returns list[{"id": filename, "text": content, "source": filename}]
    """
    docs = []
    if not os.path.isdir(folder):
        print(f"⚠️ docs folder not found: {folder}")
        return docs

    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue

        content: Optional[str] = None
        lower = fname.lower()
        if lower.endswith(".txt"):
            try:
                content = load_txt(path)
            except Exception as e:
                print(f"⚠️ Failed to load text file {fname}: {e}")
                continue
        elif lower.endswith(".pdf"):
            content = load_pdf_text(path)
            if not content:
                # skip PDFs we couldn't extract from
                continue
        else:
            # skip other filetypes
            continue

        docs.append({"id": fname, "text": content, "source": fname})
    return docs

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping character chunks (simple and deterministic).
    """
    out = []
    start = 0
    L = len(text)
    if L == 0:
        return out
    while start < L:
        end = min(start + chunk_size, L)
        out.append(text[start:end])
        start += chunk_size - overlap
    return out

def create_chunks_from_docs(folder: str = "docs") -> List[Dict]:
    """
    Load documents and produce chunk dicts:
    [{"id": "...", "text": "...", "source": "...", "chunk_index": 0}, ...]
    """
    docs = load_documents(folder)
    chunks = []
    for doc in docs:
        parts = chunk_text(doc["text"])
        for i, p in enumerate(parts):
            chunks.append({
                "id": f"{doc['id']}_chunk_{i}",
                "text": p,
                "source": doc["source"],
                "chunk_index": i
            })
    return chunks

if __name__ == "__main__":
    print(f"Using PDF library: {_pdf_lib_name}")
    c = create_chunks_from_docs("docs")
    print(f"✅ Created {len(c)} chunks from docs/")
