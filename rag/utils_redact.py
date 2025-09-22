# rag/utils_redact.py
import re

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+?\d{2,3}[-\s]?)?(\d{7,12})")
SA_ID_RE = re.compile(r"\b\d{13}\b")  # SA ID is 13 digits

def redact_text(s: str) -> str:
    s = EMAIL_RE.sub("[email]", s)
    s = SA_ID_RE.sub("[id]", s)
    s = PHONE_RE.sub("[phone]", s)
    return s
