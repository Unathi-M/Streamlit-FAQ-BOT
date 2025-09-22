# rag/logger_db.py
import sqlite3
from datetime import datetime
import json

DB = "conversations.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        channel TEXT,
        user_id TEXT,
        question TEXT,
        answer TEXT,
        score REAL,
        sources TEXT,
        escalated INTEGER DEFAULT 0
    )
    """)
    conn.commit()
    conn.close()

def log_conversation(channel, user_id, question, answer, score, sources, escalated=0):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    INSERT INTO conversations (ts, channel, user_id, question, answer, score, sources, escalated)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), channel, user_id, question, answer, score, json.dumps(sources), escalated))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    log_conversation("web", "testuser", "What is return policy?", "You can return in 30 days", 0.92, [{"source":"sample_faq.txt"}])
    print("Logged sample.")
