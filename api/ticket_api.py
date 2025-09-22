# api/ticket_api.py
from flask import Flask, request, jsonify
import sqlite3, os
from datetime import datetime

DB = os.environ.get("TICKET_DB", "tickets.db")

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        user TEXT,
        channel TEXT,
        question TEXT,
        status TEXT,
        metadata TEXT
    )
    """)
    conn.commit(); conn.close()

app = Flask(__name__)
init_db()

@app.route("/ticket", methods=["POST"])
def create_ticket():
    data = request.json or {}
    user = data.get("user", "unknown")
    channel = data.get("channel", "web")
    question = data.get("question", "")
    metadata = data.get("metadata", "")
    ts = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO tickets (ts,user,channel,question,status,metadata) VALUES (?,?,?,?,?,?)",
              (ts, user, channel, question, "open", str(metadata)))
    conn.commit()
    ticket_id = c.lastrowid
    conn.close()
    return jsonify({"ticket_id": ticket_id}), 201

@app.route("/tickets", methods=["GET"])
def list_tickets():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, ts, user, channel, question, status FROM tickets ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    data = [{"id": r[0], "ts": r[1], "user": r[2], "channel": r[3], "question": r[4], "status": r[5]} for r in rows]
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
