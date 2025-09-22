# api/twilio_webhook.py
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
import sqlite3, json, os
from datetime import datetime
from rag.qa_pinecone import answer_query

DB = os.environ.get("TICKET_DB", "tickets.db")
app = Flask(__name__)

def create_ticket_in_db(user, channel, question, metadata=None):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute("INSERT INTO tickets (ts, user, channel, question, status, metadata) VALUES (?, ?, ?, ?, ?, ?)",
              (ts, user, channel, question, "open", json.dumps(metadata or [])))
    conn.commit()
    ticket_id = c.lastrowid
    conn.close()
    return ticket_id

@app.route("/twilio_whatsapp", methods=["POST"])
def twilio_whatsapp():
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '')
    res = answer_query(incoming_msg, top_k=4)
    answer = res.get("answer", "I don't know — please contact support.")
    score = res.get("score", 0.0)
    # create ticket if low confidence or explicit fallback
    if score < 0.2 or "i don't know" in answer.lower():
        create_ticket_in_db(user=from_number, channel="whatsapp", question=incoming_msg, metadata=res.get("sources"))
    resp = MessagingResponse()
    resp.message(answer)
    return Response(str(resp), mimetype="application/xml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
