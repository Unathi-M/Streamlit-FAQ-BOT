# tests/evaluate.py
import csv
from rapidfuzz import fuzz
from rag.qa_pinecone import answer_query

THRESHOLD = 70

def load_samples(path="tests/sample_tickets.csv"):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def evaluate(samples):
    total = len(samples)
    correct = 0
    for s in samples:
        q = s['question']
        expected = s['expected_answer']
        out = answer_query(q, top_k=4)
        pred = out.get('answer','').strip()
        score = fuzz.token_set_ratio(expected, pred)
        ok = score >= THRESHOLD
        print(f"Q: {q}\nExpected: {expected}\nPred: {pred}\nSim: {score} -> {'OK' if ok else 'MISS'}\n---")
        if ok:
            correct += 1
    acc = correct / total if total else 0
    print(f"Accuracy: {acc:.2%} ({correct}/{total})")
    return acc

if __name__ == "__main__":
    samples = load_samples()
    evaluate(samples)
