# Streamlit-FAQ-BOT
Proof-of-Concept bot built in Python


📚 FAQ-BOT

An intelligent FAQ chatbot powered by Pinecone vector search and Streamlit UI. The bot retrieves answers from a knowledge base using embeddings and provides retrieval-only responses with optional summarization.

The project is containerized with Docker, supports autoscaling on free cloud platforms, and includes monitoring via Prometheus and Grafana.


🚀 Features

RAG (Retrieval-Augmented Generation) pipeline

Embeddings stored & searched in Pinecone

Retrieval-only answers (fast, reliable)

Optional summarization with Flan-T5-Small

Interactive Chat UI

Built with Streamlit

Real-time FAQ chat interface

Containerized Deployment

Dockerfile for reproducible builds

docker-compose.yaml integrates monitoring stack

Compatible with Railway.app, Render, or Hugging Face Spaces


Monitoring & Logging

Prometheus: query per second (QPS), latency

Grafana: dashboard visualization

Simple logging for debugging

Scalable

Runs locally or deploys on free cloud tiers

Ready for autoscaling containers


🛠️ Tech Stack

Language: Python 3.10

Libraries:

pinecone → vector database

transformers → embeddings + summarization (Flan-T5)

streamlit → chatbot interface


Infrastructure:

Docker → containerization

Prometheus + Grafana → monitoring/metrics

docker-compose → orchestration


📂 Project Structure
faq-bot/
├── rag/
│   ├── __init__.py
│   ├── build_embeddings_pinecone.py   # Index builder
│   ├── qa_pinecone.py                 # Retrieval-only QA
│   ├── app_conv.py                    # Streamlit chat UI
│
├── secrets.json                       # Stores API keys securely
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Container image
├── docker-compose.yaml                # Services (FAQ-Bot, Prometheus, Grafana)
└── README.md                          # Project documentation

⚙️ Setup & Usage
1️⃣ Local Setup
# Clone repo
git clone https://github.com/yourusername/faq-bot.git
cd faq-bot


# Create venv
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)


# Install dependencies
pip install -r requirements.txt


Add your Pinecone API key + index name to secrets.json:

{
  "PINECONE_API_KEY": "your-key-here",
  "PINECONE_INDEX": "faq-bot"
}


2️⃣ Run Locally
# Build embeddings
python rag/build_embeddings_pinecone.py

# Start Streamlit UI
streamlit run rag/app_conv.py


Access at 👉 http://localhost:8501

3️⃣ Docker Deployment
docker compose up --build


Services:

FAQ-BOT → http://localhost:8501

Prometheus → http://localhost:9090

Grafana → http://localhost:3000


📊 Monitoring

Prometheus scrapes metrics (QPS, latency, errors).

Grafana visualizes dashboards for performance monitoring.

Can be extended with alerts for production.


🌍 Real-World Use Cases

Internal Knowledge Base → Employees query company FAQs

Customer Support → Automated responses before escalating to agents

Education → Students query course material in natural language

Documentation Search → Developers find API usage examples quickly


🚧 Roadmap

 Add WhatsApp/Twilio webhook for chatbot integration

 Improve summarization with larger LLMs (when budget allows)

 CI/CD pipeline for auto-deployment

 Expand dataset with multi-source documents
 

🤝 Contributing

PRs are welcome! For major changes, open an issue first.


📜 License

MIT License – free to use, modify, and distribute.


📬 Contact

LinkedIn: www.linkedin.com/in/unathi-manana
Email: unathimanana77@gmail.com
