# 🚀 Elite Agentic RAG System (FastAPI + LangGraph + FAISS)

An advanced **Agentic Retrieval-Augmented Generation (RAG)** system that dynamically routes user queries between:

- 📄 **Local Knowledge Base (PDF documents via FAISS)**
- 🌐 **Real-time Web Search (DuckDuckGo)**

Built using **FastAPI, LangGraph, and Hugging Face models**, this project demonstrates a **production-ready AI architecture** for enterprise-grade knowledge assistants.

---

## 🧠 Key Features

- ✅ **Agentic Workflow (LangGraph)**  
  Intelligent routing between tools using a Supervisor node

- ✅ **RAG Pipeline (FAISS)**  
  Semantic search over uploaded PDF documents

- ✅ **Real-time Web Search**  
  Fetches up-to-date information (2026-ready system)

- ✅ **Hybrid Intelligence**  
  Combines internal + external knowledge sources

- ✅ **FastAPI Backend**  
  REST API for scalable deployment

- ✅ **Incremental Indexing**  
  Upload multiple PDFs without losing previous data

- ✅ **Source Attribution**  
  Returns document sources for explainability

---

## 🏗️ Architecture Overview
User Query
↓
Supervisor (Decision Node)
↓
├── Vector Store (FAISS) → Local PDF Knowledge
└── Web Search (DuckDuckGo) → Real-time Data
↓
Generator (LLM - Hugging Face)
↓
Final Answer + Sources


---

## 📁 Project Structure

project/
│
├── main.py # FastAPI + LangGraph Agent
├── config.py # Configuration & constants
├── requirements.txt # Dependencies
├── Dockerfile # Container setup
├── .env # API keys (not pushed)
└── faiss_index/ # Vector database (auto-created)


---

## ⚙️ Tech Stack

- **Backend:** FastAPI
- **LLM:** Hugging Face (LLaMA / Mistral)
- **Embeddings:** Sentence Transformers
- **Vector DB:** FAISS
- **Agent Framework:** LangGraph
- **Search Tool:** DuckDuckGo
- **Deployment:** Docker + Railway

---

## 🚀 Installation & Setup

For optional free web search through Parallel Search MCP, see
[Parallel provider setup](docs/parallel-search.md). DuckDuckGo remains the default.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/PRODUCTION_STYLE_AGENTIC_RAG.git
cd PRODUCTION_STYLE_AGENTIC_RAG

2. Create Virtual Environment

python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

3. Install Dependencies

pip install -r requirements.txt

4. Add Environment Variables

Create .env file:
HUGGINGFACEHUB_API_TOKEN=your_api_key_here

5. Run the Application
python main.py
The API will be available at http://127.0.0.1:8000. Access the Swagger UI at /docs.
App runs at:
http://127.0.0.1:8000

📡 API Endpoints
🔹 Health Check
GET /health

🔹 Upload PDF
POST /upload/
Upload PDF to build knowledge base

Request Body:
{
  "question": "What is this document about?",
  "thread_id": "user1"
}

🐳 Docker Deployment
Build Image

docker build -t PRODUCTION_STYLE_AGENTIC_RAG .
Run Container
docker run -p 8000:8000 agentic-rag

☁️ Deploy on Railway
Push code to GitHub
Go to Railway
Create new project → Deploy from GitHub
Add environment variable:
HUGGINGFACEHUB_API_TOKEN=your_key
