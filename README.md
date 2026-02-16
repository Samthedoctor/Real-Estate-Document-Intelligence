# 🏢 Real Estate Document Intelligence (RAG System)

A high-performance **Retrieval-Augmented Generation (RAG)** system designed to extract precise information from environmental clearance documents. This system utilizes a **Hybrid Search** architecture, **AI Reranking**, and **Streaming LLM** generation.

---

## 📥 Setup & Installation

Follow these steps to get the system running locally:

### 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Configure API Key and Index Your Documents
```bash
# Windows (CMD)
set GROQ_API_KEY=your_groq_api_key_here

# Mac/Linux or Git Bash
export GROQ_API_KEY=your_groq_api_key_here
```
Crucial Step: Before running the search, you must process the PDFs to create the vector database and keyword index. Place your PDF files in the project folder and run:
``` bash
python reindex.py
```
### 4. Run the Application
Start the FastAPI backend server:
```bash
python main.py
```
Once the server is running, open your browser and go to:
👉 http://localhost:8000/ui

### 🧪 Testing & Evaluation
To verify the system performance and retrieval quality, run the following scripts:

### 1. Performance Benchmarking
This script measures latency, including Average and P95 metrics. It includes a warm-up phase to ensure steady-state accuracy.
```bash
python test_api.py
```
Results will be saved to baseline_performance.json.

### 2. Retrieval Accuracy
This script evaluates the system against a test set of questions to calculate Top-1 and Top-3 accuracy.
```bash
python accuracy_evaluator.py
```

### 🛠️ Tech Stack
1) Backend: FastAPI, Python 3.11
2) Search: ChromaDB (Vector) + BM25 (Keyword)
3) Models: Embeddings: all-mpnet-base-v2
4) Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
5) LLM: Llama-3.3-70b-Versatile (via Groq)
6) Frontend: Vanilla JS, HTML5, CSS3 (Modern Indigo Theme)

### 🧩 Architecture Flow
1) Hybrid Retrieval: Simultaneous search using ChromaDB for semantic meaning and BM25 for exact keyword matching.
2) Reranking: top candidates are re-scored using a Cross-Encoder model.
3) Streaming Generation: LLM generates answers with real-time token streaming to the UI.


