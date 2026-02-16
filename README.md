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

### 🎥 Demo Video
Link: https://drive.google.com/file/d/1_CSQqerAGQakDI1dYpmgHvEDOx-PUFk_/view?usp=sharing

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

## 📊 Success Metrics

The system was evaluated using a rigorous test suite of 18 domain-specific questions across 3 iterations (54 total queries) to ensure statistical consistency.

### 1. Performance (Latency)
*Measured on a steady-state system (post-warmup).*

| Metric | Result | Target / Status |
| :--- | :--- | :--- |
| **Average Query Latency** | **218.07 ms** | ✅ Ultra Fast (< 500ms) |
| **P95 Latency** | **363.09 ms** | ✅ High Consistency |
| **P99 Latency (Worst Case)** | **400.03 ms** | ✅ Reliable |
| **Successful Queries** | **100% (54/54)** | ✅ Zero Failures |

### 2. Retrieval Quality (Accuracy)
*Evaluated based on the system's ability to find the correct document chunk for technical project details.*

| Metric | Accuracy | Count |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | **88.89%** | 16 / 18 |
| **Top-3 Accuracy** | **100.00%** | 18 / 18 |
| **Top-5 Accuracy** | **100.00%** | 18 / 18 |

---

## 🧪 Custom Testing & Limitations

### 🛠️ Run Your Own Tests
You can evaluate the system on your own custom questions:

1.  **Performance:** Open `test_api.py`, modify the `TEST_QUESTIONS` list with your own queries, and run:
    ```bash
    python test_api.py
    ```
    *Check `baseline_performance.json` for detailed latency metrics.*

2.  **Accuracy:** Open `accuracy_evaluator.py`, update the test set, and run:
    ```bash
    python accuracy_evaluator.py
    ```

### ⚠️ Known Limitations
* **Scanned PDFs:** This version utilizes `pdfplumber` for high-speed text extraction. It **does not support scanned image PDFs** (OCR is not enabled in this MVP to prioritize latency).
* **Scanned Tables:** Tables within images or screenshots will not be parsed. The system works best with digital-native PDFs where text is selectable.

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

## ⚠️ Challenges & Optimizations

### 1. Handling Large Documents
Processing extremely large PDFs (100+ pages) introduces significant "noise" into the vector search, as irrelevant sections may share semantic similarities with the query.
* **Current Limit:** The system is optimized for Environmental Clearance documents under **50 pages**.
* **Trade-off:** Larger documents require more RAM for the **BM25 index construction** and increase the **Reranking latency** (scoring more candidates takes longer).

### 2. The Chunking Dilemma (Accuracy vs. Speed)
Finding the perfect chunk size was a critical challenge:
* **Too Small (< 200 chars):** Precise matching but loses context (e.g., a number without its unit).
* **Too Large (> 1000 chars):** Captures context but confuses the embedding model with multiple topics in one vector.
* **Solution:** We settled on a **400-character chunk size with 100-character overlap**. This ensures that numerical values (like "333 KLD") are always captured alongside their labels.

### 3. Fine-Tuning Embeddings (Future Stretch Goal)
The current system uses the pre-trained `all-mpnet-base-v2` model. While excellent for general English, it struggles slightly with highly specific industry acronyms (e.g., "NOC", "CTE", "EC").
* **Optimization:** A future improvement would be to **fine-tune the embedding model** on a dataset of real estate and environmental regulation documents. This would align the vector space more closely with the domain-specific vocabulary.

