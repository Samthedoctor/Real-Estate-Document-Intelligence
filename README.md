# 🏢 Real Estate Document Intelligence (RAG System)

A high-performance **Retrieval-Augmented Generation (RAG)** system designed to extract precise information from environmental clearance documents. This system utilizes a **Hybrid Search** architecture, **AI Reranking**, and **Streaming LLM** generation.

---

### 🚀 Key Technical Highlight: Advanced PDF Parsing
**Solving the "Garbage Text" Problem**

Unlike standard RAG systems that fail on complex real-estate brochures (producing garbage text like `(cid:48)(cid:68)` due to Identity-H encoding), this system features a **Self-Correcting Extraction Pipeline**.

* **The Challenge:** Critical documents like `E-128-Brochure.pdf` had corrupted text layers that rendered them unreadable to standard OCR tools.
* **The Solution:** Integrated **LlamaParse ** to visually reconstruct the document structure and text.
* **The Result:** Achieved **100% data integrity** and clean embeddings from previously "unreadable" files, ensuring zero hallucinations even on difficult source files. 

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

## 📊 Success Metrics & Benchmarks

The system was rigorously evaluated on **two distinct real-estate datasets** (Estate 128 Brochure & 222 Rajpur Brochure) to ensure domain adaptability.

### 🏆 Executive Summary
* **Fastest Response:** 141ms (Estate 128)
* **Highest Precision:** 100% Top-1 Accuracy (Estate 128)


### 1. Performance (Latency)
*Measured on a steady-state system (post-warmup) over 3 iterations.*

| Metric | Estate 128 (Complex) | 222 Rajpur (Standard) | Status |
| :--- | :--- | :--- | :--- |
| **Average Latency** | **185.85 ms** | **225.96 ms** | ✅ Ultra Fast (< 300ms) |
| **P95 Latency** | **224.90 ms** | **310.92 ms** | ✅ High Consistency |
| **P99 Latency** | **292.60 ms** | **369.93 ms** | ✅ No Spikes |
| **Cache Miss Speed** | **~185 ms** | **~226 ms** | 🚀 Optimized Retrieval |

### 2. Retrieval Quality (Accuracy)
*Evaluated against a Golden Dataset of 18 domain-specific questions per document.*

| Metric | Estate 128 Accuracy | 222 Rajpur Accuracy | 
| :--- | :--- | :--- | 
| **Top-1 Accuracy** | **100.00%** (18/18) | **94.44%** (17/18) | 
| **Top-3 Accuracy** | **100.00%** (18/18) | **100.00%** (18/18) | 


---

### 📂 🔍 Verification (Raw Logs)
To verify these metrics, detailed execution logs are provided in this repository. You can inspect the exact latency per query and the retrieved chunks for every test case.

* 📄 **`E-128_latency.txt and latency_222_rajpur.txt from the terminal outputs`**: Contains the millisecond-level breakdown of all 54 queries per document, including cache hit/miss rates and P95 calculations.
* 📄 **`accuracy_222_rajpur.txt and performance_metrics_E-128.txt from the terminal outputs `**: Contains the ground-truth comparison, showing the expected answer vs. the retrieved answer and the exact rank (1-5) where the correct information was found.

> *Note: These files are automatically generated when running `test_api.py` and `accuracy_evaluator.py`.*
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



