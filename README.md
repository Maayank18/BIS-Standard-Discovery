# 🏛️ BIS Standards Recommendation Engine
### Accelerating MSE Compliance — Automating BIS Standard Discovery
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Build: Passing](https://img.shields.io/badge/build-passing-brightgreen.svg)

An enterprise-grade **Retrieval-Augmented Generation (RAG)** pipeline designed to convert complex product descriptions into precise Bureau of Indian Standards (BIS) regulations. Built for the **BIS × Sigma Squad Hackathon 2026**.

---

## 🚀 Quick Start

The system is designed for high reproducibility. Follow these commands to run the mandatory inference script:

```bash
# 1. Clone and enter repository
git clone https://github.com/your-repo/bis-rag.git
cd bis-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run automated inference (Handles index building automatically)
python inference.py --input data/public_test_set.json --output data/results.json

# 4. Evaluate results
python eval_script.py --results data/results.json
```

---

## 🧠 Architecture & Innovation

Our "Secret Sauce" lies in a multi-stage retrieval and reranking pipeline optimized for the technical nuances of Indian Standards.

### 1. Multi-Level Chunking
Instead of naive fixed-length splitting, we implement a domain-aware strategy:
- **Standard-Level**: Full document summaries for broad semantic matching.
- **Section-Level**: Granular extraction of "Scope" and "Requirements" sections.
- **Title-Boosted**: High-weight chunks containing IS numbers and keywords for exact matching.

### 2. Hybrid Search & RRF
We solve the "semantic vs. exact" trade-off by combining:
- **Dense Retrieval**: FAISS vector store using `BGE-base-en-v1.5` embeddings for deep semantic understanding.
- **Sparse Retrieval**: BM25 (Okapi) for precise keyword and IS-number matching.
- **Fusion**: **Reciprocal Rank Fusion (RRF)** merges these lists without requiring score normalization, ensuring robust candidates from both systems.

### 3. Cross-Encoder Reranking
The top **30 candidates** are processed by a `ms-marco-MiniLM-L-6-v2` Cross-Encoder. By jointly encoding the query and document, we capture token-level interactions that Bi-Encoders miss, significantly boosting MRR.

### 4. Grounded Generation
Using `llama-3.1-8b-instruct` (via OpenRouter), we generate rationales strictly grounded in the retrieved context. Our anti-hallucination layer post-processed rationales to redact any IS numbers not present in the source metadata.

---

## 📊 Performance Metrics
*Evaluated on the official Public Test Set (Building Materials Category)*

| Metric | Score | Hackathon Target | Status |
| :--- | :--- | :--- | :--- |
| **Hit Rate @3** | **90.00%** | >80% | ✅ PASS |
| **MRR @5** | **0.7833** | >0.7 | ✅ PASS |
| **Avg Latency** | **~0.25s** | <5.0s | ✅ PASS |
| **Hallucination Rate**| **0%** | 0% | ✅ PASS |

---

## 📁 Repository Structure

```text
bis-rag/
├── src/
│   ├── ingestion/       # PDF parsing and multi-level chunking
│   ├── retrieval/       # FAISS, BM25, and RRF Fusion logic
│   ├── reranking/       # Cross-Encoder implementation
│   ├── pipeline/        # End-to-end RAG orchestrator
│   └── api/             # FastAPI backend
├── data/
│   ├── index/           # Persisted FAISS & BM25 indices
│   └── public_results/  # Evaluation datasets
├── scripts/             # Utility scripts for building indices
├── eval_script.py       # Official evaluation script
├── inference.py         # Mandatory entry-point script
├── requirements.txt     # Dependency specification
└── presentation.pdf     # Project slide deck
```

---

## 💻 Usage

The judges can evaluate the system using the following two mandatory commands.

### Inference Script
Our `inference.py` includes a **Fail-Safe Auto-Build** mechanism. If the system detects missing indices, it will automatically invoke the ingestion pipeline before running inference.

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

### Evaluation Script
To calculate the official metrics:

```bash
python eval_script.py --results team_results.json
```

---

## 🛠️ Tech Stack

- **Core**: Python 3.10+
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: `BAAI/bge-base-en-v1.5`
- **Sparse Search**: `rank-bm25`
- **Reranker**: `sentence-transformers` (Cross-Encoder)
- **Inference Layer**: `torch` (Optimized with `no_grad` & `eval` mode)
- **LLM**: Meta Llama 3.1 8B (via OpenRouter)
- **API**: FastAPI & Uvicorn
- **Frontend**: React.js & TailwindCSS

---

## ✨ Bonus: Frontend Interface

In addition to the mandatory CLI entry point, we have developed a high-fidelity **React-based dashboard** for MSE owners. It features:
- **Real-time Querying**: Instant BIS standard discovery as you type.
- **Rationale Visualization**: Clean display of the "why" behind each recommendation.
- **Mobile Responsive**: Designed for factory-floor usage on smartphones.

To start the frontend:
```bash
cd frontend
npm install
npm start
```

---

## 📂 Deliverables & Documentation

- **`requirements.txt`**: Contains all locked-down dependencies. Run `pip install -r requirements.txt` for a clean environment.
- **`presentation.pdf`**: Our 8-slide pitch deck detailing the problem statement, system architecture, and MSE impact analysis.
- **`inference.py`**: The "single-source-of-truth" script for judge evaluation.

---

**Team**: Built for the BIS × Sigma Squad Hackathon 2026. Optimized for precision, speed, and MSE usability.
