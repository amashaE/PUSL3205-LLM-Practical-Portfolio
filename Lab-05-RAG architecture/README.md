# Hybrid Retrieval and Reranking for a University RAG Assistant

## 1. Project Overview

This project enhances a Retrieval-Augmented Generation (RAG) based University AI Assistant by implementing a hybrid retrieval approach that combines keyword-based retrieval and semantic vector retrieval.

The system retrieves relevant information from university PDF documents and uses a Large Language Model (LLM) to generate answers based on the retrieved context.

The proposed approach combines BM25 keyword retrieval with dense vector retrieval. Reciprocal Rank Fusion (RRF) is used to combine the retrieval rankings, followed by Cross-Encoder reranking to improve the relevance of the selected document chunks.

The enhanced system is experimentally compared with the original vector-based RAG system.

## 2. Objectives

- Implement BM25 keyword-based retrieval.
- Combine BM25 with dense vector retrieval.
- Apply Reciprocal Rank Fusion (RRF) to merge retrieval rankings.
- Use Cross-Encoder reranking to reorder retrieved document chunks.
- Compare the proposed hybrid approach with baseline vector retrieval.
- Evaluate retrieval performance, answer correctness and processing latency.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| Streamlit | User interface |
| LangChain | RAG pipeline integration |
| ChromaDB | Vector database |
| Qwen3 Embedding 0.6B | Document embeddings |
| BM25 (`rank-bm25`) | Keyword retrieval |
| Cross-Encoder | Document reranking |
| Ollama | Local LLM execution |
| Llama 3.2 3B | Answer generation |

## 4. System Architecture

The system follows a hybrid RAG architecture consisting of document ingestion, indexing, retrieval, rank fusion, reranking and answer generation.

See [architecture.md](architecture.md) for the architecture diagram.

### Processing workflow

1. University PDF documents are loaded and their text is extracted.
2. Extracted text is divided into overlapping chunks.
3. Qwen3 embeddings are generated and stored in ChromaDB.
4. BM25 indexes the document chunks for keyword-based retrieval.
5. A user submits a question through the Streamlit interface.
6. Dense vector retrieval and BM25 independently retrieve relevant chunks.
7. Reciprocal Rank Fusion combines the two ranked result lists.
8. A Cross-Encoder reranks the retrieved candidates.
9. Relevant context is passed to the Llama 3.2 3B model through Ollama.
10. The generated answer and source references are displayed to the user.

## 5. Project Structure

```text
university-ai-assistant/
│
├── app.py
├── ingest.py
├── rag_pipeline.py
├── rag_pipeline_baseline.py
├── hybrid_rag_pipeline.py
├── hybrid_retrieval.py
│
├── evaluate_retrieval.py
├── evaluate_answers.py
│
├── evaluation_dataset.json
├── retrieval_evaluation_results.csv
├── answer_evaluation_results.csv
├── answer_evaluation_results_scored.csv
│
├── architecture.md
├── README.md
├── requirements.txt
│
├── documents/
├── chroma_db/
└── rag_env/
```

**Note:** The folder structure above describes the main project files. The virtual environment and generated ChromaDB database should not normally be uploaded to GitHub. The PDF documents may also be excluded if they cannot be redistributed.

## 6. Installation and Setup

### Prerequisites

- Python 3.12
- Ollama
- Git
- Internet connection for installing Python packages and downloading models

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd university-ai-assistant
```

### Step 2: Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv rag_env
.\rag_env\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Download the required Ollama models

```powershell
ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6b
```

Ensure Ollama is running before launching the application.

### Step 5: Add university PDF documents

Place the required PDF documents inside the `documents/` directory.

### Step 6: Ingest the documents

```powershell
python ingest.py
```

This extracts and chunks the PDF content, generates embeddings and stores them in ChromaDB.

### Step 7: Run the application

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal to access the University AI Assistant.

## 7. Evaluation Methodology

The baseline vector retrieval system was compared with the proposed hybrid BM25 and vector retrieval system using a dataset of 12 evaluation questions.

The following metrics were used:

- **Recall@5:** Measures whether at least one relevant page appears within the top five retrieved pages.
- **Mean Reciprocal Rank (MRR):** Measures the ranking position of the first relevant page.
- **Answer Correctness:** Human assessment using scores of 1 (correct), 0.5 (partially correct) and 0 (incorrect).
- **Latency:** Measures retrieval, reranking and answer-generation processing time.

## 8. Experimental Results

### Retrieval performance

| Metric | Baseline | Hybrid |
|---|---:|---:|
| Recall@5 before reranking | 1.000 | 1.000 |
| MRR before reranking | 0.667 | 0.715 |
| Recall@5 after reranking | 1.000 | 0.917 |
| MRR after reranking | 0.750 | 0.725 |
| Average retrieval latency | 0.347 s | 0.464 s |
| Average reranking latency | 0.780 s | 0.635 s |
| Average total retrieval and reranking latency | 1.126 s | 1.099 s |

### Answer correctness

| System | Score | Accuracy |
|---|---:|---:|
| Baseline Vector | 7/12 | 58.3% |
| Hybrid BM25 + Vector | 7/12 | 58.3% |

### Discussion

The hybrid approach achieved a slightly higher MRR before reranking, indicating improved initial ranking performance in this evaluation.

However, after reranking, the baseline achieved higher Recall@5 and MRR. The hybrid system had slightly higher initial retrieval latency but lower measured reranking latency. The overall retrieval and reranking latency was similar for both systems.

Both systems achieved the same human-assessed answer correctness score. Therefore, the experiment does not demonstrate an overall improvement in answer correctness from hybrid retrieval.

## 9. Limitations

- The evaluation dataset contained only 12 questions.
- Latency measurements were collected from one evaluation run.
- Answer correctness was assessed manually.
- The quality of generated answers depends on the relevance and completeness of retrieved context.
- For Q07, the hybrid system retrieved the expected page before reranking, but the page was absent from the final reranked results. This inconsistency was recorded as an evaluation limitation and requires further investigation.
- The evaluation results are specific to the selected university documents and test questions.

## 10. Conclusion

This project implemented a hybrid retrieval approach combining BM25 keyword retrieval and dense vector retrieval using Reciprocal Rank Fusion, followed by Cross-Encoder reranking.

The experimental evaluation demonstrated that hybrid retrieval can improve initial document ranking for some questions. However, the results also showed that the effectiveness of the complete retrieval pipeline depends on the interaction between retrieval and reranking.

The project provided practical experience in developing, integrating and evaluating an enhanced RAG architecture using locally executed language models and university document collections.

## 11. Repository Contents

The repository contains the source code, evaluation dataset, experimental result files, architecture documentation and setup instructions required to understand and reproduce the project.


