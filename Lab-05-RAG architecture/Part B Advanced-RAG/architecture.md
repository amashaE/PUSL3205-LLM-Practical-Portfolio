
flowchart TD
    A[University PDF Documents] --> B[Document Ingestion]
    B --> C[Text Extraction and Chunking]
    C --> D[Qwen3 Embeddings]
    D --> E[(ChromaDB Vector Store)]
    C --> F[Stored Document Chunks]
    F --> G[BM25 Keyword Index]

    U[User Question] --> S[Streamlit Application]
    S --> Q[Query and Document Detection]
    Q --> V[Dense Vector Retrieval]
    E --> V
    Q --> K[BM25 Keyword Retrieval]
    G --> K

    V --> R[Reciprocal Rank Fusion]
    K --> R
    R --> X[Cross-Encoder Reranking]
    X --> N[Relevant Context Selection]
    N --> L[Ollama Llama 3.2 3B]
    L --> O[Generated Answer]
    O --> P[Answer and Source References]
