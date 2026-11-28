import re
import hashlib

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def tokenize(text):
    """
    Convert text into lowercase tokens.
    Keeps module codes such as PUSL3123 intact.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def document_key(document):
    """
    Create a stable identifier for a document chunk.
    Used to avoid duplicates during rank fusion.
    """
    filename = document.metadata.get("filename", "")
    page = document.metadata.get("page", "")
    content = document.page_content

    raw = f"{filename}|{page}|{content}"

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()


class HybridRetriever:

    def __init__(self, vectorstore, rrf_k=60):

        self.vectorstore = vectorstore
        self.rrf_k = rrf_k

        # Retrieve the same chunks already stored in ChromaDB
        stored = vectorstore.get(
            include=["documents", "metadatas"]
        )

        self.documents = []

        for content, metadata in zip(
            stored["documents"],
            stored["metadatas"]
        ):
            if content and content.strip():
                self.documents.append(
                    Document(
                        page_content=content,
                        metadata=metadata or {}
                    )
                )

        if not self.documents:
            raise ValueError(
                "No indexed documents found in ChromaDB."
            )

        # Build the BM25 index from the existing chunks
        tokenized_corpus = [
            tokenize(doc.page_content)
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print(
            f"BM25 index created with "
            f"{len(self.documents)} chunks."
        )

    def retrieve(self, query, k=8, filename=None):

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # ------------------------------------------
        # 1. Dense vector retrieval
        # ------------------------------------------

        metadata_filter = None

        if filename:
            metadata_filter = {
                "filename": filename
            }

        vector_docs = self.vectorstore.similarity_search(
            query,
            k=k,
            filter=metadata_filter
        )

        # ------------------------------------------
        # 2. BM25 keyword retrieval
        # ------------------------------------------

        bm25_scores = self.bm25.get_scores(query_tokens)

        allowed_indices = [
            i for i, doc in enumerate(self.documents)
            if (
                filename is None
                or doc.metadata.get("filename") == filename
            )
        ]

        ranked_indices = sorted(
            allowed_indices,
            key=lambda i: float(bm25_scores[i]),
            reverse=True
        )

        # Ignore chunks with no keyword matches
        bm25_indices = [
            i for i in ranked_indices
            if bm25_scores[i] > 0
        ][:k]

        bm25_docs = [
            self.documents[i] for i in bm25_indices
        ]

        # ------------------------------------------
        # 3. Reciprocal Rank Fusion (RRF)
        # ------------------------------------------

        fused_scores = {}
        document_lookup = {}

        # Add vector retrieval rankings
        for rank, doc in enumerate(vector_docs, start=1):

            key = document_key(doc)

            fused_scores[key] = (
                fused_scores.get(key, 0)
                + 1 / (self.rrf_k + rank)
            )

            document_lookup[key] = doc

        # Add BM25 rankings
        for rank, doc in enumerate(bm25_docs, start=1):

            key = document_key(doc)

            fused_scores[key] = (
                fused_scores.get(key, 0)
                + 1 / (self.rrf_k + rank)
            )

            document_lookup[key] = doc

        # Sort combined rankings
        ranked_keys = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True
        )

        results = [
            document_lookup[key]
            for key in ranked_keys[:k]
        ]

        return results
