
import json
import csv
import time

import rag_pipeline_baseline as baseline
import hybrid_rag_pipeline as hybrid

from ingest import get_vectorstore


DATASET_FILE = "evaluation_dataset.json"
OUTPUT_FILE = "retrieval_evaluation_results.csv"


def page_key(doc):
    """Identify a unique PDF page."""

    filename = doc.metadata.get("filename", "")
    page = int(doc.metadata.get("page", 0)) + 1

    return filename, page


def unique_pages(docs):
    """Remove duplicate chunks from the same page, preserving rank."""

    pages = []
    seen = set()

    for doc in docs:
        key = page_key(doc)

        if key not in seen:
            seen.add(key)
            pages.append(key)

    return pages


def calculate_metrics(docs, expected_document, expected_pages):
    """
    Calculate page-level Recall@5 and MRR.

    Recall@5 = 1 if at least one relevant page is in the top five.
    MRR = reciprocal rank of the first relevant page.
    """

    ranked_pages = unique_pages(docs)

    relevant_pages = {
        (expected_document, page)
        for page in expected_pages
    }

    top_five = ranked_pages[:5]

    recall_at_5 = int(
        any(page in relevant_pages for page in top_five)
    )

    reciprocal_rank = 0.0

    for rank, page in enumerate(ranked_pages, start=1):
        if page in relevant_pages:
            reciprocal_rank = 1.0 / rank
            break

    return recall_at_5, reciprocal_rank


def rerank_documents(module, question, docs, requested_documents):
    """
    Apply the same CrossEncoder selection logic to both systems.

    Select up to three chunks per document.
    If no document was explicitly requested, retain the two
    highest-scoring document groups.
    """

    if not docs:
        return []

    groups = {}

    for doc in docs:
        filename = doc.metadata.get(
            "filename",
            "Unknown document"
        )

        groups.setdefault(filename, []).append(doc)

    selected_by_doc = {}

    for filename, group in groups.items():

        # Fall back to the original question if sub_query is None.
        scores = module.reranker.predict(
            [
                (
                    doc.metadata.get("sub_query") or question,
                    doc.page_content
                )
                for doc in group
            ]
        )

        ranked = sorted(
            zip(group, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        # Q07 diagnostic: print the corrected reranker scores.
        if "facial expression" in question.lower():

            print(f"\nQ07 RERANKER SCORES - {filename}")

            for doc, score in ranked:
                print(
                    "Page:",
                    int(doc.metadata.get("page", 0)) + 1,
                    "| Score:",
                    round(float(score), 4)
                )

        selected_by_doc[filename] = [
            doc for doc, score in ranked[:3]
        ]

    # Retain the two highest-scoring document groups
    # when the user did not request a particular document.
    if not requested_documents and len(selected_by_doc) > 2:

        best_scores = {}

        for filename, chunks in selected_by_doc.items():

            scores = module.reranker.predict(
                [
                    (question, chunk.page_content)
                    for chunk in chunks
                ]
            )

            best_scores[filename] = float(max(scores))

        top_names = sorted(
            best_scores,
            key=best_scores.get,
            reverse=True
        )[:2]

        selected_by_doc = {
            name: selected_by_doc[name]
            for name in top_names
        }

    return [
        doc
        for chunks in selected_by_doc.values()
        for doc in chunks
    ]


def evaluate_system(name, module, questions, vectorstore):

    rows = []

    for item in questions:

        question = item["question"]

        # 1. Measure retrieval latency
        start = time.perf_counter()

        docs, requested_documents = module.retrieve_documents(
            question,
            vectorstore
        )

        retrieval_time = time.perf_counter() - start

        # 2. Evaluate initial retrieval
        recall_before, mrr_before = calculate_metrics(
            docs,
            item["expected_document"],
            item["expected_pages"]
        )

        # 3. Measure reranking latency
        start = time.perf_counter()

        reranked_docs = rerank_documents(
            module,
            question,
            docs,
            requested_documents
        )

        reranking_time = time.perf_counter() - start

        # 4. Print Q07 retrieved pages after reranking
        if item["id"] == "Q07":

            print(
                f"\nDEBUG Q07 - {name} "
                "documents after reranking:"
            )

            for doc in reranked_docs:
                print(
                    doc.metadata.get("filename"),
                    "PDF page:",
                    int(doc.metadata.get("page", 0)) + 1
                )

        # 5. Evaluate after reranking
        recall_after, mrr_after = calculate_metrics(
            reranked_docs,
            item["expected_document"],
            item["expected_pages"]
        )

        row = {
            "system": name,
            "question_id": item["id"],
            "question": question,
            "recall_at_5_before": recall_before,
            "mrr_before": mrr_before,
            "recall_at_5_after": recall_after,
            "mrr_after": mrr_after,
            "retrieval_latency_seconds": retrieval_time,
            "reranking_latency_seconds": reranking_time,
            "total_retrieval_and_reranking_seconds":
                retrieval_time + reranking_time
        }

        rows.append(row)

        print(
            f"{name} | {item['id']} | "
            f"Recall@5: {recall_before} -> {recall_after} | "
            f"MRR: {mrr_before:.3f} -> {mrr_after:.3f}"
        )

    return rows


def print_summary(name, rows):

    count = len(rows)

    print(f"\n{name}")

    print(
        "Recall@5 before reranking:",
        f"{sum(r['recall_at_5_before'] for r in rows) / count:.3f}"
    )

    print(
        "MRR before reranking:",
        f"{sum(r['mrr_before'] for r in rows) / count:.3f}"
    )

    print(
        "Recall@5 after reranking:",
        f"{sum(r['recall_at_5_after'] for r in rows) / count:.3f}"
    )

    print(
        "MRR after reranking:",
        f"{sum(r['mrr_after'] for r in rows) / count:.3f}"
    )

    print(
        "Average retrieval latency:",
        f"{sum(r['retrieval_latency_seconds'] for r in rows) / count:.4f} seconds"
    )

    print(
        "Average reranking latency:",
        f"{sum(r['reranking_latency_seconds'] for r in rows) / count:.4f} seconds"
    )

    print(
        "Average total latency:",
        f"{sum(r['total_retrieval_and_reranking_seconds'] for r in rows) / count:.4f} seconds"
    )


def main():

    # Load evaluation questions
    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:

        questions = json.load(file)

    if not questions:
        raise ValueError("The evaluation dataset is empty.")

    print(f"Loaded {len(questions)} evaluation questions.")

    vectorstore = get_vectorstore()

    # Warm up both systems before evaluation
    warmup_question = questions[0]["question"]

    print("\nWarming up retrieval systems...")

    baseline.retrieve_documents(
        warmup_question,
        vectorstore
    )

    hybrid.retrieve_documents(
        warmup_question,
        vectorstore
    )

    # Evaluate baseline
    print("\nEvaluating baseline vector retrieval...\n")

    baseline_rows = evaluate_system(
        "Baseline Vector",
        baseline,
        questions,
        vectorstore
    )

    # Evaluate hybrid
    print("\nEvaluating hybrid BM25 + Vector retrieval...\n")

    hybrid_rows = evaluate_system(
        "Hybrid BM25 + Vector",
        hybrid,
        questions,
        vectorstore
    )

    # Combine results
    all_rows = baseline_rows + hybrid_rows

    # Save detailed results to CSV
    fieldnames = list(all_rows[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_rows)

    # Print final comparison
    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print_summary(
        "Baseline Vector",
        baseline_rows
    )

    print_summary(
        "Hybrid BM25 + Vector",
        hybrid_rows
    )

    print("\nDetailed results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
