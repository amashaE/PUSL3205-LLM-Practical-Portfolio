
import json
import csv
import time

import rag_pipeline_baseline as baseline
import hybrid_rag_pipeline as hybrid


DATASET_FILE = "evaluation_dataset.json"
OUTPUT_FILE = "answer_evaluation_results.csv"


def generate_answers(system_name, module, questions):

    results = []

    print(f"\nGenerating answers using {system_name}...")

    for item in questions:

        question_id = item["id"]
        question = item["question"]

        print(f"\nProcessing {question_id}: {question}")

        start = time.perf_counter()

        try:
            result = module.ask_university(question)

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            error = ""

        except Exception as e:

            answer = ""
            sources = []
            error = str(e)

        generation_time = time.perf_counter() - start

        source_text = "; ".join(
            f"{s.get('document', 'Unknown')} (page {s.get('page', '?')})"
            for s in sources
        )

        results.append({
            "system": system_name,
            "question_id": question_id,
            "question": question,
            "reference_answer": item["reference_answer"],
            "generated_answer": answer,
            "retrieved_sources": source_text,
            "generation_time_seconds": round(generation_time, 3),
            "error": error,
            "correctness_score": ""
        })

        if error:
            print("ERROR:", error)
        else:
            print("Answer generated successfully.")

    return results


def main():

    with open(DATASET_FILE, "r", encoding="utf-8-sig") as file:
        questions = json.load(file)

    print(f"Loaded {len(questions)} evaluation questions.")

    baseline_results = generate_answers(
        "Baseline Vector",
        baseline,
        questions
    )

    hybrid_results = generate_answers(
        "Hybrid BM25 + Vector",
        hybrid,
        questions
    )

    all_results = baseline_results + hybrid_results

    fieldnames = list(all_results[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 55)
    print("ANSWER GENERATION COMPLETED")
    print("=" * 55)
    print("Baseline answers:", len(baseline_results))
    print("Hybrid answers:", len(hybrid_results))
    print("Total answers:", len(all_results))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
