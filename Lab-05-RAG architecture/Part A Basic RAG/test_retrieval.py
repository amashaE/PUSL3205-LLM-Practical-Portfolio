from ingest import get_vectorstore

vectorstore = get_vectorstore()

question = """
What is the purpose of the PUSL3123 assessment,
and what are the broader aims of the Computing Award?
"""

documents = [
    "PUSL3123_Assessment_Brief 26-27 (1).pdf",
    "COMPUTING AWARD HANDBOOK_MOHE (2) (1).pdf"
]

for filename in documents:

    print("\n" + "#" * 80)
    print("SEARCHING DOCUMENT:", filename)
    print("#" * 80)

    results = vectorstore.similarity_search(
        question,
        k=5,
        filter={"filename": filename}
    )

    print("Retrieved chunks:", len(results))

    for i, doc in enumerate(results, start=1):

        print("\n" + "=" * 60)
        print("RESULT", i)
        print("=" * 60)

        print("Page:", doc.metadata.get("page", 0) + 1)
        print("\nContent:\n")
        print(doc.page_content)

from ingest import get_vectorstore

vectorstore = get_vectorstore()

tests = [
    {
        "name": "PUSL3123 assessment aims",
        "filename": "PUSL3123_Assessment_Brief 26-27 (1).pdf",
        "question": "What are the module aims and assessed learning outcomes of PUSL3123 AI and Machine Learning?"
    },
    {
        "name": "Computing Award programme aims",
        "filename": "COMPUTING AWARD HANDBOOK_MOHE (2) (1).pdf",
        "question": "What are the aims, educational objectives and overall structure of the Computing Award?"
    }
]

for test in tests:

    print("\n" + "#" * 80)
    print(test["name"])
    print("#" * 80)

    results = vectorstore.similarity_search(
        test["question"],
        k=3,
        filter={"filename": test["filename"]}
    )

    for i, doc in enumerate(results, start=1):

        print("\n" + "=" * 60)
        print("RESULT:", i)
        print("PAGE:", doc.metadata.get("page", 0) + 1)
        print(doc.page_content)
