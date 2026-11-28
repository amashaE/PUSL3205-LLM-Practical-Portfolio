
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "documents/PUSL3123_Assessment_Brief 26-27 (1).pdf"

loader = PyPDFLoader(pdf_path)
pages = loader.load()

print("Total PDF pages:", len(pages))

for i, page in enumerate(pages):
    text = page.page_content

    if "learning" in text.lower() or "outcome" in text.lower() or "alo" in text.lower():
        print("\n" + "=" * 60)
        print(f"PAGE {i + 1}")
        print("=" * 60)
        print(text)
