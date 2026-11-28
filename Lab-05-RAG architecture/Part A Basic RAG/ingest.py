
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Folder paths
DOCUMENTS_FOLDER = "documents"
VECTORSTORE_FOLDER = "chroma_db"


# Initialize embedding model
def get_embeddings():
    return OllamaEmbeddings(
        model="qwen3-embedding:0.6b"
    )


# Initialize Chroma vector database
def get_vectorstore():
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name="university_documents",
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_FOLDER
    )

    return vectorstore


# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)


# Process and index uploaded PDF
def ingest_pdf(uploaded_file):

    # Create documents folder if it does not exist
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    # Save uploaded PDF
    file_path = os.path.join(
        DOCUMENTS_FOLDER,
        uploaded_file.name
    )

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    # Extract text from PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Add filename metadata
    for document in documents:
        document.metadata["filename"] = uploaded_file.name

    # Split extracted text into chunks
    chunks = split_documents(documents)

    # Connect to vector database
    vectorstore = get_vectorstore()

    # Delete previously indexed chunks of this PDF
    vectorstore.delete(
        where={"filename": uploaded_file.name}
    )

    # Add updated chunks
    vectorstore.add_documents(chunks)

    return f"Successfully processed {uploaded_file.name}"


# List uploaded PDF documents
def list_documents():

    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    documents = []

    for filename in os.listdir(DOCUMENTS_FOLDER):

        if filename.lower().endswith(".pdf"):
            documents.append(filename)

    return documents
