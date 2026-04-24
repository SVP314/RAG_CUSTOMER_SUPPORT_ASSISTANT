from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, PERSIST_DIR, COLLECTION_NAME


def main():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print(f"Loaded {len(docs)} pages")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    print("Loading embedding model...")
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Storing in Chroma...")
    Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    print("Ingestion complete. Vector store created successfully.")


if __name__ == "__main__":
    main()
