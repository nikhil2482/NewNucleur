"""
ingest.py — Parse PDFs in ./data/, chunk them, embed, and store in ChromaDB.

Usage:
    python ingest.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = Path("./data")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "nuclear_docs"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE = 500        # words (approximate; splitter works in chars, so ~2500 chars)
CHUNK_OVERLAP = 50      # words overlap (~250 chars)
CHARS_PER_WORD = 5


def get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env file.")
        sys.exit(1)
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)


def ingest():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"Created {DATA_DIR}/ — drop your PDFs there and re-run.")
        return

    pdf_files = sorted(DATA_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}/ or its subfolders. Drop some PDFs there and re-run.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to ingest:\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * CHARS_PER_WORD,
        chunk_overlap=CHUNK_OVERLAP * CHARS_PER_WORD,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        print(f"    Pages loaded: {len(pages)}")

        # Tag each page with source metadata before splitting
        rel_path = pdf_path.relative_to(DATA_DIR)
        for page in pages:
            page.metadata["source_file"] = str(rel_path)
            # PyPDFLoader sets page_number as 0-indexed; convert to 1-indexed
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1

        chunks = splitter.split_documents(pages)
        # Propagate source metadata to every chunk
        for chunk in chunks:
            chunk.metadata.setdefault("source_file", str(rel_path))
            chunk.metadata.setdefault("page_number", chunk.metadata.get("page", 0) + 1)

        print(f"    Chunks created: {len(chunks)}")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    print("Embedding and storing in ChromaDB (this may take a moment)...")

    embeddings = get_embeddings()

    # Chroma.from_documents will create or overwrite the collection
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"\nDone! {len(all_chunks)} chunks stored in '{CHROMA_DIR}/' under collection '{COLLECTION_NAME}'.")
    print("You can now run: python app.py")


if __name__ == "__main__":
    ingest()
