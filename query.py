"""
query.py — Retrieve relevant chunks from ChromaDB and generate a grounded answer.

Usage as a module:
    from query import answer_question
    result = answer_question("What is the neutron moderation process?")

Usage as CLI:
    python query.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "nuclear_docs"
TOP_K = 5

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a nuclear engineering subject-matter expert assistant.
Your role is to answer questions strictly and exclusively from the provided document excerpts.

Rules you must follow without exception:
1. Answer ONLY from the context provided below. Do not use your training data or general knowledge.
2. If the provided context does not contain enough information to answer the question, say exactly:
   "I don't have enough information in the loaded documents to answer that question."
3. Every factual claim in your answer must be traceable to a specific source document and page number from the context.
4. Format your answer clearly. At the end, list the sources you used as:
   Sources: [Document Name, Page X], [Document Name, Page Y], ...
5. Never speculate, never hallucinate, never infer beyond what the documents say."""


def get_vectorstore():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env file.")
        sys.exit(1)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def answer_question(question: str) -> dict:
    """
    Retrieve relevant chunks and generate a grounded answer.

    Returns:
        {
            "answer": str,
            "sources": [{"file": str, "page": int, "score": float}, ...]
        }
    """
    vectorstore = get_vectorstore()

    # Similarity search with scores (lower distance = more relevant)
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(
        question, k=TOP_K
    )

    if not results_with_scores:
        return {
            "answer": "No documents are loaded in the vector database. Please run ingest.py first.",
            "sources": [],
        }

    # Build context block for the prompt
    context_parts = []
    sources = []
    seen = set()

    for doc, score in results_with_scores:
        source_file = doc.metadata.get("source_file", doc.metadata.get("source", "Unknown"))
        page_number = doc.metadata.get("page_number", doc.metadata.get("page", 0))
        if isinstance(page_number, int) and page_number == 0:
            page_number = 1

        context_parts.append(
            f"[Source: {source_file}, Page {page_number}]\n{doc.page_content}"
        )

        key = (source_file, page_number)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": source_file,
                "page": page_number,
                "score": round(score, 4),
            })

    context_text = "\n\n---\n\n".join(context_parts)

    user_message = f"""Context from loaded documents:

{context_text}

---

Question: {question}"""

    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=api_key)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    answer = response.content

    return {
        "answer": answer,
        "sources": sources,
    }


def cli():
    print("NewNucleur — Nuclear Engineering Document Assistant")
    print("=" * 52)
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        print("\nSearching documents...\n")
        result = answer_question(question)

        print("Answer:")
        print(result["answer"])
        print()

        if result["sources"]:
            print("Sources:")
            for src in result["sources"]:
                print(f"  - {src['file']}, Page {src['page']}  (relevance: {src['score']})")
        print("\n" + "-" * 52 + "\n")


if __name__ == "__main__":
    cli()
