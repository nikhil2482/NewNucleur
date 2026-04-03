"""
app.py — Flask web interface for NewNucleur RAG application.

Usage:
    python app.py
"""

import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "nuclear_docs"


def get_chroma_stats():
    """Return document count and chunk count from ChromaDB."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(COLLECTION_NAME)
        chunk_count = collection.count()
        results = collection.get(include=["metadatas"])
        docs = set()
        for meta in results["metadatas"]:
            name = meta.get("source_file") or meta.get("source", "")
            if name:
                docs.add(name)
        return {"documents": len(docs), "chunks": chunk_count, "status": "ok"}
    except Exception as e:
        return {"documents": 0, "chunks": 0, "status": "error", "error": str(e)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify(get_chroma_stats())


@app.route("/api/documents")
def api_documents():
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(COLLECTION_NAME)
        results = collection.get(include=["metadatas"])
        docs = set()
        for meta in results["metadatas"]:
            name = meta.get("source_file") or meta.get("source", "")
            if name:
                docs.add(name)
        return jsonify({"documents": sorted(docs)})
    except Exception as e:
        return jsonify({"documents": [], "error": str(e)})


def _run_query(question):
    if not question:
        return jsonify({"error": "No question provided."}), 400
    try:
        from query import answer_question
        result = answer_question(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    return _run_query(question)


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    return _run_query(question)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"NewNucleur running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
