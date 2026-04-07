# NewNuclear

AI-powered nuclear engineering document assistant. Ask questions about your uploaded PDFs — DoE Fundamentals Handbooks, NRC ADAMS documents, plant procedures, or any technical reference — and get answers grounded exclusively in your documents, with full source citations (document name + page number).

No hallucinations. Every claim traces back to a specific page.

---

## How it works

1. You drop PDFs into `data/`
2. `ingest.py` parses each page, chunks it, embeds it with OpenAI, and stores it in a local ChromaDB vector database
3. `app.py` serves a web UI where you type a question
4. The system retrieves the 5 most relevant chunks via semantic search, passes them to GPT-4o-mini with a strict system prompt that forbids answering from training data, and returns the answer with source citations

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd NewNucleur
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Copy the example env file and fill in your key:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...your-key-here...
```

### 4. Add PDF documents

Drop any nuclear engineering PDFs into the `data/` folder:

```
data/
  doe-fundamentals-handbook-nuclear-physics.pdf
  nrc-adams-procedure-12345.pdf
  plant-procedure-rev3.pdf
```

### 5. Ingest the documents

```bash
python ingest.py
```

This parses, chunks, embeds, and stores everything in `chroma_db/`. Output will show progress per file and total chunk count.

### 6. Start the web app

```bash
python app.py
```

Open your browser to [http://localhost:5000](http://localhost:5000).

---

## Adding new documents

1. Drop the new PDF into `data/`
2. Re-run `python ingest.py` — it rebuilds the entire vector DB

---

## CLI usage (no browser required)

You can query from the terminal directly:

```bash
python query.py
```

Type a question at the prompt. Type `quit` to exit.

---

## Teammate setup

Each teammate runs the full setup independently. The `data/` and `chroma_db/` directories are gitignored — share PDFs out-of-band (shared drive, S3, etc.). Each person ingests their own local copy.

```bash
git clone <repo>
cd NewNucleur
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your own OPENAI_API_KEY
# copy PDFs into data/
python ingest.py
python app.py
```

---

## On-premises / Ollama (no OpenAI)

To run fully offline, swap in a local Ollama model via environment variables in `.env`:

```
LLM_MODEL=ollama/llama3
EMBEDDING_MODEL=ollama/nomic-embed-text
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_API_KEY=ollama   # required by the client but ignored by Ollama
```

Then start Ollama with the appropriate models pulled and run the app as normal.

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI (or Ollama) API key |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `OPENAI_API_BASE` | *(OpenAI default)* | Override for local/Ollama endpoint |
| `PORT` | `5000` | Web server port |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |

---

## Project structure

```
NewNucleur/
├── .env                  ← API keys (gitignored)
├── .env.example          ← Template for .env
├── .gitignore
├── requirements.txt
├── README.md
├── data/                 ← Your PDFs (gitignored)
├── chroma_db/            ← Auto-generated vector DB (gitignored)
├── ingest.py             ← PDF ingestion pipeline
├── query.py              ← RAG query engine + CLI
└── app.py                ← Flask web UI
```
