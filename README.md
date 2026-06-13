# Formula One RAG Knowledge Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system that answers Formula One questions using a curated Wikipedia corpus, local vector search, and a locally-served language model.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [System Overview](#2-system-overview)
3. [Design Decisions & Alternatives Considered](#3-design-decisions--alternatives-considered)
4. [Project Structure](#4-project-structure)
5. [How to Run](#5-how-to-run)
6. [Evaluation Results & Interpretation](#6-evaluation-results--interpretation)
7. [Known Limitations](#7-known-limitations)
8. [AI Usage Disclosure](#8-ai-usage-disclosure)

---

## 1. Problem Framing

Formula One is a knowledge-intensive domain with a large volume of structured facts: race results, driver statistics, technical regulations, circuit specifications, and historical records. A general-purpose language model handles F1 questions poorly for two reasons:

- **Knowledge cutoff**: training data does not include the most recent seasons
- **Hallucination**: models confidently generate plausible but incorrect statistics (lap records, championship points, race outcomes)

We framed this as a **domain-specific question answering problem** where the system must retrieve grounded evidence before generating an answer, making every claim traceable to a source. This ruled out pure generation approaches (fine-tuning, prompting a general LLM) in favour of RAG, which keeps the knowledge base separate from the model and auditable.

We chose Formula One specifically because it combines factual precision (dates, names) with narrative complexity (race strategies, season arcs), making it a meaningful stress test for both retrieval quality and generation faithfulness.

---

## 2. System Overview

The system is built around three stages.

In the first stage, we collected 41 Wikipedia articles covering F1 drivers, teams, circuits, seasons, and technical topics. Each article is cleaned and split into around 4,300 short passages of roughly 512 characters, section by section. These passages are saved to disk and serve as the knowledge base.

In the second stage, each passage is converted into a vector (a list of numbers representing its meaning) using a local embedding model. All vectors are stored in a FAISS index on disk so this step only runs once.

In the third stage, when a user asks a question, it is converted into a vector using the same embedding model. FAISS finds the 5 most similar passages. Those passages are passed to Llama 3.1 — running locally via Ollama — along with the instruction to answer only from the provided context. The model returns an answer with cited sources.

---

## 3. Design Decisions & Alternatives Considered

### 3.1 Why RAG over fine-tuning?

Fine-tuning a language model on F1 data would embed knowledge into weights, making it expensive to update and impossible to audit. RAG keeps the knowledge in a retrievable corpus, so adding a new season is a matter of ingesting new articles, not retraining. For a domain with frequent updates (race results, regulation changes), this is the more practical architecture.

### 3.2 Corpus: Wikipedia only

We considered three sources:

- **Wikipedia** — broad coverage, clean structure, freely available
- **FIA Technical Regulations PDF** — highly technical, but very domain-specific and would skew the system toward regulation queries
- **Official F1 race reports** — most up-to-date, but require scraping and have inconsistent formatting

We chose Wikipedia because it covers all four corpus categories (drivers, teams, circuits, seasons) uniformly, has a stable structure that maps well to section-level chunking, and is reproducible by anyone running the project.

### 3.3 Chunking strategy: section-by-section, 512 characters

We rejected full-article chunking because a single article (e.g. Max Verstappen, ~15,000 characters) would produce chunks that mix unrelated content — career statistics alongside personal life — degrading retrieval precision.

We chose **section-level chunking** so that each chunk represents one coherent topic. Within each section, we apply a sliding window of 512 characters with 64-character overlap, breaking on sentence boundaries where possible to avoid cutting mid-fact.

The `combined_text` field prepends the article title and section name to every chunk before embedding (`"Max Verstappen – Career: …"`). This ensures the vector carries topic context even for short or ambiguous passages.

### 3.4 Embedding model: multi-qa-MiniLM-L6-cos-v1

We considered three options:

- **`text-embedding-3-small` (OpenAI)** — strong performance, but requires an API key and charges per use. We ruled it out because we wanted the system to run fully locally at no cost.
- **`all-MiniLM-L6-v2` (local, free)** — our first choice. In practice it underperformed on factual Q&A — for example, the query *"Which team does Lando Norris race for?"* did not retrieve the intro paragraph chunk that explicitly states *"competes in Formula One for McLaren"*, because it was trained for sentence similarity rather than question-answer matching.
- **`multi-qa-MiniLM-L6-cos-v1` (local, free)** — our final choice. Same size (~80MB) and speed as the previous model, but trained on 215 million question-answer pairs specifically for semantic search. Switching to this model resolved the retrieval failures we observed.

| Model | Cost | Trained for | Result |
|---|---|---|---|
| `text-embedding-3-small` | Paid API | General embeddings | Ruled out — not free |
| `all-MiniLM-L6-v2` | Free | Sentence similarity | Poor Q&A retrieval |
| `multi-qa-MiniLM-L6-cos-v1` | Free | Question-answer retrieval | Good Q&A retrieval |

### 3.5 Vector store: FAISS

We chose FAISS because it is a single-file index with no server process, aligns with the course reference implementation, and is fast enough for 4,300 vectors with exact search. Metadata (article title, section, URL) is stored in a parallel Python list that maps 1:1 to FAISS row indices.

We initially designed the pipeline with ChromaDB (which stores metadata natively), but switched to FAISS for simplicity and course alignment.

### 3.6 Generation model: Llama 3.1 8B via Ollama

| Model | Cost | Quality | Privacy |
|---|---|---|---|
| GPT-4o-mini (OpenAI API) | ~$0.001/query | Very high | Data sent externally |
| GPT-2 (local) | Free | Poor | Local |
| Llama 3.1 8B (Ollama) | Free | Good | Local |

We rejected GPT-4o-mini because it requires an API key and sends data to external servers. We rejected GPT-2 because it was not designed for instruction following. Llama 3.1 8B runs entirely locally on Apple Silicon, produces coherent and grounded answers, and requires no API key.

### 3.7 top-k = 5

We tested top-k values of 5, 8, and 15. Higher values introduced the "lost in the middle" problem — Llama 3.1 8B tends to ignore information positioned in the middle of a long context. With top-k = 5 the context is short enough for the model to attend to all passages.

---

## 4. Project Structure

```
.
├── wiki_ingestion.py          # Fetch & chunk Wikipedia articles
├── rag_pipeline.py            # Vector store, retrieval, generation
├── evaluation.py              # Retrieval and generation metrics
├── app.py                     # Streamlit chat interface
├── data_ingestion_wiki.ipynb  # Ingestion notebook
├── rag_pipeline.ipynb         # Pipeline walkthrough notebook
├── evaluation_demo.ipynb      # Evaluation results notebook
├── data/
│   ├── raw/wiki/              # Full article JSON files
│   └── chunks/wiki/           # Chunked records (~4,300 chunks)
└── requirements.txt
```

---

## 5. How to Run

### Step 1 — Install Ollama (once)

Download and install Ollama from the official website: **[https://ollama.com/download](https://ollama.com/download)**

Available for macOS, Windows, and Linux.

### Step 2 — Pull the Llama model (once, ~4.7 GB download)

```bash
ollama pull llama3.1
```

### Step 3 — Create and activate the virtual environment (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal line.

### Step 4 — Install Python dependencies (once)

```bash
pip install -r requirements.txt
```

---

### Running the Streamlit app

You need two terminals open at the same time.

**Terminal 1 — start Ollama (keep this running)**
```bash
ollama serve
```

**Terminal 2 — start the app**
```bash
source .venv/bin/activate
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

The first time you run it, the pipeline will download the embedding model (~80MB) and build the FAISS index from the chunks (~1-2 min). Every run after that starts instantly.

---

### Asking questions from the terminal (CLI)

```bash
source .venv/bin/activate
python rag_pipeline.py "What is the DRS system in Formula One?"
```

Ollama must be running in another terminal (`ollama serve`).

---

### Running the evaluation notebook

```bash
source .venv/bin/activate
jupyter notebook evaluation_demo.ipynb
```

Inside the notebook, select the **F1 RAG (.venv)** kernel (Kernel → Change Kernel). If it is not listed, register it first:

```bash
.venv/bin/python -m ipykernel install --user --name=f1-rag --display-name "F1 RAG (.venv)"
```

Then restart Jupyter and select the kernel. Ollama must be running before executing the generation and latency cells.

---

### Re-running data ingestion (optional — data already on disk)

```bash
source .venv/bin/activate
python wiki_ingestion.py
```

---

### Quick reference

| What you want to do | venv needed | Ollama needed |
|---|---|---|
| Run the Streamlit app | ✓ | ✓ |
| Ask a question (CLI) | ✓ | ✓ |
| Run the evaluation notebook | ✓ | ✓ |
| Re-run data ingestion | ✓ | X|

---

## 6. Evaluation Results & Interpretation

We evaluated the system on 5 F1 questions covering drivers, circuits, engines, history, and regulations. Full per-question results are in `evaluation_outputs.ipynb`.

### Retrieval metrics (Precision@5, Recall@5, MRR)

These measure whether the right articles are being retrieved. Precision@5 measures how many of the 5 retrieved chunks came from the expected article. Recall@5 measures how many of the expected articles appeared in the top 5. MRR (Mean Reciprocal Rank) measures how early the first relevant chunk appears — a score of 1.0 means it is always the top result.

| Metric | Score |
|---|---|
| Precision@5 | 0.880 |
| Recall@5 | 1.000 |
| MRR | 1.000 |

The expected source appeared as the top result in every single query (MRR = 1.0). Recall@5 is perfect — the correct article was always retrieved. Precision@5 of 0.88 means that on average 4.4 of the 5 retrieved chunks came from the expected article; the small gap is because some queries retrieved one chunk from a related but non-target article.

### Generation metrics (Faithfulness, Relevance, Correctness, scale 1–5)

- **Faithfulness**: token overlap between the generated answer and the retrieved context. A high score means the model is answering from what it retrieved, not from memory.
- **Relevance**: cosine similarity between the question and answer embeddings. A high score means the answer addresses the question.
- **Correctness**: cosine similarity between the generated answer and the ground truth answer.

| Metric | Average score (1–5) |
|---|---|
| Faithfulness | 4.4 |
| Relevance | 4.2 |
| Correctness | 4.4 |

All three generation metrics averaged above 4 out of 5, indicating that the model consistently uses the retrieved context, stays on topic, and produces answers that match the expected ground truth.

### Latency

| Stage | Average time |
|---|---|
| Retrieval (FAISS) | 0.082 seconds |
| Generation (Llama 3.1 8B) | 2.708 seconds |

Retrieval is near-instant since FAISS performs exact search over 4,300 vectors in memory. Generation dominates latency at around 2.7 seconds per query on Apple Silicon (M1/M2). This is significantly faster than the 5–15 seconds observed during initial testing, likely due to Ollama model warm-up caching.

### Source citation

| Check | Result |
|---|---|
| Answers containing citations | 100% |
| Answers citing the expected source | 100% |

Every answer included at least one source, and in every case the expected article was present in the returned sources — meaning the system always surfaces the correct attribution to the user.

### What the results tell us

The system performs well on questions with self-contained answers in a single chunk (technical definitions, driver biographies, circuit descriptions). Retrieval is reliable and fast; generation is faithful to the context. It performs weaker on questions whose answers are distributed across multiple chunks or expressed indirectly. This is a known limitation of fixed-size chunking and a direction for future improvement.

---

## 7. Known Limitations

**Fragmented answers**: some facts are described narratively across multiple chunks rather than stated directly. The embedding model may not rank the most relevant chunk highest if the answer is split at a chunk boundary.

**Lost in the middle**: Llama 3.1 8B tends to under-attend to information positioned in the middle of a long context. We mitigated this by keeping top-k at 5.

**No temporal reasoning**: the system cannot answer questions that require comparing across seasons (e.g. "which driver improved the most between 2023 and 2024?") because such reasoning requires synthesising multiple chunks.

**Wikipedia only**: the corpus does not include race-day telemetry, lap times, or official FIA documents, limiting the depth of technical answers.

---

## 8. AI Usage Disclosure

AI tools (Claude) assisted this project in the following ways:

- **Boilerplate scaffolding**: initial class structures for `WikipediaIngester`, `F1VectorStore`, and `F1RAGPipeline`
- **Debugging**: resolving FAISS API usage, chunk metadata alignment, and Ollama client configuration

All design decisions, such as corpus selection, model choices and evaluation methodology, were made by the project group. Every piece of code was reviewed and tested by the team. 
