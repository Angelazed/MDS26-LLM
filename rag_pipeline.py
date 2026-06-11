"""
RAG pipeline for F1 knowledge base.

Embeds Wikipedia chunks with a local SentenceTransformer model (all-MiniLM-L6-v2),
stores them in a FAISS index, and answers questions with a local Llama 3.1 model
served by Ollama — no API key or internet connection required at query time.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from openai import OpenAI # used as Ollama client (compatible API)
from sentence_transformers import SentenceTransformer

from wiki_ingestion import WikipediaIngester

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL = "all-MiniLM-L6-v2" # local, free, ~80 MB
CHAT_MODEL = "llama3.1" # pulled via: ollama pull llama3.1
OLLAMA_BASE = "http://localhost:11434/v1" # Ollama's OpenAI-compatible endpoint
TOP_K = 5

FAISS_INDEX_PATH = Path("data/faiss_index/f1.index")
FAISS_META_PATH = Path("data/faiss_index/f1_meta.pkl")

SYSTEM_PROMPT = """You are an expert Formula 1 analyst. Answer the user's question
using ONLY the provided context passages. If the context does not contain enough
information to answer confidently, say so clearly. Cite the article title when
you reference specific facts. Be concise and precise."""


# ---------------------------------------------------------------------------
# Vector store — SentenceTransformer + FAISS
# ---------------------------------------------------------------------------

class F1VectorStore:
    """Local vector store: SentenceTransformer embeddings + FAISS index."""

    def __init__(
        self,
        embed_model: str  = EMBED_MODEL,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_META_PATH,
    ):
        self._index_path = Path(index_path)
        self._meta_path = Path(meta_path)

        logger.info(f"Loading embedding model '{embed_model}' …")
        self._embedder = SentenceTransformer(embed_model)

        self._index: Optional[faiss.Index] = None
        self._meta:  list[dict] = [] # parallel list to FAISS rows

        # Load from disk if available
        if self._index_path.exists() and self._meta_path.exists():
            logger.info("Loading FAISS index from disk …")
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                self._meta = pickle.load(f)
            logger.info(f"  Loaded {self._index.ntotal} vectors.")

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    def is_populated(self) -> bool:
        return self.count > 0

    def ingest_chunks(self, chunks: list[dict]) -> None:
        """Embed all chunks and build a FAISS flat index."""
        logger.info(f"Embedding {len(chunks)} chunks (this runs once) …")

        texts = [c["combined_text"] for c in chunks]
        embeddings = self._embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        # Normalise for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim) # inner product on normalised vecs = cosine
        index.add(embeddings)

        self._index = index
        self._meta  = chunks # keep full chunk dicts for metadata

        # Persist
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(chunks, f)

        logger.info(f"FAISS index saved ({index.ntotal} vectors).")

    def query(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Return top-k chunks most similar to the question."""
        if not self._index:
            raise RuntimeError("Vector store is empty — call ingest_chunks() first.")

        q_vec = self._embedder.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        scores, indices = self._index.search(q_vec, top_k)

        passages = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._meta[idx]
            passages.append({
                "text": chunk["text"],
                "combined_text": chunk["combined_text"],
                "article_title": chunk["article_title"],
                "section": chunk["section"],
                "url": chunk["url"],
                "score": round(float(score), 4),
            })
        return passages


# ---------------------------------------------------------------------------
# RAG pipeline — Ollama (Llama 3.1) via OpenAI-compatible client
# ---------------------------------------------------------------------------

class F1RAGPipeline:
    """Full retrieval-augmented generation pipeline (fully local)."""

    def __init__(
        self,
        vector_store: Optional[F1VectorStore] = None,
        chat_model:   str  = CHAT_MODEL,
        ollama_base:  str  = OLLAMA_BASE,
        top_k:        int  = TOP_K,
    ):
        # Ollama exposes an OpenAI-compatible REST API — no real key needed
        self._client = OpenAI(api_key="ollama", base_url=ollama_base)
        self._store  = vector_store or F1VectorStore()
        self._model  = chat_model
        self._top_k  = top_k

    def ask(self, question: str) -> dict:
        """
        Retrieve relevant chunks and generate an answer.

        Returns:
            {
                "question": str,
                "answer": str,
                "contexts": list[str], # raw chunk texts (for RAGAS)
                "sources": list[dict], # metadata for citations
            }
        """
        passages = self._store.query(question, self._top_k)
        context_block = self._build_context(passages)
        answer = self._generate(question, context_block)

        return {
            "question": question,
            "answer":   answer,
            "contexts": [p["text"] for p in passages],
            "sources":  [
                {
                    "title": p["article_title"],
                    "section": p["section"],
                    "url": p["url"],
                    "score": p["score"],
                }
                for p in passages
            ],
        }

    def _build_context(self, passages: list[dict]) -> str:
        parts = []
        for i, p in enumerate(passages, 1):
            header = f"[{i}] {p['article_title']}"
            if p["section"]:
                header += f" › {p['section']}"
            # Prepend section title into the body so the model sees it as content
            body = f"[Section: {p['section']}]\n{p['text']}" if p["section"] else p["text"]
            parts.append(f"{header}\n{body}")
        return "\n\n---\n\n".join(parts)

    def _generate(self, question: str, context: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Convenience builder — auto-populates FAISS index if empty
# ---------------------------------------------------------------------------

def build_pipeline(chunk_dir: Optional[Path] = None) -> F1RAGPipeline:
    """
    Build a ready-to-use F1RAGPipeline.

    If the FAISS index doesn't exist yet it embeds all chunks from disk
    (one-time cost, ~1-2 min locally on M1).
    """
    store = F1VectorStore()

    if not store.is_populated():
        ingester = WikipediaIngester(
            **({"chunk_dir": chunk_dir} if chunk_dir else {})
        )
        chunks = ingester.load_all_chunks()
        store.ingest_chunks(chunks)
    else:
        logger.info(f"FAISS index already populated ({store.count} vectors). Skipping ingestion.")

    return F1RAGPipeline(vector_store=store)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is the DRS system in Formula One?"
    pipeline = build_pipeline()
    result   = pipeline.ask(question)

    print(f"\nQ: {result['question']}")
    print(f"\nA: {result['answer']}")
    print("\nSources:")
    for s in result["sources"]:
        label = f"{s['title']} › {s['section']}" if s["section"] else s["title"]
        print(f"  • {label}  (score={s['score']})")
