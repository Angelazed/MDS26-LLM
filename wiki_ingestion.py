"""
Wikipedia ingestion module for F1 RAG project.

Fetches articles, cleans content, chunks text with overlap, and saves both
raw articles and chunk-ready records to disk.

Chunk format is designed to feed directly into the embedding + FAISS pipeline
described in NLP14_1_RAG_Pipeline.ipynb:
  - each chunk has a numeric `chunk_id` used as the FAISS index row
  - `combined_text` mirrors the notebook's title + text concatenation pattern
  - metadata fields (source, article_title, section, url, char_offset) survive
    into the vector store for citation and faithfulness scoring
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
import wikipediaapi
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
RAW_OUTPUT_DIR   = Path("data/raw/wiki")
CHUNK_OUTPUT_DIR = Path("data/chunks/wiki")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
 
# Chunking hyper-parameters (tune here, document in README)
DEFAULT_CHUNK_SIZE    = 512   # characters — roughly 100-130 tokens for English
DEFAULT_CHUNK_OVERLAP = 64    # characters of overlap between consecutive chunks
# ---------------------------------------------------------------------------
# F1 article list — extend as needed
# ---------------------------------------------------------------------------
 
F1_ARTICLES = [
    # Championships & seasons
    "Formula One",
    "2024 Formula One World Championship",
    "2023 Formula One World Championship",
    "Formula One World Championship",
 
    # Constructors
    "Red Bull Racing",
    "Scuderia Ferrari",
    "Mercedes-AMG Petronas Formula One Team",
    "McLaren Racing",
    "Aston Martin in Formula One",
    "Alpine F1 Team",
    "Williams Racing",
    "Haas F1 Team",
    "Visa Cash App RB Formula One Team",
    "Stake F1 Team Kick Sauber",
 
    # Drivers (current & legendary)
    "Max Verstappen",
    "Lewis Hamilton",
    "Charles Leclerc",
    "Lando Norris",
    "Carlos Sainz Jr.",
    "Fernando Alonso",
    "George Russell (racing driver)",
    "Sergio Pérez",
    "Ayrton Senna",
    "Michael Schumacher",
    "Niki Lauda",
    "Alain Prost",
 
    # Circuits
    "Circuit de Monaco",
    "Silverstone Circuit",
    "Monza Circuit",
    "Spa-Francorchamps",
    "Circuit of the Americas",
    "Suzuka International Racing Course",
    "Interlagos",
    "Yas Marina Circuit",
 
    # Technical & regulations
    "Formula One car",
    "Formula One regulations",
    "DRS (Formula One)",
    "KERS",
    "Formula One engines",
    "Formula One tyres",
 
    # History & culture
    "History of Formula One",
    "Formula One racing",
    "FIA",
]
# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
 
@dataclass
class WikiArticle:
    title: str
    url: str
    summary: str
    full_text: str
    sections: dict          # {section_title: text}
    categories: list        # list[str]
    fetched_at: str         # ISO timestamp
 
 
@dataclass
class Chunk:
    """
    One unit that will become a single row in the FAISS index.
 
    Fields mirror the notebook's knowledge_base dicts so the same
    retrieve_documents() / create_augmented_prompt() functions work
    without modification.
    """
    chunk_id:      int          # sequential integer → FAISS row index
    combined_text: str          # title + ": " + text  (notebook pattern)
    text:          str          # raw chunk text
    # --- metadata kept for citation / faithfulness scoring ---
    article_title: str
    section:       str          # section heading (empty string = lead paragraph)
    source:        str          # "wikipedia"
    url:           str
    char_offset:   int          # start position in the article's full_text
    fetched_at:    str

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
 
def clean_text(text: str) -> str:
    """Remove excess whitespace and clean Wikipedia artifacts."""
    # Remove citation brackets like [1], [22]
    text = re.sub(r'\[\d+\]', '', text)
    # Normalize multiple spacing/newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()
 
 
def extract_sections(page: wikipediaapi.WikipediaPage) -> dict:
    """Recursively extract all sections as {title: text} dict."""
    sections: dict[str, str] = {}
 
    def _recurse(section_list, depth: int = 0) -> None:
        for section in section_list:
            key = ("  " * depth + section.title).strip()
            sections[key] = clean_text(section.text)
            _recurse(section.sections, depth + 1)
 
    _recurse(page.sections)
    return sections
 
 
def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[int, str]]:
    """
    Split text into (char_offset, chunk) pairs with sliding-window overlap.
 
    Tries to break on sentence boundaries ('. ') first to avoid cutting
    mid-sentence; falls back to hard character splits if none are found.
    """
    if not text:
        return []
 
    chunks: list[tuple[int, str]] = []
    start = 0
 
    while start < len(text):
        end = min(start + chunk_size, len(text))
 
        if end < len(text):
            # Try to find a sentence boundary in the last 20 % of the window
            search_from = start + int(chunk_size * 0.8)
            boundary = text.rfind(". ", search_from, end)
            if boundary != -1:
                end = boundary + 1   # include the period
 
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, chunk))
 
        if end >= len(text):
            break
        start = end - overlap   # slide back by overlap amount
 
    return chunks

# ---------------------------------------------------------------------------
# Core fetcher
# ---------------------------------------------------------------------------
 
class WikipediaIngester:
    def __init__(
        self,
        raw_dir:    Path = RAW_OUTPUT_DIR,
        chunk_dir:  Path = CHUNK_OUTPUT_DIR,
        language:   str  = "en",
        request_delay: float = 0.5,
        chunk_size: int  = DEFAULT_CHUNK_SIZE,
        overlap:    int  = DEFAULT_CHUNK_OVERLAP,
    ):
        self.raw_dir   = Path(raw_dir)
        self.chunk_dir = Path(chunk_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
 
        self.request_delay = request_delay
        self.chunk_size    = chunk_size
        self.overlap       = overlap
 
        # Running counter so every chunk in the corpus gets a unique id
        # that maps directly to a FAISS index row
        self._chunk_counter = 0
 
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="F1-RAG-Project/1.0 (educational; contact@example.com)",
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def fetch_article(self, title: str) -> Optional[WikiArticle]:
        """Fetch a single article. Returns None on failure."""
        logger.info(f"Fetching: {title}")
        try:
            page = self.wiki.page(title)
            if not page.exists():
                logger.warning(f"  Page does not exist: '{title}'")
                return None
 
            article = WikiArticle(
                title=page.title,
                url=page.fullurl,
                summary=clean_text(page.summary),
                full_text=clean_text(page.text),
                sections=extract_sections(page),
                categories=list(page.categories.keys()),
                fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
 
            self._save_raw(article)
            chunks = self._make_chunks(article)
            self._save_chunks(article.title, chunks)
 
            logger.info(f"  → {len(chunks)} chunks saved")
            time.sleep(self.request_delay)
            return article
 
        except Exception as exc:
            logger.error(f"  Error fetching '{title}': {exc}")
            return None
 
    def fetch_all(self, titles: list[str]) -> list[WikiArticle]:
        """Fetch all titles; skip if both raw + chunk files exist."""
        results = []
        for i, title in enumerate(titles, 1):
            logger.info(f"[{i}/{len(titles)}] {title}")
 
            if self._already_saved(title):
                logger.info("  Already on disk — skipping.")
                article = self._load_raw(title)
                if article:
                    results.append(article)
                continue
 
            article = self.fetch_article(title)
            if article:
                results.append(article)
 
        logger.info(f"Done. Processed {len(results)}/{len(titles)} articles.")
        return results
 
    def load_all_chunks(self) -> list[dict]:
        """
        Load every saved chunk as a plain dict — ready to pass directly
        to SentenceTransformer.encode() and FAISS.
 
        Usage:
            ingester = WikipediaIngester()
            chunks = ingester.load_all_chunks()
            texts = [c['combined_text'] for c in chunks]
            embeddings = embedding_model.encode(texts)
        """
        all_chunks: list[dict] = []
        for path in sorted(self.chunk_dir.glob("*_chunks.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    all_chunks.extend(json.load(f))
            except Exception as exc:
                logger.error(f"Could not load {path}: {exc}")
        logger.info(f"Loaded {len(all_chunks)} total chunks from disk.")
        return all_chunks
 
    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
 
    def _make_chunks(self, article: WikiArticle) -> list[Chunk]:
        """
        Produce chunks from an article.
 
        Strategy: chunk section-by-section rather than over the full text.
        This keeps semantically coherent units together (e.g. "Career"
        stays separate from "Personal life") and makes citation easier.
        """
        chunks: list[Chunk] = []
 
        # Lead paragraph (summary) as its own mini-chunk
        if article.summary:
            for offset, text in chunk_text(article.summary, self.chunk_size, self.overlap):
                combined = f"{article.title}: {text}"
                chunks.append(Chunk(
                    chunk_id=self._chunk_counter,
                    combined_text=combined,
                    text=text,
                    article_title=article.title,
                    section="",           # empty = lead paragraph
                    source="wikipedia",
                    url=article.url,
                    char_offset=offset,
                    fetched_at=article.fetched_at,
                ))
                self._chunk_counter += 1
 
        # Section-level chunks
        for section_title, section_text in article.sections.items():
            if not section_text.strip():
                continue
            for offset, text in chunk_text(section_text, self.chunk_size, self.overlap):
                combined = f"{article.title} – {section_title}: {text}"
                chunks.append(Chunk(
                    chunk_id=self._chunk_counter,
                    combined_text=combined,
                    text=text,
                    article_title=article.title,
                    section=section_title,
                    source="wikipedia",
                    url=article.url,
                    char_offset=offset,
                    fetched_at=article.fetched_at,
                ))
                self._chunk_counter += 1
 
        return chunks
 
    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
 
    def _slug(self, title: str) -> str:
        return re.sub(r"[^\w\-]", "_", title).lower()[:100]
 
    def _raw_path(self, title: str) -> Path:
        return self.raw_dir / f"{self._slug(title)}.json"
 
    def _chunk_path(self, title: str) -> Path:
        return self.chunk_dir / f"{self._slug(title)}_chunks.json"
 
    def _already_saved(self, title: str) -> bool:
        return self._raw_path(title).exists() and self._chunk_path(title).exists()
 
    def _save_raw(self, article: WikiArticle) -> None:
        path = self._raw_path(article.title)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(article), f, ensure_ascii=False, indent=2)
        logger.info(f"  Raw  → {path}")
 
    def _save_chunks(self, title: str, chunks: list[Chunk]) -> None:
        path = self._chunk_path(title)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
        logger.info(f"  Chunks → {path}")
 
    def _load_raw(self, title: str) -> Optional[WikiArticle]:
        path = self._raw_path(title)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return WikiArticle(**data)
        except Exception as exc:
            logger.error(f"  Could not load '{title}': {exc}")
            return None

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
 
def main() -> None:
    import argparse
 
    parser = argparse.ArgumentParser(description="Fetch & chunk F1 Wikipedia articles")
    parser.add_argument("--titles", nargs="*", default=None,
                        help="Article titles (default: full F1_ARTICLES list)")
    parser.add_argument("--raw-dir",   default=str(RAW_OUTPUT_DIR))
    parser.add_argument("--chunk-dir", default=str(CHUNK_OUTPUT_DIR))
    parser.add_argument("--delay",     type=float, default=0.5)
    parser.add_argument("--chunk-size",type=int,   default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap",   type=int,   default=DEFAULT_CHUNK_OVERLAP)
    args, _ = parser.parse_known_args()
 
    ingester = WikipediaIngester(
        raw_dir=Path(args.raw_dir),
        chunk_dir=Path(args.chunk_dir),
        request_delay=args.delay,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    ingester.fetch_all(args.titles or F1_ARTICLES)


if __name__ == "__main__":
    main()

    