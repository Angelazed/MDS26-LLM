# Pipeline Integration Instructions

This document provides instructions for building the embedding, retrieval, and evaluation steps. It explains exactly how to consume the output of `wiki_ingestion.py` so the pipeline fits together end-to-end with zero surprises.

## 1. Folder Layout After Ingestion Runs

```text
data/
  raw/wiki/          ← full articles as JSON (for reference / debugging)
  chunks/wiki/       ← one *_chunks.json per article  ← YOU WORK HERE

```
## 2. Loading all chunks (one call)
```r
from wiki_ingestion import WikipediaIngester

ingester = WikipediaIngester()          # points at data/raw/wiki & data/chunks/wiki
chunks   = ingester.load_all_chunks()   # list[dict], sorted by chunk_id

# chunks[i]["chunk_id"] == i  ← GUARANTEED
# This means FAISS row i maps directly to chunks[i] — no secondary lookup needed.
```
