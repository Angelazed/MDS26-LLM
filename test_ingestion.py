"""
Ingestion Verification Framework for F1 RAG Pipeline.
Run this script to test if Wikipedia data collection and chunking succeeded.
"""

import os
import sys
import json
sys.path.append(os.getcwd())
import logging
from pathlib import Path
from wiki_ingestion import WikipediaIngester, F1_ARTICLES

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestionTest")

def run_pipeline_test():
    logger.info("Starting F1 Ingestion Pipeline Verification Test...")

    # 1. Setup paths to test directories
    test_raw_dir = Path("data/raw/test_wiki")
    test_chunk_dir = Path("data/chunks/test_wiki")
    
    # 2. Select a subset of articles to test execution speed safely
    test_titles = [
        "Formula One",            # Heavy document with nested sections
        "Max Verstappen",         # Dynamic text fields 
        "Circuit de Monaco"       # Geographic/historical breakdown
    ]
    
    # 3. Initialize Ingester with a clean User-Agent configuration
    ingester = WikipediaIngester(
        raw_dir=test_raw_dir,
        chunk_dir=test_chunk_dir,
        request_delay=1.0,        # Safe delay for local verification runs
        chunk_size=512,
        overlap=64
    )
    
    # Force override user agent string directly to guarantee clearance
    ingester.wiki.headers = {
        "User-Agent": "F1RAGUniversityProject/1.0 (student.evaluation@icatt.it)"
    }

    # 4. Execute Fetching Process
    logger.info(f"Fetching target validation subset: {test_titles}")
    fetched_articles = ingester.fetch_all(test_titles)
    
    # -------------------------------------------------------------------------
    # Assertions & System Diagnostics
    # -------------------------------------------------------------------------
    print("\n" + "="*50 + "\n📊 INGESTION PIPELINE DIAGNOSTIC REPORT\n" + "="*50)
    
    # Test 1: Did any pages download?
    assert len(fetched_articles) > 0, "❌ CRITICAL FAILURE: No articles were successfully fetched from Wikipedia API."
    print(f"✅ Success: Fetched {len(fetched_articles)}/{len(test_titles)} test target articles.")

    # Test 2: File Persistence Verification
    for title in test_titles:
        slug_name = ingester._slug(title)
        raw_file = test_raw_dir / f"{slug_name}.json"
        chunk_file = test_chunk_dir / f"{slug_name}_chunks.json"
        
        assert raw_file.exists(), f"❌ Missing raw storage file for: {title}"
        assert chunk_file.exists(), f"❌ Missing processed chunk data file for: {title}"
        
        # Test 3: Structural Integrity Check inside JSON schemas
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
            
        assert isinstance(chunk_data, list), f"❌ Chunk file for {title} is corrupted or not formatted as a list."
        assert len(chunk_data) > 0, f"❌ Chunking pipeline returned 0 chunks for: {title}"
        
        # Validate critical metadata layout required for downstream FAISS indexing
        sample_chunk = chunk_data[0]
        required_keys = ["chunk_id", "combined_text", "text", "article_title", "section", "source", "url"]
        for key in required_keys:
            assert key in sample_chunk, f"❌ Missing key '{key}' in chunk layout for {title}."
            
        print(f" -> Article '{title}' parsed beautifully into {len(chunk_data)} chunks. Metadata schema looks clean.")

    # Test 4: Verify the Global Chunk Aggregator Loader
    all_loaded_chunks = ingester.load_all_chunks()
    print(f"\n✅ Pipeline Integrity Secured: Global corpus aggregation tool successfully retrieved {len(all_loaded_chunks)} total text blocks.")
    print("🚀 Your ingestion framework is 100% ready to pipe into your FAISS Vector store matrix construction.")
    print("="*50)

if __name__ == "__main__":
    run_pipeline_test()