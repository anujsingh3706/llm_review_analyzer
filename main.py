"""
main.py
End-to-end pipeline orchestrator.
Run: python main.py
  or: python main.py --url "http://books.toscrape.com/catalogue/..."
"""

import argparse
import logging
import sys
import time

from scraper       import scrape_reviews, PRODUCT_URL
from preprocessor  import preprocess_reviews
from llm_service   import analyze_all_reviews
from storage       import save_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(url: str):
    start = time.time()

    print("\n" + "═" * 60)
    print("   LLM REVIEW ANALYZER — PIPELINE START")
    print("═" * 60)

    # ── Step 1: Scrape ──────────────────────────────────────────
    print("\n[1/4] Scraping product reviews...")
    reviews, product_metadata = scrape_reviews(url)

    if not reviews:
        logger.error("No reviews scraped. Exiting.")
        sys.exit(1)

    print(f"      ✓ Scraped {len(reviews)} reviews from: {product_metadata.get('product_title')}")

    # ── Step 2: Preprocess ──────────────────────────────────────
    print("\n[2/4] Preprocessing review text...")
    preprocessed = preprocess_reviews(reviews)
    total_chunks  = sum(r["chunk_count"] for r in preprocessed)
    print(f"      ✓ Cleaned {len(preprocessed)} reviews | {total_chunks} total chunk(s) to analyze")

    # ── Step 3: LLM Analysis ────────────────────────────────────
    print(f"\n[3/4] Sending to Groq LLM ({__import__('os').getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')})...")
    print(f"      (Rate-limit safe delays applied between calls)\n")
    enriched = analyze_all_reviews(preprocessed)

    successful = sum(1 for r in enriched if r.get("sentiment") not in ("Unknown", None, ""))
    print(f"\n      ✓ LLM analyzed {successful}/{len(enriched)} reviews successfully")

    # ── Step 4: Save ────────────────────────────────────────────
    print("\n[4/4] Saving results...")
    csv_path, json_path = save_all(enriched, product_metadata)
    print(f"      ✓ CSV  → {csv_path}")
    print(f"      ✓ JSON → {json_path}")
    print(f"      ✓ Log  → pipeline.log")

    elapsed = time.time() - start
    print(f"\n{'═' * 60}")
    print(f"   PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'═' * 60}\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Review Analyzer Pipeline")
    parser.add_argument(
        "--url",
        type=str,
        default=PRODUCT_URL,
        help="Product page URL to scrape (default: Books to Scrape sandbox)"
    )
    args = parser.parse_args()
    run_pipeline(args.url)