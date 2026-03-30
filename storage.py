"""
storage.py
Persists enriched review data to CSV and JSON.
Handles path creation, safe serialization, and summary statistics.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_output_dir():
    """Create output/ directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ─── DataFrame Builder ────────────────────────────────────────────────────────

def reviews_to_dataframe(reviews: list[dict]) -> pd.DataFrame:
    """
    Flatten enriched review dicts into a clean DataFrame.
    Selects and renames only the columns relevant for output.
    """
    rows = []
    for r in reviews:
        rows.append({
            "review_id":       r.get("review_id", ""),
            "product_title":   r.get("product_title", ""),
            "author":          r.get("author", ""),
            "rating":          r.get("rating", ""),
            "date":            r.get("date", ""),
            "original_review": r.get("review_text", ""),
            "cleaned_review":  r.get("cleaned_text", ""),
            "token_count":     r.get("token_count", ""),
            "chunk_count":     r.get("chunk_count", 1),
            "sentiment":       r.get("sentiment", ""),
            "confidence":      r.get("confidence", ""),
            "key_points":      r.get("key_points_str", ""),
            "llm_summary":     r.get("llm_summary", ""),
            "source_url":      r.get("source_url", ""),
        })

    df = pd.DataFrame(rows)
    logger.info(f"DataFrame built: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─── CSV Export ───────────────────────────────────────────────────────────────

def save_to_csv(df: pd.DataFrame, filename: str = None) -> str:
    """Save DataFrame to a timestamped CSV file."""
    ensure_output_dir()
    if not filename:
        filename = f"reviews_{timestamp_suffix()}.csv"
    path = os.path.join(OUTPUT_DIR, filename)

    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel compat
        logger.info(f"CSV saved: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        return ""


# ─── JSON Export ──────────────────────────────────────────────────────────────

def save_to_json(reviews: list[dict], product_metadata: dict, filename: str = None) -> str:
    """
    Save full enriched data (including raw LLM responses) to JSON.
    Includes product metadata as a top-level key.
    """
    ensure_output_dir()
    if not filename:
        filename = f"reviews_{timestamp_suffix()}.json"
    path = os.path.join(OUTPUT_DIR, filename)

    # Remove non-serializable fields (e.g. raw chunk lists with complex objects)
    clean_reviews = []
    for r in reviews:
        clean = {k: v for k, v in r.items() if k != "chunks"}
        clean_reviews.append(clean)

    payload = {
        "product_metadata": product_metadata,
        "total_reviews":    len(clean_reviews),
        "generated_at":     datetime.now().isoformat(),
        "reviews":          clean_reviews,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON saved: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        return ""


# ─── Summary Statistics ───────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, product_metadata: dict):
    """Print a readable summary report to the console."""
    print("\n" + "=" * 60)
    print("  ANALYSIS SUMMARY REPORT")
    print("=" * 60)

    print(f"\n  Product   : {product_metadata.get('product_title', 'N/A')}")
    print(f"  Price     : {product_metadata.get('price', 'N/A')}")
    print(f"  Rating    : {product_metadata.get('overall_rating', 'N/A')} / 5")
    print(f"  URL       : {product_metadata.get('scraped_url', 'N/A')}")
    print(f"  Scraped   : {product_metadata.get('scraped_at', 'N/A')}")

    print(f"\n  Total Reviews Analyzed : {len(df)}")

    if "rating" in df.columns and df["rating"].notna().any():
        avg_rating = pd.to_numeric(df["rating"], errors="coerce").mean()
        print(f"  Average Review Rating  : {avg_rating:.2f} / 5")

    if "sentiment" in df.columns:
        print("\n  Sentiment Breakdown:")
        counts = df["sentiment"].value_counts()
        for sentiment, count in counts.items():
            bar = "█" * count
            print(f"    {sentiment:<10}: {bar} ({count})")

    if "confidence" in df.columns:
        avg_conf = pd.to_numeric(df["confidence"], errors="coerce").mean()
        if not pd.isna(avg_conf):
            print(f"\n  Avg LLM Confidence : {avg_conf:.3f}")

    if "token_count" in df.columns:
        avg_tok = pd.to_numeric(df["token_count"], errors="coerce").mean()
        print(f"  Avg Tokens/Review  : {avg_tok:.1f}")

    print("\n  Sample LLM Summaries:")
    for _, row in df.head(3).iterrows():
        print(f"\n    [{row.get('author', '?')} | ⭐{row.get('rating', '?')} | {row.get('sentiment', '?')}]")
        print(f"    {row.get('llm_summary', 'N/A')}")

    print("\n" + "=" * 60)


# ─── Master Save Function ─────────────────────────────────────────────────────

def save_all(reviews: list[dict], product_metadata: dict) -> tuple[str, str]:
    """
    Convenience function: build DataFrame, save CSV + JSON, print summary.
    Returns (csv_path, json_path).
    """
    df       = reviews_to_dataframe(reviews)
    ts       = timestamp_suffix()
    csv_path  = save_to_csv(df,      filename=f"reviews_{ts}.csv")
    json_path = save_to_json(reviews, product_metadata, filename=f"reviews_{ts}.json")

    print_summary(df, product_metadata)
    return csv_path, json_path


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_reviews = [
        {
            "review_id": "abc1", "product_title": "Test Book", "author": "Alice",
            "rating": 5, "date": "2024-01-01",
            "review_text": "Great book!", "cleaned_text": "Great book!",
            "token_count": 3, "chunk_count": 1,
            "sentiment": "Positive", "confidence": 0.95,
            "key_points_str": "Great read | Loved it",
            "llm_summary": "A highly positive review praising the book.",
            "source_url": "http://example.com",
        }
    ]
    meta = {"product_title": "Test Book", "price": "£10", "overall_rating": 5,
            "scraped_url": "http://example.com", "scraped_at": "2024-01-01"}

    csv_p, json_p = save_all(sample_reviews, meta)
    print(f"\nFiles saved:\n  CSV : {csv_p}\n  JSON: {json_p}")