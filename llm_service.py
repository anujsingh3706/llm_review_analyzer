"""
llm_service.py
Handles all LLM interactions via Groq API (OpenAI-compatible).
Includes retry logic, rate limit handling, and token-aware batching.
"""

import os
import time
import logging
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is not set. Please add it to your .env file.")

client = Groq(api_key=GROQ_API_KEY)

# Rate limit config (Groq free tier: ~30 req/min)
MAX_RETRIES      = 4
BASE_BACKOFF     = 2.0   # seconds
MAX_BACKOFF      = 60.0  # seconds cap
INTER_REQUEST_DELAY = 2.5  # polite delay between every API call


# ─── Prompt Builder ───────────────────────────────────────────────────────────

def build_prompt(review_text: str, rating: int | None, author: str | None) -> str:
    """
    Build a structured prompt that asks the LLM to return:
    - Sentiment label
    - Confidence score
    - Key points (bullet list)
    - One-line summary
    """
    rating_context = f"The reviewer gave a rating of {rating}/5." if rating else ""
    author_context = f"Written by: {author}." if author else ""

    return f"""You are a product review analyst. Analyze the following customer review and return a structured analysis.

{author_context} {rating_context}

REVIEW:
\"\"\"
{review_text}
\"\"\"

Respond in exactly this format (no extra text):

SENTIMENT: <Positive | Negative | Neutral | Mixed>
CONFIDENCE: <a score from 0.0 to 1.0>
KEY_POINTS:
- <point 1>
- <point 2>
- <point 3 if applicable>
SUMMARY: <one concise sentence summarizing the review>
"""


# ─── Response Parser ──────────────────────────────────────────────────────────

def parse_llm_response(raw: str) -> dict:
    """
    Parse the structured LLM output into a Python dict.
    Gracefully handles partial or malformed responses.
    """
    result = {
        "sentiment":   "Unknown",
        "confidence":  None,
        "key_points":  [],
        "llm_summary": "",
        "raw_response": raw,
    }

    lines = raw.strip().splitlines()
    in_key_points = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("SENTIMENT:"):
            result["sentiment"] = line.replace("SENTIMENT:", "").strip()
            in_key_points = False

        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
            except ValueError:
                result["confidence"] = None
            in_key_points = False

        elif line.startswith("KEY_POINTS:"):
            in_key_points = True

        elif in_key_points and line.startswith("-"):
            result["key_points"].append(line.lstrip("- ").strip())

        elif line.startswith("SUMMARY:"):
            result["llm_summary"] = line.replace("SUMMARY:", "").strip()
            in_key_points = False

    # Flatten key points to a pipe-separated string for CSV storage
    result["key_points_str"] = " | ".join(result["key_points"])
    return result


# ─── Core API Call ────────────────────────────────────────────────────────────

def call_groq_api(prompt: str, retries: int = MAX_RETRIES) -> str | None:
    """
    Call the Groq API with exponential backoff on rate limits and errors.
    Returns raw text response or None on failure.
    """
    backoff = BASE_BACKOFF

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"  API call attempt {attempt}/{retries}...")

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,      # low temp = consistent structured output
                max_tokens=300,
            )

            raw_text = response.choices[0].message.content
            logger.info("  API call successful.")
            return raw_text

        except Exception as e:
            err_str = str(e).lower()

            # Rate limit detected
            if "rate limit" in err_str or "429" in err_str:
                sleep_time = min(backoff, MAX_BACKOFF)
                logger.warning(f"  Rate limit hit. Sleeping {sleep_time:.1f}s before retry...")
                time.sleep(sleep_time)
                backoff *= 2  # exponential backoff

            # Transient server error
            elif "500" in err_str or "502" in err_str or "503" in err_str:
                logger.warning(f"  Server error on attempt {attempt}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)

            # Auth error — no point retrying
            elif "401" in err_str or "invalid api key" in err_str:
                logger.error("  Invalid API key. Check your GROQ_API_KEY in .env")
                return None

            # Unknown error
            else:
                logger.error(f"  Unexpected error on attempt {attempt}: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)

    logger.error("  All API attempts exhausted.")
    return None


# ─── Per-Review Analyzer ──────────────────────────────────────────────────────

def analyze_review(review: dict) -> dict:
    """
    Analyze a single preprocessed review.
    If the review has multiple chunks, analyze each and merge results.
    Returns the review dict enriched with LLM analysis fields.
    """
    chunks     = review.get("chunks", [review.get("cleaned_text", "")])
    rating     = review.get("rating")
    author     = review.get("author")

    all_sentiments  = []
    all_key_points  = []
    all_summaries   = []
    all_confidences = []
    raw_responses   = []

    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        if len(chunks) > 1:
            logger.info(f"  Processing chunk {idx + 1}/{len(chunks)} for review {review.get('review_id')}")

        prompt   = build_prompt(chunk, rating, author)
        raw_resp = call_groq_api(prompt)

        if raw_resp:
            parsed = parse_llm_response(raw_resp)
            all_sentiments.append(parsed["sentiment"])
            all_key_points.extend(parsed["key_points"])
            all_summaries.append(parsed["llm_summary"])
            if parsed["confidence"] is not None:
                all_confidences.append(parsed["confidence"])
            raw_responses.append(raw_resp)
        else:
            logger.warning(f"  No response for chunk {idx + 1} of review {review.get('review_id')}")

        # Polite delay between chunk calls
        if idx < len(chunks) - 1:
            time.sleep(INTER_REQUEST_DELAY)

    # ── Merge multi-chunk results ──
    # Sentiment: majority vote; if tie → "Mixed"
    if all_sentiments:
        from collections import Counter
        counts = Counter(all_sentiments)
        top_sentiment, top_count = counts.most_common(1)[0]
        final_sentiment = top_sentiment if top_count > 1 or len(all_sentiments) == 1 else "Mixed"
    else:
        final_sentiment = "Unknown"

    final_confidence  = round(sum(all_confidences) / len(all_confidences), 3) if all_confidences else None
    final_key_points  = list(dict.fromkeys(all_key_points))[:5]  # deduplicate, cap at 5
    final_summary     = all_summaries[0] if all_summaries else "No summary generated."

    return {
        **review,
        "sentiment":       final_sentiment,
        "confidence":      final_confidence,
        "key_points":      final_key_points,
        "key_points_str":  " | ".join(final_key_points),
        "llm_summary":     final_summary,
        "raw_llm_response": "\n---\n".join(raw_responses) if raw_responses else "",
        "chunks_analyzed": len(chunks),
    }


# ─── Batch Analyzer ──────────────────────────────────────────────────────────

def analyze_all_reviews(reviews: list[dict]) -> list[dict]:
    """
    Analyze all reviews with a polite inter-request delay.
    Returns list of enriched review dicts.
    """
    results = []
    total   = len(reviews)

    for i, review in enumerate(reviews, 1):
        logger.info(f"Analyzing review {i}/{total} (ID: {review.get('review_id')} | Author: {review.get('author')})")
        enriched = analyze_review(review)
        results.append(enriched)

        # Polite delay between reviews (skip after last one)
        if i < total:
            logger.info(f"  Waiting {INTER_REQUEST_DELAY}s before next review...")
            time.sleep(INTER_REQUEST_DELAY)

    logger.info(f"LLM analysis complete. {len(results)}/{total} reviews processed.")
    return results


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_review = {
        "review_id":    "test001",
        "author":       "Test User",
        "rating":       4,
        "cleaned_text": "This book was absolutely fantastic. The story kept me engaged throughout and I loved the character development. Highly recommended!",
        "chunks":       ["This book was absolutely fantastic. The story kept me engaged throughout and I loved the character development. Highly recommended!"],
    }

    result = analyze_review(sample_review)
    print("\n─── LLM Analysis Result ───")
    print(f"Sentiment  : {result['sentiment']}")
    print(f"Confidence : {result['confidence']}")
    print(f"Key Points : {result['key_points_str']}")
    print(f"Summary    : {result['llm_summary']}")