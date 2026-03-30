"""
preprocessor.py
Cleans review text, fixes encoding issues, and chunks long reviews
so they stay within LLM token limits.
"""

import re
import unicodedata
import logging
import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Token limit per chunk sent to the LLM (leave headroom for prompt + response)
MAX_TOKENS_PER_CHUNK = 400
ENCODING_MODEL = "cl100k_base"  # Compatible with GPT-3.5/4 and Groq LLaMA


# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def fix_encoding(text: str) -> str:
    """Normalize unicode and fix common encoding artifacts."""
    # Normalize to NFC (composed form)
    text = unicodedata.normalize("NFC", text)
    # Replace common mojibake patterns
    replacements = {
        "\u2019": "'", "\u2018": "'",   # curly apostrophes
        "\u201c": '"', "\u201d": '"',   # curly quotes
        "\u2013": "-", "\u2014": "--",  # en/em dash
        "\u2026": "...",                # ellipsis
        "\xa0": " ",                    # non-breaking space
        "\u200b": "",                   # zero-width space
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline:
    1. Fix encoding artifacts
    2. Strip HTML remnants
    3. Remove excessive whitespace
    4. Remove non-printable characters
    5. Normalize punctuation spacing
    """
    if not text or not isinstance(text, str):
        return ""

    text = fix_encoding(text)

    # Remove any leftover HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove non-printable / control characters (keep newlines)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)

    # Collapse multiple spaces/tabs into one
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize punctuation spacing (no space before . , ! ?)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    return text.strip()


# ─── Token Counting & Chunking ─────────────────────────────────────────────────

def count_tokens(text: str, model: str = ENCODING_MODEL) -> int:
    """Count tokens using tiktoken."""
    try:
        enc = tiktoken.get_encoding(model)
        return len(enc.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed ({e}), estimating by word count.")
        return int(len(text.split()) * 1.3)


def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[str]:
    """
    Split text into chunks that each fit within max_tokens.
    Splits on sentence boundaries where possible.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    # Split into sentences (basic sentence boundary detection)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = count_tokens(sentence)

        if current_tokens + s_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = s_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += s_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.debug(f"Text split into {len(chunks)} chunk(s).")
    return chunks


# ─── Main Preprocessing Pipeline ──────────────────────────────────────────────

def preprocess_reviews(reviews: list[dict]) -> list[dict]:
    """
    Run all reviews through the cleaning + chunking pipeline.
    Adds 'cleaned_text', 'token_count', and 'chunks' keys to each review.
    """
    processed = []

    for review in reviews:
        raw_text = review.get("review_text", "")
        cleaned = clean_text(raw_text)
        tokens = count_tokens(cleaned)
        chunks = chunk_text(cleaned)

        processed_review = {
            **review,
            "cleaned_text": cleaned,
            "token_count": tokens,
            "chunks": chunks,
            "chunk_count": len(chunks),
        }
        processed.append(processed_review)

    logger.info(f"Preprocessed {len(processed)} reviews.")
    avg_tokens = sum(r["token_count"] for r in processed) / max(len(processed), 1)
    logger.info(f"Average token count per review: {avg_tokens:.1f}")

    return processed


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = [
        {
            "review_id": "abc123",
            "review_text": (
                "I absolutely loved this book!!   The writing was    fantastic\u2014couldn\u2019t put it down. "
                "Highly recommend to anyone who enjoys a great read. The pacing was perfect and "
                "the characters felt very real. Would buy again without hesitation."
            )
        }
    ]
    result = preprocess_reviews(sample)
    print("Cleaned:", result[0]["cleaned_text"])
    print("Tokens:", result[0]["token_count"])
    print("Chunks:", result[0]["chunks"])