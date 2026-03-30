"""
test_connection.py
Run this BEFORE main.py to verify your environment is correctly configured.
Checks: .env loading, Groq API key, model availability, scraper connectivity.

Usage: python test_connection.py
"""

import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
WARN  = "⚠️  WARN"
GROQ_TEST_URL = "https://api.groq.com/openai/v1/models"


def separator(title: str = ""):
    width = 58
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


# ─── Check 1: .env file ───────────────────────────────────────────────────────

def check_env_file():
    separator("ENV FILE")
    if os.path.exists(".env"):
        print(f"{PASS}  .env file found")
    else:
        print(f"{WARN}  No .env file found — using system environment variables")

    key   = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if key:
        masked = key[:6] + "*" * (len(key) - 10) + key[-4:]
        print(f"{PASS}  GROQ_API_KEY found  → {masked}")
    else:
        print(f"{FAIL}  GROQ_API_KEY is missing! Add it to your .env file.")
        return False

    print(f"{PASS}  GROQ_MODEL          → {model}")
    return True


# ─── Check 2: Python packages ─────────────────────────────────────────────────

def check_packages():
    separator("PYTHON PACKAGES")
    required = [
        ("requests",        "requests"),
        ("bs4",             "beautifulsoup4"),
        ("pandas",          "pandas"),
        ("numpy",           "numpy"),
        ("groq",            "groq"),
        ("dotenv",          "python-dotenv"),
        ("tiktoken",        "tiktoken"),
        ("fake_useragent",  "fake-useragent"),
    ]

    all_ok = True
    for module_name, pip_name in required:
        try:
            __import__(module_name)
            print(f"{PASS}  {pip_name}")
        except ImportError:
            print(f"{FAIL}  {pip_name}  →  pip install {pip_name}")
            all_ok = False

    return all_ok


# ─── Check 3: Groq API reachability ──────────────────────────────────────────

def check_groq_api():
    separator("GROQ API CONNECTIVITY")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(f"{FAIL}  Cannot test — GROQ_API_KEY not set")
        return False

    try:
        resp = requests.get(
            GROQ_TEST_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            timeout=10
        )

        if resp.status_code == 200:
            data   = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            print(f"{PASS}  Groq API reachable (HTTP 200)")
            print(f"{PASS}  Available models: {len(models)} found")

            target = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            if any(target in m for m in models):
                print(f"{PASS}  Target model '{target}' is available")
            else:
                print(f"{WARN}  Target model '{target}' not explicitly listed")
                print(f"       Available: {', '.join(models[:5])}...")
            return True

        elif resp.status_code == 401:
            print(f"{FAIL}  Authentication failed — check your GROQ_API_KEY")
            return False
        elif resp.status_code == 429:
            print(f"{WARN}  Rate limited — API key works but quota reached")
            return True
        else:
            print(f"{FAIL}  Unexpected status: {resp.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"{FAIL}  Cannot reach api.groq.com — check internet connection")
        return False
    except requests.exceptions.Timeout:
        print(f"{FAIL}  Request timed out after 10s")
        return False


# ─── Check 4: Test LLM call ───────────────────────────────────────────────────

def check_llm_call():
    separator("LIVE LLM TEST CALL")
    api_key = os.getenv("GROQ_API_KEY")
    model   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        print(f"{FAIL}  Skipping — no API key")
        return False

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        print(f"       Sending test prompt to {model}...")
        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role":    "user",
                "content": (
                    "Analyze this review in one line: "
                    "'Great product, fast delivery, would buy again!'\n"
                    "SENTIMENT: <Positive|Negative|Neutral>\n"
                    "SUMMARY: <one sentence>"
                )
            }],
            temperature=0.3,
            max_tokens=60,
        )
        elapsed  = time.time() - t0
        raw_text = response.choices[0].message.content.strip()

        print(f"{PASS}  LLM responded in {elapsed:.2f}s")
        print(f"       Model output preview:")
        for line in raw_text.splitlines():
            print(f"         {line}")
        return True

    except Exception as e:
        err = str(e).lower()
        if "rate limit" in err or "429" in err:
            print(f"{WARN}  Rate limited — API key is valid but quota hit")
            print(f"       Wait ~60s and re-run, or upgrade your Groq plan")
            return True
        elif "401" in err or "invalid api key" in err:
            print(f"{FAIL}  Invalid API key — double-check GROQ_API_KEY in .env")
            return False
        else:
            print(f"{FAIL}  Unexpected error: {e}")
            return False


# ─── Check 5: Scraper connectivity ────────────────────────────────────────────

def check_scraper():
    separator("SCRAPER CONNECTIVITY")
    test_url = "http://books.toscrape.com"

    try:
        resp = requests.get(test_url, timeout=10)
        if resp.status_code == 200:
            print(f"{PASS}  books.toscrape.com is reachable (HTTP 200)")

            from bs4 import BeautifulSoup
            soup  = BeautifulSoup(resp.content, "lxml")
            title = soup.find("title")
            print(f"{PASS}  BeautifulSoup parsed page: <title>{title.text.strip()}</title>")
            return True
        else:
            print(f"{WARN}  Site returned HTTP {resp.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"{FAIL}  Cannot reach books.toscrape.com")
        return False
    except Exception as e:
        print(f"{FAIL}  Error: {e}")
        return False


# ─── Check 6: Output directory ────────────────────────────────────────────────

def check_output_dir():
    separator("OUTPUT DIRECTORY")
    os.makedirs("output", exist_ok=True)

    if os.path.isdir("output"):
        print(f"{PASS}  output/ directory ready")

    # Quick write test
    test_file = "output/.write_test"
    try:
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        print(f"{PASS}  Write permissions confirmed")
        return True
    except Exception as e:
        print(f"{FAIL}  Cannot write to output/: {e}")
        return False


# ─── Final Report ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 58)
    print("   LLM REVIEW ANALYZER — PRE-FLIGHT CHECKS")
    print("═" * 58)

    results = {
        "Environment Variables":  check_env_file(),
        "Python Packages":        check_packages(),
        "Groq API Connectivity":  check_groq_api(),
        "Live LLM Test Call":     check_llm_call(),
        "Scraper Connectivity":   check_scraper(),
        "Output Directory":       check_output_dir(),
    }

    separator("FINAL REPORT")
    all_passed = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🚀  All checks passed! Run the pipeline with:")
        print("      python main.py")
    else:
        print("  🔧  Fix the failed checks above, then run:")
        print("      python main.py")

    print("═" * 58 + "\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())