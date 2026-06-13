"""
STEP 2 — Preprocessing all 6 language JSONL files
Input  : news_test_english.jsonl, news_test_malayalam.jsonl,
         news_test_odia.jsonl, news_test_punjabi.jsonl,
         news_test_assamese.jsonl, news_test_hindi.jsonl  (add your other 2)
Output : output/news_clean_<lang>.jsonl
         Each line: { "sentence": "...", "language": "en", "source": "..." }
"""

import json
import re
import os

# ================================================================
# Boilerplate phrases that appear at the top of The Hindu articles
# and other Indian news sources — strip these out
# ================================================================
BOILERPLATE_PHRASES = [
    "The View From India Looking at World Affairs from the Indian perspective",
    "First Day First Show News and reviews from the world of cinema",
    "Today's Cache Your download of the top 5 technology stories",
    "Science For All The weekly newsletter from science writers",
    "Data Point Decoding the headlines with facts, figures, and numbers",
    "Health Matters Ramya Kannan writes to you on getting to good health",
    "The Hindu On Books Books of the week, reviews, excerpts",
    "Activate your premium subscription today",
    "Photo credit:",
    "Photo Credit:",
    "File photo",
    "Also Read |",
    "Also read |",
    "Published -",
    "Updated -",
    "Sign in to read",
    "Subscribe now",
]

# Short noise keywords — if a sentence contains only these, skip it
NOISE_KEYWORDS = [
    "subscribe", "advertisement", "sign in", "register",
    "click here", "read more", "breaking news", "live updates",
    "follow us", "download app", "get app",
]


def remove_boilerplate(text: str) -> str:
    """Strip known newsletter/promo blocks from article body."""
    for phrase in BOILERPLATE_PHRASES:
        text = text.replace(phrase, "")
    return text


def clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters."""
    # Replace \n\n paragraph breaks with period so splitter works
    text = re.sub(r'\n\n+', '. ', text)
    text = text.replace('\n', ' ')
    # Remove excessive spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def split_sentences(text: str):
    """
    Split on . ! ? and Indian ।
    Returns list of clean sentence strings.
    """
    # Split after sentence-ending punctuation followed by space or end
    raw = re.split(r'(?<=[.!?।])\s+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


def is_noise(sentence: str) -> bool:
    """Return True if this sentence should be discarded."""
    s = sentence.lower().strip()

    # Too short
    if len(s) < 20:
        return True

    # Fewer than 4 words
    if len(s.split()) < 4:
        return True

    # Contains noise keyword as the dominant content
    if any(kw in s for kw in NOISE_KEYWORDS):
        # Only skip if the sentence is short (noise is usually short)
        if len(s.split()) < 10:
            return True

    # Looks like a URL
    if s.startswith("http"):
        return True

    # Looks like a timestamp / date only line
    if re.match(r'^(published|updated|january|february|march|april|may|june|july|august|september|october|november|december)', s):
        if len(s.split()) < 6:
            return True

    return False


def process_file(input_path: str, output_path: str, language: str):
    articles = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"  [{language}] Loaded {len(articles)} articles")

    total_sentences = 0
    kept_sentences  = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for article in articles:
            body = article.get("body") or article.get("text", "")
            if not body:
                continue

            # Step A: remove boilerplate blocks
            body = remove_boilerplate(body)

            # Step B: normalize whitespace
            body = clean_text(body)

            # Step C: split into sentences
            sentences = split_sentences(body)

            for sent in sentences:
                total_sentences += 1
                if is_noise(sent):
                    continue

                record = {
                    "sentence": sent,
                    "language": article.get("language", language),
                    "source":   article.get("source", ""),
                    "url":      article.get("url", ""),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_sentences += 1

    pct = round(kept_sentences / max(total_sentences, 1) * 100, 1)
    print(f"  [{language}] {total_sentences} sentences -> kept {kept_sentences} ({pct}%) -> {output_path}")


# ================================================================
# MAIN — add / remove files as needed for your 6 languages
# ================================================================
if __name__ == "__main__":
    os.makedirs("output_files_step2", exist_ok=True)

    files = [
        # (input file,                                      output file,                                    language code)
        ("Output_files_step1/news_test_english.jsonl",    "output_files_step2/news_clean_english.jsonl",    "en"),
        ("Output_files_step1/news_test_malayalam.jsonl",  "output_files_step2/news_clean_malayalam.jsonl",  "ml"),
        ("Output_files_step1/news_test_odia.jsonl",       "output_files_step2/news_clean_odia.jsonl",       "or"),
        ("Output_files_step1/news_test_punjabi.jsonl",    "output_files_step2/news_clean_punjabi.jsonl",    "pa"),
        ("Output_files_step1/news_test_assamese.jsonl",   "output_files_step2/news_clean_assamese.jsonl",   "as"),
        ("Output_files_step1/news_test_kannada.jsonl",    "output_files_step2/news_clean_kannada.jsonl",    "kn"),
    ]

    for inp, out, lang in files:
        if not os.path.exists(inp):
            print(f"  Skipping {inp} — file not found")
            continue
        print(f"\nProcessing: {inp}")
        process_file(inp, out, lang)

    print("\n[OK] Step 2 complete. Check output_files_step2/ folder.")
    print("  Next: run step3_ner_label.py to do NER tagging on these clean sentences.")