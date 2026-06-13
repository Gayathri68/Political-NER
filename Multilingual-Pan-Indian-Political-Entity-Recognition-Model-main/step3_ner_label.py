"""
STEP 3 - NER Labeling (Full 3-Layer System)
============================================
INPUT  : output_step2/news_clean_<lang>.jsonl
OUTPUT : output_step3/news_labeled_<lang>.jsonl

HOW IT WORKS - 3 LAYERS:

  LAYER 1 — XLM-RoBERTa
      Detects entity SPANS in all 6 languages
      Returns generic labels: PER / ORG / LOC / MISC

  LAYER 2 — Context Title Check (fast, no internet needed)
      Looks at words AROUND the entity in the sentence
      If "PM", "CM", "Minister", "ಮುಖ್ಯಮಂತ್ರಿ" found near entity  → POLITICIAN
      If "BJP", "Congress", "Party" / "ಪಕ್ಷ" found               → POLITICAL_PARTY
      If "PM Kisan", "Ayushman" found                             → GOVERNMENT_SCHEME

  LAYER 3 — Wikidata API (backup, only when Layer 2 fails)
      Called ONLY for unknown PER and ORG entities
      Looks up entity on Wikipedia's knowledge graph
      Returns occupation:
          politician → POLITICIAN
          cricketer / actor / CEO → PERSON
          not found → PERSON (safe fallback)
      Results are CACHED so same name is never looked up twice

FINAL 5 LABELS:
    POLITICIAN        - Narendra Modi, Amit Shah, Revanth Reddy etc.
    PERSON            - Varsha Janaki, Virat Kohli, any non-politician
    POLITICAL_PARTY   - BJP, Congress, AAP, TMC etc.
    GOVERNMENT_SCHEME - PM Kisan, Ayushman Bharat etc.
    LOCATION          - Mumbai, Assam, India etc.

EDGE CASES HANDLED:
    "PM Narendra Modi visited"   → Layer 2 finds "PM" nearby      → POLITICIAN
    "Amit Shah announced"        → Layer 2 fails → Wikidata        → POLITICIAN
    "Hi I am Varsha Janaki"      → Layer 2 fails → Wikidata miss   → PERSON
    "Virat Kohli met Modi"       → Wikidata → occupation=cricketer → PERSON
    "BJP won the election"       → Layer 2 party keyword           → POLITICAL_PARTY
    "ಭಾರತೀಯ ಜನತಾ ಪಕ್ಷ"          → "ಪಕ್ಷ" indicator word           → POLITICAL_PARTY
    "PM Kisan launched today"    → Layer 2 scheme keyword          → GOVERNMENT_SCHEME
    "in Mumbai yesterday"        → Layer 1 LOC                     → LOCATION
"""

import json
import os
import time
import requests
from collections import Counter
from transformers import pipeline


# ================================================================
# LAYER 1 — XLM-RoBERTa NER model
# ================================================================
print("Loading NER model (XLM-RoBERTa)...")
ner = pipeline(
    "ner",
    model="Davlan/xlm-roberta-base-ner-hrl",
    aggregation_strategy="simple",
    device=-1   # CPU (no GPU needed)
)
print("Model loaded!\n")


# ================================================================
# LAYER 2A — POLITICIAN TITLE WORDS
# In all 6 languages + English
# If any title appears within 60 chars of entity → POLITICIAN
# ================================================================
POLITICIAN_TITLES = [
    # English
    "prime minister", "chief minister", "home minister",
    "finance minister", "defence minister", "foreign minister",
    "deputy chief minister", "deputy cm",
    "pm ", " pm", "cm ", " cm",
    "minister", "mp ", " mp", "mla ", " mla",
    "president", "governor", "speaker",
    "lok sabha", "rajya sabha",
    "pradhan mantri", "mukhyamantri", "mantri",
    "rashtrapati", "uparashtrapati",

    # Kannada (ಕನ್ನಡ)
    "ಮುಖ್ಯಮಂತ್ರಿ", "ಪ್ರಧಾನಮಂತ್ರಿ", "ಮಂತ್ರಿ",
    "ಶಾಸಕ", "ಸಂಸದ", "ರಾಜ್ಯಪಾಲ",

    # Malayalam (മലയാളം)
    "മുഖ്യമന്ത്രി", "പ്രധാനമന്ത്രി", "മന്ത്രി",
    "എംഎൽഎ", "എംപി", "ഗവർണർ",

    # Assamese (অসমীয়া)
    "মুখ্যমন্ত্ৰী", "প্ৰধানমন্ত্ৰী", "মন্ত্ৰী",
    "বিধায়ক", "সাংসদ", "ৰাজ্যপাল",

    # Punjabi (ਪੰਜਾਬੀ)
    "ਮੁੱਖ ਮੰਤਰੀ", "ਪ੍ਰਧਾਨ ਮੰਤਰੀ", "ਮੰਤਰੀ",
    "ਵਿਧਾਇਕ", "ਸੰਸਦ ਮੈਂਬਰ", "ਰਾਜਪਾਲ",

    # Odia (ଓଡ଼ିଆ)
    "ମୁଖ୍ୟମନ୍ତ୍ରୀ", "ପ୍ରଧାନମନ୍ତ୍ରୀ", "ମନ୍ତ୍ରୀ",
    "ବିଧାୟକ", "ସାଂସଦ", "ରାଜ୍ୟପାଳ",
]


# ================================================================
# LAYER 2B — POLITICAL PARTY KEYWORDS
# English + native script indicator words per language
# ================================================================
POLITICAL_PARTIES = {
    # National parties
    "bjp", "bharatiya janata party", "bharatiya janata",
    "inc", "indian national congress", "congress",
    "aap", "aam aadmi party",
    "tmc", "trinamool congress", "all india trinamool",
    "sp", "samajwadi party",
    "bsp", "bahujan samaj party",
    "ncp", "nationalist congress party",
    "shiv sena", "shivsena",
    "cpim", "cpi", "communist party",

    # State parties
    "ysrcp", "ysr congress",
    "tdp", "telugu desam party", "telugu desam",
    "jdu", "janata dal united",
    "rjd", "rashtriya janata dal",
    "ljp", "lok jan shakti",
    "akali dal", "shiromani akali dal", "sad",
    "dmk", "aiadmk", "anna dravida",
    "bjd", "biju janata dal",
    "trs", "telangana rashtra samithi",
    "brs", "bharat rashtra samithi",
    "aimim", "majlis",
    "jkpdp", "national conference",
    "mns", "maharashtra navnirman sena",
    "agp", "asom gana parishad",
    "aiudf", "all india united democratic front",
    "uppl", "jjp", "jannayak janata party",

    # Generic "party" word in each language — catches unknown parties too
    "party",        # English
    "ಪಕ್ಷ",         # Kannada
    "പാർട്ടി",      # Malayalam
    "দল",           # Assamese
    "ਪਾਰਟੀ",        # Punjabi
    "ଦଳ",           # Odia
}


# ================================================================
# LAYER 2C — GOVERNMENT SCHEME KEYWORDS
# ================================================================
GOVT_SCHEMES = {
    "pm kisan", "pradhan mantri kisan",
    "ayushman bharat", "pmjay",
    "pmay", "pradhan mantri awas yojana",
    "ujjwala", "pradhan mantri ujjwala",
    "swachh bharat", "clean india",
    "make in india", "digital india",
    "startup india", "skill india",
    "jan dhan", "pradhan mantri jan dhan",
    "beti bachao", "beti padhao",
    "mudra", "pm mudra",
    "fasal bima", "pradhan mantri fasal bima",
    "mnrega", "mgnrega",
    "one nation one ration",
    "agnipath", "agniveer",
    "ladki bahin", "ladki bahin yojana",
    "national education policy", "nep",
    "jal jeevan mission", "har ghar jal",
    "saubhagya yojana", "pm garib kalyan",
    "atmanirbhar bharat", "kisan credit card",
    "poshan abhiyan", "mission indradhanush",
    "atal pension yojana", "sukanya samriddhi",
}


# ================================================================
# LAYER 3 — WIKIDATA API
# Only called when Layer 2 fails
# Results cached — same name never looked up twice
# ================================================================
CACHE_FILE = "project_outputs/wikidata_cache.json"

_wikidata_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _wikidata_cache = json.load(f)
        print(f"Loaded {len(_wikidata_cache)} cached entities from {CACHE_FILE}")
    except Exception as e:
        print(f"Failed to load cache from {CACHE_FILE}: {e}")

def save_cache():
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_wikidata_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save cache to {CACHE_FILE}: {e}")

def update_cache(key, val):
    _wikidata_cache[key] = val
    save_cache()

POLITICIAN_OCCUPATIONS = {
    "politician", "statesperson", "member of parliament",
    "member of legislative assembly", "chief minister",
    "prime minister", "president", "governor",
    "political activist", "minister", "legislator",
    "head of government", "head of state",
}

NON_POLITICIAN_OCCUPATIONS = {
    "cricketer", "cricket player", "actor", "actress",
    "film actor", "film actress", "singer", "musician",
    "sportsperson", "athlete", "businessperson",
    "entrepreneur", "scientist", "writer", "author",
    "journalist", "judge", "lawyer",
}

WIKIDATA_API = "https://www.wikidata.org/w/api.php"


def query_wikidata(entity_text: str) -> str:
    """
    Looks up entity_text on Wikidata.
    Returns: "POLITICIAN" / "PERSON" / "POLITICAL_PARTY" / "UNKNOWN"
    """
    key = entity_text.lower().strip()
    if key in _wikidata_cache:
        return _wikidata_cache[key]

    try:
        # Step 1: Search Wikidata for the entity
        resp = requests.get(WIKIDATA_API, params={
            "action": "wbsearchentities",
            "search": entity_text,
            "language": "en",
            "limit": 3,
            "format": "json",
        }, timeout=5)
        results = resp.json().get("search", [])

        if not results:
            update_cache(key, "UNKNOWN")
            return "UNKNOWN"

        entity_id = results[0].get("id", "")
        if not entity_id:
            update_cache(key, "UNKNOWN")
            return "UNKNOWN"

        # Step 2: Get claims (P31 = instance of, P106 = occupation)
        resp2  = requests.get(WIKIDATA_API, params={
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "claims",
            "format": "json",
        }, timeout=5)
        claims = (resp2.json()
                  .get("entities", {})
                  .get(entity_id, {})
                  .get("claims", {}))

        # Check P31 (instance of) — Q7278 = political party
        for claim in claims.get("P31", []):
            val = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            if isinstance(val, dict) and val.get("id") == "Q7278":
                update_cache(key, "POLITICAL_PARTY")
                time.sleep(0.5)
                return "POLITICAL_PARTY"

        # Check P106 (occupation) labels
        for claim in claims.get("P106", []):
            val = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            if not isinstance(val, dict):
                continue
            occ_id   = val.get("id", "")
            occ_resp = requests.get(WIKIDATA_API, params={
                "action": "wbgetentities",
                "ids": occ_id,
                "props": "labels",
                "languages": "en",
                "format": "json",
            }, timeout=5)
            occ_label = (
                occ_resp.json()
                .get("entities", {})
                .get(occ_id, {})
                .get("labels", {})
                .get("en", {})
                .get("value", "")
                .lower()
            )
            if any(p in occ_label for p in POLITICIAN_OCCUPATIONS):
                update_cache(key, "POLITICIAN")
                time.sleep(0.5)
                return "POLITICIAN"
            if any(p in occ_label for p in NON_POLITICIAN_OCCUPATIONS):
                update_cache(key, "PERSON")
                time.sleep(0.5)
                return "PERSON"

        update_cache(key, "UNKNOWN")
        time.sleep(0.5)
        return "UNKNOWN"

    except Exception:
        _wikidata_cache[key] = "UNKNOWN"
        return "UNKNOWN"


# ================================================================
# MAIN CLASSIFIER — all 3 layers combined
# ================================================================
def classify_entity(entity_text: str, generic_label: str, sentence: str) -> str:
    """
    Returns: POLITICIAN / PERSON / POLITICAL_PARTY /
             GOVERNMENT_SCHEME / LOCATION / O
    """
    text_lower     = entity_text.lower().strip()
    sentence_lower = sentence.lower()

    # ── LOCATION ─────────────────────────────────────────────────
    if generic_label == "LOC":
        return "LOCATION"

    # ── ORG ──────────────────────────────────────────────────────
    if generic_label == "ORG":
        for party in POLITICAL_PARTIES:
            if party in text_lower or text_lower in party:
                return "POLITICAL_PARTY"
        for scheme in GOVT_SCHEMES:
            if scheme in text_lower:
                return "GOVERNMENT_SCHEME"
        # Layer 3 backup
        wiki = query_wikidata(entity_text)
        if wiki == "POLITICAL_PARTY":
            return "POLITICAL_PARTY"
        return "O"   # Unknown ORG — skip

    # ── MISC ─────────────────────────────────────────────────────
    if generic_label == "MISC":
        for scheme in GOVT_SCHEMES:
            if scheme in text_lower:
                return "GOVERNMENT_SCHEME"
        for party in POLITICAL_PARTIES:
            if party in text_lower:
                return "POLITICAL_PARTY"
        return "O"   # Unknown MISC — skip

    # ── PER ──────────────────────────────────────────────────────
    if generic_label == "PER":

        # LAYER 2: Context title check
        entity_pos = sentence_lower.find(text_lower)
        for title in POLITICIAN_TITLES:
            if title in sentence_lower:
                title_pos = sentence_lower.find(title)
                if entity_pos != -1 and abs(title_pos - entity_pos) < 60:
                    return "POLITICIAN"

        # LAYER 3: Wikidata
        wiki = query_wikidata(entity_text)
        if wiki == "POLITICIAN":
            return "POLITICIAN"

        # Virat Kohli, Varsha Janaki, anyone else → PERSON
        return "PERSON"

    return "O"


# ================================================================
# BIO TAGGING — character offset alignment
# ================================================================
def convert_to_bio(sentence: str, ner_entities: list):
    tokens = sentence.split()
    labels = ["O"] * len(tokens)

    char_offsets = []
    cursor = 0
    for tok in tokens:
        start = sentence.find(tok, cursor)
        if start == -1:
            char_offsets.append((cursor, cursor))
            continue
        end = start + len(tok)
        char_offsets.append((start, end))
        cursor = end

    for ent in ner_entities:
        generic_label = ent.get("entity_group", "")
        entity_text   = ent.get("word", "").strip()
        custom_label  = classify_entity(entity_text, generic_label, sentence)

        if custom_label == "O":
            continue

        ent_start = ent.get("start", -1)
        ent_end   = ent.get("end",   -1)
        if ent_start < 0 or ent_end < 0:
            continue

        first_token = True
        for i, (tok_start, tok_end) in enumerate(char_offsets):
            if tok_end > ent_start and tok_start < ent_end:
                if labels[i] == "O":
                    labels[i] = ("B-" if first_token else "I-") + custom_label
                first_token = False

    return tokens, labels


# ================================================================
# PROCESS ONE LANGUAGE FILE
# ================================================================
MAX_PER_LANG = 500  # Cap per language for practical CPU runtime

def process_file(input_path: str, output_path: str, language: str):
    sentences = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sentences.append(json.loads(line))
            except Exception:
                continue

    # Cap to MAX_PER_LANG for manageable CPU runtime
    if len(sentences) > MAX_PER_LANG:
        sentences = sentences[:MAX_PER_LANG]

    total   = len(sentences)
    written = 0
    skipped = 0

    print(f"  [{language.upper()}] {total} sentences to label...", flush=True)
    print(f"  NOTE: Wikidata adds ~0.5s per unknown entity — be patient!", flush=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, record in enumerate(sentences):
            sentence = record.get("sentence", "").strip()
            if not sentence:
                continue

            if i % 50 == 0:
                print(f"    {i}/{total} ... (cache: {len(_wikidata_cache)} entities, labeled so far: {written})", flush=True)

            try:
                ner_out = ner(sentence)
            except Exception as e:
                print(f"    [WARN] Sentence {i} skipped: {e}", flush=True)
                skipped += 1
                continue

            tokens, labels = convert_to_bio(sentence, ner_out)

            if all(l == "O" for l in labels):
                skipped += 1
                continue

            fout.write(json.dumps({
                "sentence": sentence,
                "tokens":   tokens,
                "labels":   labels,
                "language": record.get("language", language),
                "source":   record.get("source", ""),
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  [{language.upper()}] Done! Labeled: {written} | Skipped: {skipped}", flush=True)
    print(f"  [{language.upper()}] Saved -> {output_path}\n", flush=True)
    return written


# ================================================================
# LABEL DISTRIBUTION REPORT
# ================================================================
def print_label_distribution(output_path: str, language: str):
    if not os.path.exists(output_path):
        return
    label_counts    = Counter()
    total_sentences = 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                total_sentences += 1
                for l in rec["labels"]:
                    if l != "O":
                        label_counts[l] += 1
            except Exception:
                continue
    print(f"\n  Label distribution [{language.upper()}] — {total_sentences} sentences:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:35s} {count}")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    os.makedirs("output_files_step3", exist_ok=True)

    files = [
        ("output_files_step2/news_clean_english.jsonl",   "output_files_step3/news_labeled_english.jsonl",   "en"),
        ("output_files_step2/news_clean_malayalam.jsonl",  "output_files_step3/news_labeled_malayalam.jsonl",  "ml"),
        ("output_files_step2/news_clean_odia.jsonl",       "output_files_step3/news_labeled_odia.jsonl",       "or"),
        ("output_files_step2/news_clean_punjabi.jsonl",    "output_files_step3/news_labeled_punjabi.jsonl",    "pa"),
        ("output_files_step2/news_clean_assamese.jsonl",   "output_files_step3/news_labeled_assamese.jsonl",   "as"),
        ("output_files_step2/news_clean_kannada.jsonl",    "output_files_step3/news_labeled_kannada.jsonl",    "kn"),
    ]

    total_labeled   = 0
    processed_files = []

    for inp, out, lang in files:
        if not os.path.exists(inp):
            print(f"  [SKIP] {inp} — file not found")
            continue
        print(f"\nProcessing: {inp}")
        count = process_file(inp, out, lang)
        total_labeled += count
        processed_files.append((out, lang))

    print("=" * 60)
    print(f"Step 3 Complete!")
    print(f"Total labeled sentences : {total_labeled}")
    print(f"Wikidata cache size     : {len(_wikidata_cache)} unique entities looked up")
    print("=" * 60)

    for out, lang in processed_files:
        print_label_distribution(out, lang)

    print("\nNext step: run Train.py")