"""
nlp_processor.py
----------------
Core NLP utility module — handles text cleaning, tokenization, lemmatization,
and skill extraction from resume + JD text.

Used by: scoring_engine.py and app.py directly for radar chart data.

TODO: try spaCy's en_core_web_sm as an alternative to NLTK lemmatizer — might
      give better results on multi-word tech terms like "machine learning"
"""

import re
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Download NLTK assets quietly if missing (first-run setup) ─────────────────
_NLTK_PACKAGES = [
    ("corpora/stopwords",    "stopwords"),
    ("tokenizers/punkt",     "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/wordnet",      "wordnet"),
]
for _find_path, _pkg_name in _NLTK_PACKAGES:
    try:
        nltk.data.find(_find_path)
    except LookupError:
        nltk.download(_pkg_name, quiet=True)

lemmatizer = WordNetLemmatizer()

# fallback stopword list in case NLTK corpus is corrupted / unavailable
_FALLBACK_STOPS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of",
    "with", "and", "or", "is", "are", "was", "were", "be",
    "been", "by", "as", "it", "this", "that", "from", "have"
}

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    # shouldn't happen after the download above, but just in case
    STOP_WORDS = _FALLBACK_STOPS


# ── Skill DB loader ───────────────────────────────────────────────────────────

def load_skill_db(filepath: str = None) -> dict:
    """Load the JSON skill taxonomy from data/skill_db.json."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "skill_db.json")

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # corrupted JSON — return empty so the app doesn't crash
            print(f"[WARN] skill_db.json parse error: {e}")
    return {}


# ── Text cleaning helpers ─────────────────────────────────────────────────────

def _strip_noise(text: str) -> str:
    """
    Remove URLs, emails, social handles, and punctuation noise from raw text.
    Runs before tokenization so these don't pollute TF-IDF vocabulary.
    """
    # strip out http/https links — they add no semantic value
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # remove email addresses (already captured by extractor.py)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Twitter / Insta handles
    text = re.sub(r"@\S+", " ", text)
    # hashtags sometimes sneak into copied JD text
    text = re.sub(r"#\S+", " ", text)
    # remove leftover punctuation but keep hyphens inside words (e.g. "full-stack")
    text = re.sub(r"[^\w\s\-]", " ", text)
    # collapse repeated hyphens that now stand alone
    text = re.sub(r"\s-\s", " ", text)
    return text


def normalize_text(text: str) -> str:
    """
    Lowercase + strip noise + collapse whitespace.
    This is the 'light' cleaning used before skill regex matching.
    """
    if not text or not text.strip():
        return ""
    text = text.lower()
    text = _strip_noise(text)
    # collapse multiple spaces/tabs/newlines into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_for_tfidf(text: str) -> str:
    """
    Full NLP pipeline: normalize -> tokenize -> remove stopwords -> lemmatize.
    Output is a clean string ready for TF-IDF vectorization.

    NOTE: single-character tokens are dropped (usually leftover punctuation
    artefacts or Roman numeral bullets like 'i', 'v').
    """
    cleaned = normalize_text(text)

    if not cleaned:
        return ""

    # tokenize — fallback to simple split if punkt fails (edge case on some
    # environments where punkt_tab is the only downloaded resource)
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()

    processed = []
    for token in tokens:
        # skip stopwords and very short tokens
        if token in STOP_WORDS or len(token) <= 1:
            continue
        # lemmatize to collapse plurals / tenses (python -> python, working -> work)
        lemma = lemmatizer.lemmatize(token)
        processed.append(lemma)

    return " ".join(processed)


# keep the old name as an alias so existing calls in scoring_engine don't break
preprocess_text = preprocess_for_tfidf


# ── Skill extraction ──────────────────────────────────────────────────────────

def _build_skill_pattern(skill: str) -> re.Pattern:
    """
    Build a case-insensitive word-boundary regex pattern for a skill string.
    Multi-word skills like "machine learning" or "react.js" need special care —
    the dot in "react.js" is a regex metachar so we escape the whole string.
    """
    return re.compile(r"\b" + re.escape(skill.lower()) + r"\b", re.IGNORECASE)


def extract_skills(text: str, skill_db: dict = None) -> dict:
    """
    Scan normalized text for known skills from the taxonomy database.
    Returns skills grouped by category AND as a flat sorted list.

    Edge case: if text is empty (e.g., scanned PDF that returned no text),
    we return empty dicts so callers don't crash.
    """
    if skill_db is None:
        skill_db = load_skill_db()

    if not text or not text.strip():
        # handling edge case for empty/scanned PDF text
        return {"by_category": {cat: [] for cat in skill_db}, "all_skills": []}

    # pad with spaces so boundary anchors work at start/end of string
    normalized = f" {normalize_text(text)} "

    found_by_category: dict = {}
    all_found: set = set()

    for category, skill_list in skill_db.items():
        matches = []
        for skill in skill_list:
            pattern = _build_skill_pattern(skill)
            if pattern.search(normalized):
                matches.append(skill)
                all_found.add(skill)
        found_by_category[category] = matches

    return {
        "by_category": found_by_category,
        "all_skills": sorted(all_found)
    }


# ── Keyword density helper (used by ats_score_calculator) ────────────────────

def compute_keyword_density(text: str, keywords: list) -> float:
    """
    What percentage of the provided keyword list appears in the text?
    Simple but useful for the ATS score penalty logic.
    """
    if not keywords or not text.strip():
        return 0.0
    normalized = normalize_text(text)
    hits = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw.lower()) + r"\b", normalized))
    return round((hits / len(keywords)) * 100, 2)
