import re
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download essential NLTK data quietly if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {"a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "and", "or", "is", "are", "was", "were", "be", "been"}


def load_skill_db(filepath: str = None) -> dict:
    """
    Load skill taxonomy JSON database.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "skill_db.json")
        
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def clean_text(text: str) -> str:
    """
    Clean raw text: remove special symbols, extra spaces, URLs, lowercasing.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)  # remove URLs
    text = re.sub(r'#\S+', '', text)        # remove hashtags
    text = re.sub(r'@\S+', '  ', text)     # remove mentions
    text = re.sub(r'[^\w\s]', ' ', text)   # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # remove extra whitespace
    return text


def preprocess_text(text: str) -> str:
    """
    Clean, tokenize, remove stop words, and lemmatize text for TF-IDF vectorization.
    """
    cleaned = clean_text(text)
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()
        
    filtered_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in STOP_WORDS and len(word) > 1
    ]
    
    return " ".join(filtered_tokens)


def extract_skills(text: str, skill_db: dict = None) -> dict:
    """
    Extract skills present in text, grouped by category and as a combined list.
    """
    if skill_db is None:
        skill_db = load_skill_db()
        
    text_cleaned = f" {clean_text(text)} "
    found_skills_by_category = {}
    all_found_skills = set()
    
    for category, skill_list in skill_db.items():
        found_in_cat = []
        for skill in skill_list:
            skill_pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(skill_pattern, text_cleaned):
                found_in_cat.append(skill)
                all_found_skills.add(skill)
        found_skills_by_category[category] = found_in_cat

    return {
        "by_category": found_skills_by_category,
        "all_skills": sorted(list(all_found_skills))
    }
