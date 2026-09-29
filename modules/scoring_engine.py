"""
scoring_engine.py  (was: scoring_engine.py)
-------------------------------------------
resume_matcher module — computes the hybrid match score between a resume and
a job description using SBERT + TF-IDF + skill overlap + experience metrics.

Also handles XAI explanation generation, interview question bank lookup,
and the anonymization pass before returning results to the UI.

KNOWN ISSUE: SBERT cold start takes ~3-4 seconds on first call due to model
download; subsequent calls use the cached global _SBERT_MODEL.

TODO: optimize TF-IDF vectorizer — currently fit_transform on 2 docs only,
      might benefit from a pre-fitted corpus for faster inference in batch mode
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.nlp_processor import preprocess_text, extract_skills, load_skill_db
from modules.extractor import extract_contact_info, extract_education, extract_experience_years
from modules.github_verifier import verify_github_profile

# ── SBERT lazy loader ─────────────────────────────────────────────────────────
# Using a global to avoid re-loading the 80MB model on every function call.
# Falls back to TF-IDF if sentence-transformers isn't installed.

_SBERT_MODEL = None

def _load_sbert():
    """
    Lazy-load Sentence-BERT so it doesn't block app startup.
    Returns the model object, or False if unavailable (triggers TF-IDF fallback).
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SBERT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            # sentence-transformers not installed or HuggingFace hub unreachable
            print(f"[WARN] SBERT unavailable ({e}), falling back to TF-IDF only.")
            _SBERT_MODEL = False
    return _SBERT_MODEL


# ── TF-IDF similarity ─────────────────────────────────────────────────────────

def compute_tfidf_score(resume_text: str, jd_text: str) -> float:
    """
    Calculate cosine similarity between resume and JD using TF-IDF bigrams.

    ngram_range=(1,2) captures two-word technical phrases like
    'machine learning', 'rest api', 'git workflow' — single unigrams alone
    miss a lot of domain-specific context.
    """
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(jd_text)

    # edge case: if either text is empty after preprocessing, score is 0
    if not processed_resume or not processed_jd:
        return 0.0

    try:
        # TODO: optimize TF-IDF vectorizer — pre-fitting on a larger corpus
        #       would give better IDF weights than fitting on just 2 docs
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,           # all terms matter when corpus has only 2 docs
            sublinear_tf=True   # log(1+tf) dampens effect of very frequent terms
        )
        tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_jd])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(sim) * 100, 2)
    except Exception as e:
        print(f"[ERROR] TF-IDF computation failed: {e}")
        return 0.0


def compute_sbert_score(resume_text: str, jd_text: str) -> float:
    """
    Compute semantic similarity using Sentence-BERT embeddings.
    Truncates input to 2000 chars to stay within the model's 512-token window.

    If SBERT model fails to load, transparently falls back to TF-IDF score
    so the app still works in offline/restricted environments.
    """
    model = _load_sbert()
    if not model:
        # graceful fallback — tell the caller via score that SBERT wasn't used
        return compute_tfidf_score(resume_text, jd_text)

    try:
        # truncate to 2000 chars — roughly 512 BERT tokens, good enough for resumes
        resume_chunk = resume_text[:2000]
        jd_chunk = jd_text[:2000]

        embeddings = model.encode([resume_chunk, jd_chunk], convert_to_tensor=True)
        from sentence_transformers import util
        sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        return round(max(float(sim), 0.0) * 100, 2)
    except Exception as e:
        print(f"[ERROR] SBERT scoring failed: {e}")
        # fallback so we still have a meaningful score
        return compute_tfidf_score(resume_text, jd_text)


# ── XAI explanation builder ───────────────────────────────────────────────────

def build_xai_card(
    tfidf_score: float,
    sbert_score: float,
    skill_score: float,
    exp_score: float,
    matched_skills: list,
    missing_skills: list,
    weights: dict
) -> dict:
    """
    Build the Explainable AI breakdown card shown on the recruiter dashboard.
    Positive contributors show what boosted the score; deductions highlight gaps.
    The justification_summary is a one-liner used as the natural language verdict.

    weights dict keys: w_sbert, w_tfidf, w_skills, w_exp (should sum to 1.0)
    """
    w_sbert  = weights.get("w_sbert",  0.35)
    w_tfidf  = weights.get("w_tfidf",  0.25)
    w_skills = weights.get("w_skills", 0.30)
    w_exp    = weights.get("w_exp",    0.10)

    positive_contributors = [
        f"+{round(sbert_score  * w_sbert,  1)}% SBERT Semantic Alignment",
        f"+{round(skill_score  * w_skills, 1)}% Hard Skills Overlap ({len(matched_skills)} skills matched)",
        f"+{round(tfidf_score  * w_tfidf,  1)}% TF-IDF Keyword Similarity",
        f"+{round(exp_score    * w_exp,    1)}% Experience Alignment",
    ]

    deductions = []
    if missing_skills:
        top_missing = ", ".join(missing_skills[:3])
        deductions.append(
            f"-{round((100 - skill_score) * w_skills, 1)}% Missing Skills: {top_missing}"
        )
    if exp_score < 80:
        deductions.append(
            f"-{round((100 - exp_score) * w_exp, 1)}% Experience Below Requirement"
        )

    # build a human-readable verdict sentence
    if matched_skills:
        top_matched = ", ".join(matched_skills[:4])
        summary = f"Candidate shows strong technical fit in {top_matched}."
    else:
        summary = "Candidate has general experience but limited overlap with required skills."

    if missing_skills:
        summary += f" Key gaps: {', '.join(missing_skills[:3])}."

    return {
        "positive_contributors": positive_contributors,
        "deductions": deductions if deductions else ["No major deductions — solid profile."],
        "justification_summary": summary
    }


# ── Interview question bank ───────────────────────────────────────────────────

# TODO: expand this with more domain-specific questions
# Currently covers the most commonly missing skills in our test dataset
_INTERVIEW_QB = {
    "docker":           "How do you structure multi-stage Docker builds to minimize production image size?",
    "kubernetes":       "Explain the difference between StatefulSets and Deployments in Kubernetes.",
    "aws":              "What is your approach to IAM least-privilege access and S3 bucket policy design?",
    "react":            "How does React's Virtual DOM reconciliation differ from direct DOM manipulation?",
    "python":           "How does Python's GIL affect multi-threaded vs multi-process workloads?",
    "machine learning": "How do you detect and handle class imbalance before training an ML model?",
    "sql":              "Walk me through reading and optimizing a slow SQL EXPLAIN plan.",
    "postgresql":       "What are WAL logs in PostgreSQL and how do they support point-in-time recovery?",
    "tensorflow":       "When would you use a custom training loop with GradientTape over model.fit()?",
    "pytorch":          "Explain autograd in PyTorch — when do you use torch.no_grad() and why?",
    "rest api":         "What makes an API endpoint idempotent and why does it matter for PUT vs POST?",
    "ci/cd":            "How would you design a zero-downtime blue/green deployment with GitHub Actions?",
}

_GENERIC_FALLBACK_QUESTIONS = [
    "How do you approach debugging a performance bottleneck in a production microservice?",
    "Explain how you'd set up a CI/CD pipeline for a project you haven't worked on before.",
    "What's the difference between horizontal and vertical scaling — when would you choose each?",
    "Describe a technical decision that turned out to be wrong — what did you learn from it?",
    "How do you ensure test coverage when deadlines are tight in an Agile sprint?",
]


def generate_interview_questions(missing_skills: list, jd_text: str = "") -> list:
    """
    Pick 5 interview questions based on the candidate's skill gaps.
    Falls back to generic engineering questions if the bank doesn't cover all gaps.
    """
    questions = []

    for skill in missing_skills:
        key = skill.lower()
        if key in _INTERVIEW_QB:
            questions.append(_INTERVIEW_QB[key])
        if len(questions) >= 5:
            break

    # pad with generic questions if we haven't hit 5 yet
    for q in _GENERIC_FALLBACK_QUESTIONS:
        if len(questions) >= 5:
            break
        if q not in questions:
            questions.append(q)

    return questions[:5]


# ── Main candidate analysis function ─────────────────────────────────────────

def analyze_candidate(
    resume_text: str,
    jd_text: str,
    required_skills: list = None,
    min_experience_years: float = 0.0,
    weights: dict = None,
    anonymize: bool = False
) -> dict:
    """
    Full analysis pipeline for a single candidate resume vs. a job description.

    Steps:
      1. Extract contact info, education, experience
      2. Run GitHub credibility check
      3. Extract and compare skills
      4. Compute TF-IDF and SBERT similarity scores
      5. Calculate weighted hybrid final score
      6. Classify into tier (Highly Recommended / Potential Match / Low Match)
      7. Build XAI card and interview questions
      8. Mask PII if anonymize=True
    """
    if weights is None:
        weights = {"w_sbert": 0.35, "w_tfidf": 0.25, "w_skills": 0.30, "w_exp": 0.10}

    # ── Step 1: basic profile extraction ─────────────────────────────────────
    contact_info = extract_contact_info(resume_text)
    education    = extract_education(resume_text)
    exp_years    = extract_experience_years(resume_text)

    # ── Step 2: GitHub credibility ────────────────────────────────────────────
    # tries to find a github URL in contact_info first, then scans raw text
    github_url   = contact_info.get("github", "") or resume_text
    github_audit = verify_github_profile(github_url)

    # ── Step 3: skill extraction and overlap ─────────────────────────────────
    skill_db = load_skill_db()
    resume_skills = extract_skills(resume_text, skill_db)["all_skills"]
    jd_skills     = extract_skills(jd_text, skill_db)["all_skills"]

    # use provided required_skills if given, else fall back to JD-extracted ones
    target_skills = [s.lower() for s in (required_skills if required_skills else jd_skills)]

    if target_skills:
        matched_skills = [s for s in target_skills if s in resume_skills]
        missing_skills = [s for s in target_skills if s not in resume_skills]
        skill_score    = round((len(matched_skills) / len(target_skills)) * 100, 2)
    else:
        # no JD skills found — give a moderate default, don't penalize hard
        matched_skills = resume_skills
        missing_skills = []
        skill_score    = 70.0

    extra_skills = [s for s in resume_skills if s not in target_skills]

    # ── Step 4: NLP similarity scores ────────────────────────────────────────
    tfidf_score = compute_tfidf_score(resume_text, jd_text)
    sbert_score = compute_sbert_score(resume_text, jd_text)

    # ── Step 5: experience score ──────────────────────────────────────────────
    if min_experience_years > 0:
        exp_score = min(round((exp_years / min_experience_years) * 100, 2), 100.0)
    else:
        # no minimum set — if they have any experience, score 100, else 75
        exp_score = 100.0 if exp_years >= 1.0 else 75.0

    # ── Step 6: weighted hybrid final score ───────────────────────────────────
    w_sbert  = weights.get("w_sbert",  0.35)
    w_tfidf  = weights.get("w_tfidf",  0.25)
    w_skills = weights.get("w_skills", 0.30)
    w_exp    = weights.get("w_exp",    0.10)

    # normalize weights in case sliders don't sum to exactly 1.0
    total_w = w_sbert + w_tfidf + w_skills + w_exp
    if total_w > 0:
        w_sbert  /= total_w
        w_tfidf  /= total_w
        w_skills /= total_w
        w_exp    /= total_w

    overall_score = round(
        (sbert_score  * w_sbert) +
        (tfidf_score  * w_tfidf) +
        (skill_score  * w_skills) +
        (exp_score    * w_exp),
        2
    )

    # ── Step 7: tier classification ───────────────────────────────────────────
    if overall_score >= 75.0:
        tier = "Highly Recommended"
    elif overall_score >= 55.0:
        tier = "Potential Match"
    else:
        tier = "Low Match"

    # ── Step 8: XAI card + interview questions ────────────────────────────────
    xai_explanation    = build_xai_card(
        tfidf_score, sbert_score, skill_score, exp_score,
        matched_skills, missing_skills, weights
    )
    interview_questions = generate_interview_questions(missing_skills, jd_text)

    # ── Step 9: anonymization ─────────────────────────────────────────────────
    display_email    = contact_info["email"]
    display_phone    = contact_info["phone"]
    display_linkedin = contact_info["linkedin"]
    display_github   = contact_info["github"]

    if anonymize:
        display_email    = "[ANONYMIZED]"
        display_phone    = "[ANONYMIZED]"
        display_linkedin = "[ANONYMIZED]"
        display_github   = "[ANONYMIZED]"

    return {
        "contact_info": {
            "email":    display_email,
            "phone":    display_phone,
            "linkedin": display_linkedin,
            "github":   display_github,
        },
        "education":          education,
        "experience_years":   exp_years,
        "overall_score":      overall_score,
        "tfidf_score":        tfidf_score,
        "sbert_score":        sbert_score,
        "skill_score":        skill_score,
        "exp_score":          exp_score,
        "edu_score":          80.0,   # TODO: implement proper education tier scoring
        "tier":               tier,
        "extracted_skills":   resume_skills,
        "matched_skills":     matched_skills,
        "missing_skills":     missing_skills,
        "extra_skills":       extra_skills,
        "github_audit":       github_audit,
        "xai_explanation":    xai_explanation,
        "interview_questions": interview_questions,
        "anonymized":         anonymize,
    }
