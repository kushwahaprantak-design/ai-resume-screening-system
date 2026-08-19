import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.nlp_processor import preprocess_text, extract_skills, load_skill_db
from modules.extractor import extract_contact_info, extract_education, extract_experience_years
from modules.github_verifier import verify_github_profile

_SBERT_MODEL = None

def get_sbert_model():
    """
    Lazy loader for Sentence-BERT model ('all-MiniLM-L6-v2').
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SBERT_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer model ({e}). Using TF-IDF fallback.")
            _SBERT_MODEL = False
    return _SBERT_MODEL


def calculate_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """
    Calculate Cosine Similarity using TF-IDF N-grams (1,2).
    """
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(jd_text)
    
    if not processed_resume or not processed_jd:
        return 0.0
        
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_jd])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(sim) * 100, 2)
    except Exception:
        return 0.0


def calculate_sbert_similarity(resume_text: str, jd_text: str) -> float:
    """
    Calculate Semantic Similarity using Sentence-BERT ('all-MiniLM-L6-v2') Embeddings.
    """
    model = get_sbert_model()
    if not model:
        return calculate_tfidf_similarity(resume_text, jd_text)

    try:
        clean_res = resume_text[:2000]
        clean_jd = jd_text[:2000]

        embeddings = model.encode([clean_res, clean_jd], convert_to_tensor=True)
        from sentence_transformers import util
        sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        return round(max(float(sim), 0.0) * 100, 2)
    except Exception as e:
        print(f"SBERT calculation error: {e}")
        return calculate_tfidf_similarity(resume_text, jd_text)


def generate_xai_explanation(
    tfidf_score: float,
    sbert_score: float,
    skill_score: float,
    exp_score: float,
    matched_skills: list,
    missing_skills: list,
    weights: dict
) -> dict:
    """
    Generate Explainable AI (XAI) score contributors, deductions, and natural language justification summary.
    """
    w_sbert = weights.get("w_sbert", 0.35)
    w_tfidf = weights.get("w_tfidf", 0.25)
    w_skills = weights.get("w_skills", 0.30)
    w_exp = weights.get("w_exp", 0.10)

    positive_contributors = [
        f"+{round(sbert_score * w_sbert, 1)}% SBERT Semantic Alignment",
        f"+{round(skill_score * w_skills, 1)}% Hard Skills Overlap ({len(matched_skills)} skills present)",
        f"+{round(tfidf_score * w_tfidf, 1)}% TF-IDF Technical Keyword Similarity",
        f"+{round(exp_score * w_exp, 1)}% Experience Alignment"
    ]

    deductions = []
    if missing_skills:
        missing_str = ", ".join(missing_skills[:3])
        deductions.append(f"-{round((100 - skill_score) * w_skills, 1)}% Missing Target Skills: {missing_str}")
    if exp_score < 80:
        deductions.append(f"-{round((100 - exp_score) * w_exp, 1)}% Experience Below Position Requirement")

    # Natural Language Justification Summary
    if matched_skills:
        top_matched = ", ".join(matched_skills[:4])
        summary = f"Candidate demonstrates strong technical alignment in {top_matched}."
    else:
        summary = "Candidate has general experience but lacks core required skill overlap."

    if missing_skills:
        top_missing = ", ".join(missing_skills[:3])
        summary += f" However, candidate shows skill gaps in {top_missing}."

    return {
        "positive_contributors": positive_contributors,
        "deductions": deductions if deductions else ["No major score deductions identified."],
        "justification_summary": summary
    }


def generate_interview_questions(missing_skills: list, jd_text: str) -> list:
    """
    Generate 5 candidate-specific technical interview questions based on missing skills and JD.
    """
    question_bank = {
        "docker": "How do you structure multi-stage Docker builds to reduce container image size in production?",
        "kubernetes": "Explain how Kubernetes StatefulSets differ from Deployments and how storage volumes are bound.",
        "aws": "What strategies do you use for secure IAM policy management and AWS S3 bucket encryption?",
        "react": "Explain React's Virtual DOM reconciliation process and how useMemo optimizes re-renders.",
        "python": "How does Python handle memory management and garbage collection with cyclic references?",
        "machine learning": "How do you detect and mitigate class imbalance during model training?",
        "sql": "Explain execution plan optimization and how composite indexes affect query performance.",
        "postgresql": "What are PostgreSQL WAL logs and how do they support point-in-time recovery?",
        "tensorflow": "How do custom gradient tape functions work in TensorFlow 2.x?",
        "pytorch": "Explain autograd dynamics in PyTorch and when torch.no_grad() should be applied.",
        "rest api": "What are idempotency requirements in REST API design for POST vs PUT operations?",
        "ci/cd": "How do you implement zero-downtime blue/green deployment pipelines in Jenkins or GitHub Actions?"
    }

    questions = []
    # 1. Add questions for missing skills
    for skill in missing_skills:
        skill_lower = skill.lower()
        if skill_lower in question_bank:
            questions.append(question_bank[skill_lower])
        if len(questions) >= 5:
            break

    # 2. Generic technical questions if fewer than 5 generated
    generic_fallback = [
        "Explain how you handle exception logging and debugging in high-concurrency microservices.",
        "What metrics do you evaluate to determine whether an ML model is overfitting in training?",
        "How do you design database schemas to maintain 3rd Normal Form while preserving query speed?",
        "Describe a challenging technical architectural decision you made in your previous project.",
        "How do you approach unit testing and code coverage in an Agile CI/CD setup?"
    ]

    for q in generic_fallback:
        if len(questions) >= 5:
            break
        if q not in questions:
            questions.append(q)

    return questions[:5]


def analyze_candidate(
    resume_text: str,
    jd_text: str,
    required_skills: list = None,
    min_experience_years: float = 0.0,
    weights: dict = None,
    anonymize: bool = False
) -> dict:
    """
    Perform multi-factor hybrid candidate analysis, GitHub verification, XAI breakdown, and question generation.
    """
    if weights is None:
        weights = {"w_sbert": 0.35, "w_tfidf": 0.25, "w_skills": 0.30, "w_exp": 0.10}

    # 1. Contact & Basic Info Extraction
    contact_info = extract_contact_info(resume_text)
    education = extract_education(resume_text)
    exp_years = extract_experience_years(resume_text)

    # 2. GitHub Credibility Analysis
    github_audit = verify_github_profile(contact_info.get("github", "") or resume_text)

    # 3. Skill Extraction & Matching
    skill_db = load_skill_db()
    extracted_resume_skills = extract_skills(resume_text, skill_db)["all_skills"]
    extracted_jd_skills = extract_skills(jd_text, skill_db)["all_skills"]

    target_skills = [s.lower() for s in (required_skills if required_skills else extracted_jd_skills)]
    
    if target_skills:
        matched_skills = [s for s in target_skills if s in extracted_resume_skills]
        missing_skills = [s for s in target_skills if s not in extracted_resume_skills]
        skill_score = round((len(matched_skills) / len(target_skills)) * 100, 2)
    else:
        matched_skills = extracted_resume_skills
        missing_skills = []
        skill_score = 70.0

    extra_skills = [s for s in extracted_resume_skills if s not in target_skills]

    # 4. Compute TF-IDF and SBERT Scores
    tfidf_score = calculate_tfidf_similarity(resume_text, jd_text)
    sbert_score = calculate_sbert_similarity(resume_text, jd_text)

    # 5. Experience Score
    if min_experience_years > 0:
        exp_score = min(round((exp_years / min_experience_years) * 100, 2), 100.0)
    else:
        exp_score = 100.0 if exp_years >= 1.0 else 75.0

    # 6. Overall Hybrid Score
    w_sbert = weights.get("w_sbert", 0.35)
    w_tfidf = weights.get("w_tfidf", 0.25)
    w_skills = weights.get("w_skills", 0.30)
    w_exp = weights.get("w_exp", 0.10)

    tot_w = w_sbert + w_tfidf + w_skills + w_exp
    if tot_w > 0:
        w_sbert /= tot_w
        w_tfidf /= tot_w
        w_skills /= tot_w
        w_exp /= tot_w

    overall_score = round(
        (sbert_score * w_sbert) + 
        (tfidf_score * w_tfidf) + 
        (skill_score * w_skills) + 
        (exp_score * w_exp), 
        2
    )

    # 7. Status Tier Classification
    if overall_score >= 75.0:
        tier = "Highly Recommended"
    elif overall_score >= 55.0:
        tier = "Potential Match"
    else:
        tier = "Low Match"

    # 8. XAI Explanation & Question Generation
    xai_explanation = generate_xai_explanation(
        tfidf_score, sbert_score, skill_score, exp_score, matched_skills, missing_skills, weights
    )
    interview_questions = generate_interview_questions(missing_skills, jd_text)

    # 9. Anonymization Masking
    display_email = contact_info["email"]
    display_phone = contact_info["phone"]
    display_linkedin = contact_info["linkedin"]
    display_github = contact_info["github"]

    if anonymize:
        display_email = "[ANONYMIZED EMAIL]"
        display_phone = "[ANONYMIZED PHONE]"
        display_linkedin = "[ANONYMIZED PROFILE]"
        display_github = "[ANONYMIZED PROFILE]"

    return {
        "contact_info": {
            "email": display_email,
            "phone": display_phone,
            "linkedin": display_linkedin,
            "github": display_github
        },
        "education": education,
        "experience_years": exp_years,
        "overall_score": overall_score,
        "tfidf_score": tfidf_score,
        "sbert_score": sbert_score,
        "skill_score": skill_score,
        "exp_score": exp_score,
        "edu_score": 80.0,
        "tier": tier,
        "extracted_skills": extracted_resume_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
        "github_audit": github_audit,
        "xai_explanation": xai_explanation,
        "interview_questions": interview_questions,
        "anonymized": anonymize
    }
