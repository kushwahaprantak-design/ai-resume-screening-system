"""
Platform AI Co-Pilot / Assistant Chatbot Module.

Provides an agentic, context-aware chatbot that inspects Streamlit session_state
to understand the user's current portal, scoring weights, threshold, anonymization
settings, and uploaded data — then delivers actionable guidance and trade-off analysis.

Architecture:
    - Rule-based NLU intent classifier with keyword matching.
    - Dynamic session state inspector for contextual responses.
    - Optional LLM API hook (OpenAI / Google Gemini) for free-form queries.
    - Streamlit st.chat_message / st.chat_input native UI integration.
"""

import streamlit as st
import re
from datetime import datetime

# ------------------------------------------------------------------
# System Prompt (used if an external LLM API is connected)
# ------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are the Intelligent Co-Pilot for the Resume Screening & Candidate Analytics Engine. "
    "Your job is to guide HR recruiters and job candidates, explain the technical and hiring "
    "consequences of changing system settings (weights, thresholds, anonymization), and provide "
    "actionable recommendations. Be concise, professional, and clear about trade-offs."
)

# ------------------------------------------------------------------
# Intent Classification Keywords
# ------------------------------------------------------------------
INTENT_KEYWORDS = {
    "weight_sbert": [
        "sbert", "sentence-bert", "sentence bert", "semantic", "embedding",
        "dense", "transformer", "meaning", "context"
    ],
    "weight_tfidf": [
        "tfidf", "tf-idf", "tf idf", "keyword", "exact match",
        "term frequency", "vectorizer", "ngram", "n-gram"
    ],
    "weight_skills": [
        "hard skill", "skill match", "skill weight", "technical skill",
        "skill overlap", "programming", "framework"
    ],
    "weight_experience": [
        "experience", "years", "exp weight", "seniority", "work history"
    ],
    "threshold": [
        "threshold", "minimum score", "cutoff", "qualification",
        "filter", "shortlist", "false negative", "false positive"
    ],
    "anonymization": [
        "anonymiz", "anonymis", "bias", "privacy", "blind",
        "mask", "name", "gender", "fair", "dei", "diversity"
    ],
    "ats_resume": [
        "ats", "resume", "optimize", "keyword", "bullet",
        "formatting", "readability", "parse", "score"
    ],
    "cover_letter": [
        "cover letter", "application letter", "cover", "letter"
    ],
    "interview": [
        "interview", "question", "preparation", "mock", "behavioral"
    ],
    "recruiter_jd": [
        "job description", "jd", "recruiter", "posting",
        "requirement", "position", "role"
    ],
    "how_it_works": [
        "how does", "how do", "explain", "architecture",
        "algorithm", "scoring", "formula", "pipeline", "system"
    ],
    "status": [
        "status", "current", "settings", "configuration",
        "what are my", "show me", "dashboard", "state"
    ],
    "help": [
        "help", "guide", "what can you", "capabilities",
        "assist", "support", "menu", "options"
    ],
    "greeting": [
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "greetings"
    ]
}


def _classify_intent(user_message: str) -> str:
    """Classify user message into a known intent using keyword matching."""
    msg_lower = user_message.lower().strip()

    # Score each intent by counting keyword hits
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in msg_lower:
                score += len(kw)  # Longer matches weighted higher
        if score > 0:
            scores[intent] = score

    if not scores:
        return "unknown"

    return max(scores, key=scores.get)


def _get_context_snapshot() -> dict:
    """Capture a snapshot of current session state for contextual responses."""
    return {
        "portal": st.session_state.get("_copilot_portal", "Unknown"),
        "weights": st.session_state.get("_copilot_weights", {
            "w_sbert": 0.35, "w_tfidf": 0.25, "w_skills": 0.30, "w_exp": 0.10
        }),
        "threshold": st.session_state.get("_copilot_threshold", 40),
        "anonymize": st.session_state.get("_copilot_anonymize", False),
        "has_results": "screening_results" in st.session_state,
        "has_candidate": "cand_analysis" in st.session_state,
    }


def _format_pct(value: float) -> str:
    """Format a 0-1 float weight as a readable percentage."""
    return f"{value * 100:.0f}%"


# ------------------------------------------------------------------
# Response Generators (one per intent)
# ------------------------------------------------------------------

def _respond_greeting(ctx: dict) -> str:
    portal = ctx["portal"]
    return (
        f"Welcome to the Platform Co-Pilot. You are currently in the **{portal}** portal.\n\n"
        "I can help you with:\n"
        "- Understanding scoring algorithm weight trade-offs\n"
        "- Threshold and bias configuration impact analysis\n"
        "- Resume optimization and ATS compliance guidance\n"
        "- Recruiter JD construction best practices\n"
        "- Interview preparation strategies\n\n"
        "Ask me anything about the system or type **help** for a full command reference."
    )


def _respond_help(ctx: dict) -> str:
    return (
        "**Platform Co-Pilot Command Reference**\n\n"
        "| Topic | Example Questions |\n"
        "|---|---|\n"
        "| Scoring Weights | *\"What happens if I increase SBERT weight?\"* |\n"
        "| Threshold Impact | *\"What if I raise the threshold to 70%?\"* |\n"
        "| Anonymization | *\"Should I enable anonymization?\"* |\n"
        "| Resume / ATS | *\"How do I optimize my resume for ATS?\"* |\n"
        "| Cover Letter | *\"Help me write a cover letter\"* |\n"
        "| Interview Prep | *\"Give me interview tips\"* |\n"
        "| JD Construction | *\"How should I write a job description?\"* |\n"
        "| System Status | *\"Show me my current settings\"* |\n"
        "| Architecture | *\"How does the scoring algorithm work?\"* |\n"
    )


def _respond_status(ctx: dict) -> str:
    w = ctx["weights"]
    lines = [
        "**Current System Configuration Snapshot**\n",
        f"- **Active Portal:** {ctx['portal']}",
        f"- **Sentence-BERT Weight:** {_format_pct(w.get('w_sbert', 0))}",
        f"- **TF-IDF Keyword Weight:** {_format_pct(w.get('w_tfidf', 0))}",
        f"- **Hard Skill Match Weight:** {_format_pct(w.get('w_skills', 0))}",
        f"- **Experience Match Weight:** {_format_pct(w.get('w_exp', 0))}",
        f"- **Qualification Threshold:** {ctx['threshold']}%",
        f"- **Anonymization Enabled:** {'Yes' if ctx['anonymize'] else 'No'}",
        f"- **Screening Results Available:** {'Yes' if ctx['has_results'] else 'No'}",
        f"- **Candidate Analysis Available:** {'Yes' if ctx['has_candidate'] else 'No'}",
    ]
    return "\n".join(lines)


def _respond_weight_sbert(ctx: dict) -> str:
    current = _format_pct(ctx["weights"].get("w_sbert", 0))
    return (
        f"**Sentence-BERT Weight Analysis** (Current: {current})\n\n"
        "**What Sentence-BERT measures:**\n"
        "Sentence-BERT generates dense 384-dimensional embeddings that capture the *semantic meaning* "
        "of text. It understands synonyms, paraphrases, and contextual relevance — not just exact words.\n\n"
        "**If you increase this weight:**\n"
        "- Candidates who describe equivalent experience using different terminology will score higher.\n"
        "- A resume mentioning *\"built REST APIs\"* will match a JD requiring *\"developed web services\"*.\n"
        "- Reduces false negatives from keyword mismatch.\n"
        "- **Trade-off:** May rank candidates who write eloquently but lack specific hard skills higher than "
        "technically precise candidates.\n\n"
        "**If you decrease this weight:**\n"
        "- The system relies more on exact keyword overlap (TF-IDF) and hard skill enumeration.\n"
        "- Better for roles with strict, non-negotiable technical requirements (e.g., specific certifications).\n\n"
        "**Recommendation:** For creative and cross-functional roles, keep SBERT at 30-40%. "
        "For compliance-heavy roles (medical, legal), reduce to 15-20%."
    )


def _respond_weight_tfidf(ctx: dict) -> str:
    current = _format_pct(ctx["weights"].get("w_tfidf", 0))
    return (
        f"**TF-IDF Keyword Weight Analysis** (Current: {current})\n\n"
        "**What TF-IDF measures:**\n"
        "Term Frequency-Inverse Document Frequency counts exact keyword overlap between the resume and "
        "job description. It rewards candidates who use the *same specific terms* as the JD.\n\n"
        "**If you increase this weight:**\n"
        "- Candidates using identical JD terminology rank higher.\n"
        "- ATS-optimized resumes with keyword stuffing may score disproportionately well.\n"
        "- **Trade-off:** Strong candidates who use synonyms or industry-equivalent terms get penalized. "
        "A candidate writing *\"Machine Learning\"* might score lower if the JD says *\"AI/ML Engineering\"*.\n\n"
        "**If you decrease this weight:**\n"
        "- Reduces keyword-gaming advantage.\n"
        "- Relies more on semantic understanding (SBERT) and demonstrated skill overlap.\n\n"
        "**Recommendation:** Keep TF-IDF at 20-30% as a keyword safety net. "
        "Never rely on it alone — it cannot understand context."
    )


def _respond_weight_skills(ctx: dict) -> str:
    current = _format_pct(ctx["weights"].get("w_skills", 0))
    return (
        f"**Hard Skill Match Weight Analysis** (Current: {current})\n\n"
        "**What Hard Skill Match measures:**\n"
        "Compares extracted technical skills (Python, Docker, React, AWS, etc.) from the resume against "
        "the JD's required skills using our 500+ skill taxonomy database.\n\n"
        "**If you increase this weight:**\n"
        "- Only candidates with explicitly listed matching technologies rank high.\n"
        "- Excellent for engineering roles where specific stack proficiency is mandatory.\n"
        "- **Trade-off:** Penalizes candidates with transferable skills who could learn the required "
        "stack quickly. A strong Java developer might score poorly for a Python role despite similar fundamentals.\n\n"
        "**If you decrease this weight:**\n"
        "- Allows broader candidate pools with adjacent skillsets.\n"
        "- Better for generalist or leadership roles where adaptability matters.\n\n"
        "**Recommendation:** Set to 25-35% for specialized engineering roles. "
        "Reduce to 15-20% for management, product, or cross-functional positions."
    )


def _respond_weight_experience(ctx: dict) -> str:
    current = _format_pct(ctx["weights"].get("w_exp", 0))
    return (
        f"**Experience Match Weight Analysis** (Current: {current})\n\n"
        "**What Experience Match measures:**\n"
        "Extracts years of experience from the resume using pattern matching and compares against "
        "the JD's minimum experience requirement.\n\n"
        "**If you increase this weight:**\n"
        "- Senior candidates with extensive tenure get a strong scoring boost.\n"
        "- **Trade-off:** Discriminates against exceptional early-career candidates, career changers, "
        "and bootcamp graduates who may outperform on skills despite fewer years.\n\n"
        "**If you decrease this weight:**\n"
        "- Opens the pipeline to high-potential junior candidates.\n"
        "- Better for startups and fast-moving teams valuing skill over seniority.\n\n"
        "**Recommendation:** Keep at 5-15% unless hiring for strictly senior-level positions. "
        "Experience years alone are a poor predictor of competence."
    )


def _respond_threshold(ctx: dict) -> str:
    current = ctx["threshold"]
    return (
        f"**Qualification Threshold Impact Analysis** (Current: {current}%)\n\n"
        "The threshold determines the minimum score a candidate must achieve to appear in the "
        "qualified shortlist and detailed XAI breakdown.\n\n"
        f"**At {current}% threshold:**\n"
        f"- {'This is a permissive threshold. Most candidates will pass, giving recruiters a broad pool to review manually.' if current <= 40 else ''}"
        f"{'This is a moderate threshold. Balances pipeline size with quality filtering.' if 40 < current <= 60 else ''}"
        f"{'This is a strict threshold. Only strong-match candidates will appear. Risk of filtering out unconventional but talented candidates.' if 60 < current <= 80 else ''}"
        f"{'This is a very aggressive threshold. Expect very few candidates to qualify. High risk of false negatives.' if current > 80 else ''}\n\n"
        "**Raising the threshold:**\n"
        "- Smaller shortlist, less manual review work for recruiters.\n"
        "- Higher risk of false negatives (strong candidates filtered out due to resume formatting or terminology mismatch).\n\n"
        "**Lowering the threshold:**\n"
        "- Larger, more inclusive candidate pool.\n"
        "- More manual screening effort required.\n"
        "- Better for diversity hiring and roles with limited applicant pools.\n\n"
        "**Recommendation:** Start at 40-50% for initial screening, then tighten to 60-70% after reviewing the score distribution."
    )


def _respond_anonymization(ctx: dict) -> str:
    enabled = ctx["anonymize"]
    return (
        f"**Candidate Anonymization Analysis** (Currently: {'Enabled' if enabled else 'Disabled'})\n\n"
        "**What anonymization does:**\n"
        "Masks personally identifiable information — candidate name, email, phone number, LinkedIn URL, "
        "and gender-indicating details — replacing them with neutral identifiers like *Candidate #101*.\n\n"
        "**Benefits of enabling:**\n"
        "- Mitigates unconscious bias based on name, gender, ethnicity, or institutional prestige.\n"
        "- Forces evaluation based purely on skills, experience, and qualifications.\n"
        "- Aligns with DEI (Diversity, Equity & Inclusion) best practices.\n"
        "- Recommended for initial screening rounds.\n\n"
        "**Trade-offs:**\n"
        "- Recruiters cannot see contact details during early review.\n"
        "- Cannot assess cultural fit cues or personal branding.\n"
        "- GitHub profile verification results are still visible (skill-based, not identity-based).\n\n"
        "**Recommendation:** Enable anonymization for the first screening pass to build an unbiased shortlist. "
        "Disable it only after candidates are shortlisted for interviews."
    )


def _respond_ats_resume(ctx: dict) -> str:
    return (
        "**ATS Resume Optimization Guide**\n\n"
        "**1. Use Standard Section Headers:**\n"
        "- Use conventional headings: *Experience*, *Education*, *Skills*, *Projects*, *Certifications*.\n"
        "- Avoid creative headers like *\"My Journey\"* or *\"What I Bring\"* — ATS parsers cannot categorize them.\n\n"
        "**2. Mirror JD Keywords Naturally:**\n"
        "- Read the target JD and incorporate its exact technical terms in your bullet points.\n"
        "- If the JD says *\"React.js\"*, use *\"React.js\"* — not just *\"React\"* or *\"frontend framework\"*.\n\n"
        "**3. Quantify Achievements:**\n"
        "- Replace: *\"Improved application performance\"*\n"
        "- With: *\"Reduced API response latency by 40% (450ms to 270ms) by implementing Redis caching layer\"*\n\n"
        "**4. Avoid Formatting Pitfalls:**\n"
        "- No tables, columns, or text boxes — they break PDF parsers.\n"
        "- Use a single-column layout with clear hierarchy.\n"
        "- Save as PDF (not DOCX) for consistent rendering.\n\n"
        "**5. Include a Technical Skills Section:**\n"
        "- List technologies explicitly: *Python, TensorFlow, Docker, AWS EC2, PostgreSQL*.\n"
        "- This directly feeds into our Hard Skill Match scoring component.\n\n"
        "Upload your resume in the **Candidate Workspace** to get a detailed ATS compliance audit."
    )


def _respond_cover_letter(ctx: dict) -> str:
    return (
        "**Cover Letter Best Practices**\n\n"
        "Our system generates a tailored 3-paragraph executive cover letter in the "
        "**Career Advisor & Interview Prep** tab. Here are tips for maximum impact:\n\n"
        "**Paragraph 1 — Strong Hook:**\n"
        "- Lead with your most relevant qualification and genuine enthusiasm for the specific role.\n"
        "- Mention the company by name and reference something specific about their work.\n\n"
        "**Paragraph 2 — Evidence & Impact:**\n"
        "- Highlight 2-3 achievements with quantified metrics that directly align with JD requirements.\n"
        "- Use the STAR format: Situation, Task, Action, Result.\n\n"
        "**Paragraph 3 — Confident Close:**\n"
        "- Express eagerness for a conversation, not desperation for the job.\n"
        "- Include a clear call-to-action: *\"I welcome the opportunity to discuss how my experience in X aligns with your goals.\"*\n\n"
        "Upload your resume and specify a target role to generate a tailored draft automatically."
    )


def _respond_interview(ctx: dict) -> str:
    return (
        "**Interview Preparation Strategy**\n\n"
        "Our system generates candidate-specific practice questions based on skill gaps and project experience. "
        "Here is a general preparation framework:\n\n"
        "**Technical Questions:**\n"
        "- Expect deep-dives into technologies listed on your resume.\n"
        "- Practice explaining your projects: *architecture decisions, trade-offs, and outcomes*.\n"
        "- Review fundamentals of any skills listed in the JD that you are less confident in.\n\n"
        "**Behavioral Questions (STAR Method):**\n"
        "- *\"Tell me about a time you handled a tight deadline.\"*\n"
        "- *\"Describe a conflict with a teammate and how you resolved it.\"*\n"
        "- Structure: Situation → Task → Action → Result (with metrics).\n\n"
        "**System Design (for senior roles):**\n"
        "- Practice designing scalable systems: load balancers, caching layers, database sharding.\n"
        "- Think out loud — interviewers evaluate your reasoning process, not just the answer.\n\n"
        "Run the **Candidate Workspace** analysis to get personalized questions based on your resume gaps."
    )


def _respond_recruiter_jd(ctx: dict) -> str:
    return (
        "**Job Description Construction Best Practices**\n\n"
        "A well-structured JD dramatically improves our scoring engine's accuracy:\n\n"
        "**1. Separate Must-Have vs. Nice-to-Have Skills:**\n"
        "- Clearly label required vs. preferred qualifications.\n"
        "- Our skill matcher extracts and weights these differently.\n\n"
        "**2. Be Specific About Technologies:**\n"
        "- Write *\"Python 3.x, FastAPI, PostgreSQL, Docker\"* instead of *\"modern tech stack\"*.\n"
        "- Specific terms improve TF-IDF and Hard Skill matching accuracy.\n\n"
        "**3. Include Experience Context:**\n"
        "- Specify minimum years and the *type* of experience expected.\n"
        "- *\"3+ years building production ML pipelines\"* is far more useful than *\"3+ years experience\"*.\n\n"
        "**4. Role-Specific Weight Recommendations:**\n\n"
        "| Role Type | SBERT | TF-IDF | Skills | Exp |\n"
        "|---|---|---|---|---|\n"
        "| Software Engineer | 30% | 25% | 35% | 10% |\n"
        "| Data Scientist | 35% | 20% | 30% | 15% |\n"
        "| Product Manager | 40% | 20% | 20% | 20% |\n"
        "| DevOps / SRE | 25% | 25% | 40% | 10% |\n"
        "| Junior / Intern | 35% | 25% | 35% | 5% |\n"
    )


def _respond_how_it_works(ctx: dict) -> str:
    return (
        "**System Architecture Overview**\n\n"
        "The Resume Screening Engine uses a 4-component hybrid scoring pipeline:\n\n"
        "**1. Sentence-BERT Semantic Similarity:**\n"
        "- Encodes resume and JD into 384-dim dense vectors using `all-MiniLM-L6-v2`.\n"
        "- Computes cosine similarity to measure overall contextual alignment.\n\n"
        "**2. TF-IDF Keyword Overlap:**\n"
        "- Builds term-frequency vectors from both documents.\n"
        "- Measures exact keyword co-occurrence via cosine similarity.\n\n"
        "**3. Hard Skill Taxonomy Match:**\n"
        "- Extracts technical skills using a curated 500+ skill database across 6 categories.\n"
        "- Computes intersection ratio between resume skills and JD requirements.\n\n"
        "**4. Experience Alignment:**\n"
        "- Regex-extracts years of experience from the resume.\n"
        "- Compares against JD minimum experience threshold.\n\n"
        "**Final Score Formula:**\n"
        "```\n"
        "Score = (w_sbert × S_SBERT) + (w_tfidf × S_TFIDF) + (w_skills × S_Skills) + (w_exp × S_Exp)\n"
        "```\n"
        "All weights are configurable via the sidebar sliders and auto-normalized to sum to 100%.\n\n"
        "Visit **System Analytics & Reference** for detailed architecture diagrams and mathematical formulations."
    )


def _respond_unknown(ctx: dict) -> str:
    portal = ctx["portal"]
    return (
        f"I am not sure I understood that. You are currently in the **{portal}** portal.\n\n"
        "Here are some things I can help with:\n"
        "- **\"What happens if I increase SBERT weight?\"** — Algorithm impact analysis\n"
        "- **\"Show me my current settings\"** — Configuration snapshot\n"
        "- **\"How do I optimize my resume?\"** — ATS compliance tips\n"
        "- **\"Help\"** — Full command reference\n\n"
        "Try rephrasing your question or type **help** for available topics."
    )


# Intent → Response function mapping
_RESPONSE_DISPATCH = {
    "greeting": _respond_greeting,
    "help": _respond_help,
    "status": _respond_status,
    "weight_sbert": _respond_weight_sbert,
    "weight_tfidf": _respond_weight_tfidf,
    "weight_skills": _respond_weight_skills,
    "weight_experience": _respond_weight_experience,
    "threshold": _respond_threshold,
    "anonymization": _respond_anonymization,
    "ats_resume": _respond_ats_resume,
    "cover_letter": _respond_cover_letter,
    "interview": _respond_interview,
    "recruiter_jd": _respond_recruiter_jd,
    "how_it_works": _respond_how_it_works,
    "unknown": _respond_unknown,
}


def generate_copilot_response(user_message: str) -> str:
    """
    Main entry point: classify intent and generate a contextual response.
    Inspects current session state for dynamic context injection.
    """
    intent = _classify_intent(user_message)
    ctx = _get_context_snapshot()
    handler = _RESPONSE_DISPATCH.get(intent, _respond_unknown)
    return handler(ctx)


# ------------------------------------------------------------------
# Streamlit UI Renderer
# ------------------------------------------------------------------

def render_copilot_chatbot(app_mode: str, weights: dict, threshold: int, anonymize: bool):
    """
    Render the AI Co-Pilot chatbot panel in the sidebar.
    Syncs current configuration into session state for context-aware responses.

    Args:
        app_mode: Current active portal name.
        weights: Current scoring algorithm weight dictionary.
        threshold: Current minimum score threshold percentage.
        anonymize: Whether candidate anonymization is enabled.
    """
    # Sync live configuration into session state for the response engine
    st.session_state["_copilot_portal"] = app_mode
    st.session_state["_copilot_weights"] = weights
    st.session_state["_copilot_threshold"] = threshold
    st.session_state["_copilot_anonymize"] = anonymize

    # Initialize chat history
    if "_copilot_history" not in st.session_state:
        st.session_state["_copilot_history"] = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to the Platform Co-Pilot. I can help you understand scoring weights, "
                    "threshold trade-offs, anonymization impact, resume optimization, and more.\n\n"
                    "Type **help** for a full command reference, or ask me anything."
                )
            }
        ]

    st.sidebar.markdown("---")
    with st.sidebar.expander("Platform Co-Pilot Assistant", expanded=False):
        # Render conversation history
        for msg in st.session_state["_copilot_history"]:
            role = msg["role"]
            with st.chat_message(role, avatar="🤖" if role == "assistant" else "👤"):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.text_input(
            "Ask the Co-Pilot:",
            key="_copilot_input",
            placeholder="e.g. What if I increase SBERT weight?"
        )

        if st.button("Send", key="_copilot_send", use_container_width=True):
            if user_input and user_input.strip():
                # Append user message
                st.session_state["_copilot_history"].append({
                    "role": "user",
                    "content": user_input.strip()
                })

                # Generate and append response
                response = generate_copilot_response(user_input.strip())
                st.session_state["_copilot_history"].append({
                    "role": "assistant",
                    "content": response
                })

                st.rerun()

        # Clear conversation button
        if len(st.session_state["_copilot_history"]) > 1:
            if st.button("Clear Conversation", key="_copilot_clear"):
                st.session_state["_copilot_history"] = [
                    st.session_state["_copilot_history"][0]
                ]
                st.rerun()
