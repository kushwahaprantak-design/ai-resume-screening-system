"""
app.py
------
Main Streamlit entry point for the AI Resume Screening System.

Modes:
  - Candidate Workspace  → personal ATS audit, cover letter, interview prep
  - Recruiter Suite      → batch screening, leaderboard, PDF/CSV export
  - System Analytics     → skill taxonomy explorer, architecture reference

Run with:  streamlit run app.py
"""

import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.custom_css import apply_custom_css, inject_custom_css
from components.visualizations import (
    create_score_gauge,
    create_radar_chart,
    create_4d_candidate_radar_chart,
    create_candidate_comparison_chart,
    create_score_breakdown_bar,
    create_skill_gap_chart,
    create_interactive_taxonomy_chart,
)
from modules.extractor import extract_text_from_file, detect_ats_fraud
from modules.nlp_processor import load_skill_db, extract_skills
from modules.scoring_engine import analyze_candidate
from modules.ats_analyzer import analyze_ats_compliance
from modules.report_generator import generate_csv_report, generate_pdf_report
from modules.career_advisor import (
    generate_career_coaching_report,
    generate_executive_cover_letter,
    generate_copilot_unified_report,
)
from components.chatbot import render_copilot_chatbot

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_sample_jds() -> dict:
    """Load preset job descriptions from data/sample_jds.json."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        jd_path  = os.path.join(base_dir, "data", "sample_jds.json")
        if os.path.exists(jd_path):
            with open(jd_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load sample JDs: {e}")
    return {}


def load_sample_resumes() -> list:
    """Scan data/sample_resumes/ for .txt files and return [(filename, text)]."""
    samples = []
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        res_dir  = os.path.join(base_dir, "data", "sample_resumes")
        if os.path.exists(res_dir):
            for fname in os.listdir(res_dir):
                if fname.endswith(".txt"):
                    fpath = os.path.join(res_dir, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        samples.append((fname, f.read()))
    except Exception as e:
        st.warning(f"Could not load sample resumes: {e}")
    return samples


# ── Debug / log helper ────────────────────────────────────────────────────────

def _debug_log(msg: str):
    """Print to console AND append to session_state debug log (shown in UI if toggled)."""
    print(f"[DEBUG] {msg}")
    if "debug_logs" not in st.session_state:
        st.session_state["debug_logs"] = []
    st.session_state["debug_logs"].append(msg)


def _render_debug_panel():
    """
    Debug/log toggle — useful during viva to show the pipeline steps live.
    Toggle is in the sidebar; logs are shown as an expander in the main area.
    """
    if st.session_state.get("show_debug_logs"):
        logs = st.session_state.get("debug_logs", [])
        if logs:
            with st.expander("🛠️  Pipeline Debug Logs (viva demo mode)", expanded=True):
                for line in logs[-30:]:   # cap at last 30 entries
                    st.code(f"[LOG] {line}", language="bash")
        else:
            st.info("No pipeline logs yet — run an analysis first.")


# ── Header ────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
        <div class="enterprise-header">
            <h1 class="header-title">AI Resume Screening System</h1>
            <p class="header-subtitle">
                B.Tech Final Year Project &mdash; NLP-powered candidate evaluation
                using Sentence-BERT, TF-IDF Cosine Similarity, and Explainable AI.
            </p>
            <div style="margin-top: 0.75rem;">
                <span class="hero-tech-badge">Sentence-BERT</span>
                <span class="hero-tech-badge">TF-IDF + Cosine Sim</span>
                <span class="hero-tech-badge">Explainable AI (XAI)</span>
                <span class="hero-tech-badge">ATS Fraud Detector</span>
                <span class="hero-tech-badge">GitHub Verifier</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    """
    Navigation + scoring weight controls + debug toggle.
    Returns: (app_mode, weights_dict, min_threshold, anonymize_flag)
    """
    st.sidebar.markdown("### Navigation")
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["Candidate Workspace", "Recruiter Suite", "System Analytics & Reference"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Anti-Bias Settings")
    anonymize = st.sidebar.toggle(
        "Enable Candidate Anonymization",
        value=False,
        help="Masks name, email, phone, and social links for bias-free screening.",
    )

    st.sidebar.markdown("---")
    with st.sidebar.expander("Scoring Weights (Hybrid Engine)", expanded=False):
        st.caption("Adjust how much each signal contributes to the final score.")
        w_sbert  = st.slider("Sentence-BERT Weight",   0.0, 1.0, 0.35, 0.05)
        w_tfidf  = st.slider("TF-IDF Keyword Weight",  0.0, 1.0, 0.25, 0.05)
        w_skills = st.slider("Hard Skill Match Weight", 0.0, 1.0, 0.30, 0.05)
        w_exp    = st.slider("Experience Weight",       0.0, 1.0, 0.10, 0.05)

        total_w = w_sbert + w_tfidf + w_skills + w_exp
        if total_w > 0:
            weights = {
                "w_sbert":  w_sbert  / total_w,
                "w_tfidf":  w_tfidf  / total_w,
                "w_skills": w_skills / total_w,
                "w_exp":    w_exp    / total_w,
            }
        else:
            # shouldn't happen, but safe default
            weights = {"w_sbert": 0.35, "w_tfidf": 0.25, "w_skills": 0.30, "w_exp": 0.10}

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Qualification Threshold")
    min_threshold = st.sidebar.slider("Min Match Score (%)", 0, 100, 40, 5)

    st.sidebar.markdown("---")
    # debug toggle — handy during viva to walk the panel through the pipeline
    st.session_state["show_debug_logs"] = st.sidebar.toggle(
        "🛠️ Show Pipeline Debug Logs",
        value=st.session_state.get("show_debug_logs", False),
        help="Shows step-by-step pipeline logs below each analysis result.",
    )

    return app_mode, weights, min_threshold, anonymize


# ── Custom metric card renderer ───────────────────────────────────────────────

def _metric_card(label: str, value, color: str = "#818CF8", suffix: str = ""):
    """
    Render a styled metric card using custom CSS class.
    Cleaner than st.metric() for our glassmorphism theme.
    """
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color}">{value}{suffix}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _progress_metric(label: str, value: float, color: str = "#818CF8"):
    """Show label + progress bar side by side — better than bare st.progress()."""
    st.caption(label)
    st.progress(min(int(value), 100))


# ── Candidate Workspace ───────────────────────────────────────────────────────

def render_candidate_workspace(weights: dict):
    """
    Lets a student/candidate upload their resume, pick a target role,
    and get back: ATS score, skill gaps, cover letter, interview prep.
    """
    st.markdown("### Candidate Workspace")
    st.caption(
        "Upload your resume to get ATS compliance scores, skill gap analysis, "
        "a tailored cover letter, and interview prep questions."
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 1. Upload Your Resume")
        uploaded_resume = st.file_uploader(
            "Resume (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="cand_resume_upload",
        )
        cand_text = ""
        if st.checkbox("Use Sample Benchmark Resume", key="cand_use_sample"):
            samples = load_sample_resumes()
            if samples:
                cand_text = samples[0][1]
                st.info(f"Loaded: {samples[0][0]}")
        elif uploaded_resume:
            _debug_log(f"Extracting text from uploaded file: {uploaded_resume.name}")
            cand_text = extract_text_from_file(uploaded_resume, uploaded_resume.name)

    with c2:
        st.markdown("#### 2. Target Job Description (Optional)")
        sample_jds = load_sample_jds()
        jd_source  = st.segmented_control(
            "JD Source:", ["Preset Position", "Custom Role"], default="Preset Position"
        )
        target_jd       = ""
        target_role     = "Software Engineer"
        target_company  = ""

        if jd_source == "Preset Position":
            role_select = st.selectbox("Select Position:", list(sample_jds.keys()))
            if role_select:
                target_jd   = sample_jds[role_select].get("description", "")
                target_role = sample_jds[role_select].get("title", role_select)
        else:
            target_role    = st.text_input("Position Title", value="AI / ML Engineer")
            target_company = st.text_input("Company Name", placeholder="e.g. Google DeepMind")
            target_jd      = st.text_area("Job Requirements:", height=120, placeholder="Paste JD here...")

    st.markdown("---")

    if st.button("Generate My Resume Analysis", use_container_width=True, type="primary"):
        if not cand_text.strip():
            st.error("Please upload a resume or select the sample benchmark resume.")
        else:
            # default JD if none provided — first preset role
            if not target_jd.strip() and sample_jds:
                target_jd = list(sample_jds.values())[0].get("description", "")

            with st.spinner("Running NLP pipeline: SBERT embeddings → TF-IDF → skill matching..."):
                try:
                    _debug_log("Starting analyze_candidate()")
                    analysis = analyze_candidate(
                        resume_text=cand_text,
                        jd_text=target_jd,
                        weights=weights,
                        anonymize=False,
                    )
                    _debug_log(f"Overall score: {analysis['overall_score']}%")

                    _debug_log("Running ATS compliance audit")
                    ats_audit = analyze_ats_compliance(cand_text, analysis)

                    _debug_log("Generating AI Copilot report")
                    copilot_report = generate_copilot_unified_report(
                        cand_text,
                        target_jd_text=target_jd,
                        target_role=target_role,
                        company_name=target_company,
                    )

                    fraud_info = (
                        detect_ats_fraud(uploaded_resume, uploaded_resume.name)
                        if uploaded_resume else {"fraud_detected": False, "reasons": []}
                    )
                    _debug_log(f"ATS fraud check: {fraud_info['fraud_detected']}")

                    # store in session so results survive reruns
                    st.session_state.update({
                        "cand_analysis":       analysis,
                        "cand_ats_audit":      ats_audit,
                        "cand_copilot_report": copilot_report,
                        "cand_fraud_info":     fraud_info,
                        "cand_resume_text":    cand_text,
                    })
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    _debug_log(f"ERROR in analyze_candidate: {e}")

    _render_debug_panel()

    # ── Show results if we have them ──────────────────────────────────────────
    if "cand_analysis" not in st.session_state:
        return

    analysis       = st.session_state["cand_analysis"]
    ats_audit      = st.session_state["cand_ats_audit"]
    copilot_report = st.session_state["cand_copilot_report"]
    fraud_info     = st.session_state["cand_fraud_info"]
    cand_text      = st.session_state["cand_resume_text"]

    st.markdown("### Analysis Results")

    tab1, tab2, tab3 = st.tabs([
        "ATS Audit & Resume Optimization",
        "Job Matching & Company Strategy",
        "Career Advisor & Interview Prep",
    ])

    # ── Tab 1: ATS Audit ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### ATS Compliance & Resume Health")

        if fraud_info.get("fraud_detected"):
            st.error(
                f"⚠️  Anti-Fraud Flag: Hidden text or micro-fonts detected "
                f"({', '.join(fraud_info.get('reasons', []))})"
            )
        else:
            st.success("✅  Anti-Fraud Check Passed — no hidden text or font manipulation found.")

        # ── Score KPI cards ───────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1: _metric_card("ATS Compatibility",  f"{ats_audit['ats_score']}%",       "#818CF8")
        with k2: _metric_card("Hybrid Match Score", f"{analysis['overall_score']}%",    "#34D399")
        with k3: _metric_card("Word Count",          ats_audit["word_count"],            "#38BDF8")
        with k4: _metric_card("Skill Overlap",       f"{analysis['skill_score']}%",      "#FBBF24")

        st.markdown("<br/>", unsafe_allow_html=True)

        # ── Score progress bars (nice viva visual) ────────────────────────────
        pb1, pb2 = st.columns(2)
        with pb1:
            _progress_metric(f"SBERT Semantic Score: {analysis['sbert_score']}%",  analysis["sbert_score"],  "#818CF8")
            _progress_metric(f"TF-IDF Keyword Score: {analysis['tfidf_score']}%",  analysis["tfidf_score"],  "#38BDF8")
        with pb2:
            _progress_metric(f"Hard Skill Match: {analysis['skill_score']}%",      analysis["skill_score"],  "#34D399")
            _progress_metric(f"Experience Match: {analysis['exp_score']}%",        analysis["exp_score"],    "#FBBF24")

        st.markdown("---")

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown("##### Matched Skills")
            if analysis["matched_skills"]:
                pills = "".join([f'<span class="skill-pill">✓ {s}</span>' for s in analysis["matched_skills"]])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.write("No direct skill matches found.")
            st.markdown("</div>", unsafe_allow_html=True)

        with s2:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown("##### Missing Skills")
            if analysis["missing_skills"]:
                pills = "".join([f'<span class="skill-pill skill-pill-missing">✗ {s}</span>' for s in analysis["missing_skills"]])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.write("All target skills are present — great!")
            st.markdown("</div>", unsafe_allow_html=True)

        with s3:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown("##### Bullet Point Suggestions")
            for fix in copilot_report.get("bullet_fixes", []):
                st.markdown(f"- *Current:* \"{fix['current']}\"")
                st.markdown(f"  - *Improved:* <font color='#34D399'>**\"{fix['improved']}\"**</font>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        g1, g2 = st.columns(2)
        with g1:
            fig_gauge = create_score_gauge(ats_audit["ats_score"], "ATS Readability Score")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g2:
            categories  = ["Languages", "Frameworks", "Data/AI", "Databases", "Cloud/DevOps", "CS Core"]
            skill_db    = load_skill_db()
            by_cat      = extract_skills(cand_text, skill_db)["by_category"]
            cat_keys    = ["programming_languages", "web_frameworks", "data_science_ml", "databases", "cloud_devops", "core_cs"]
            cat_scores  = []
            for key in cat_keys:
                found = len(by_cat.get(key, []))
                total = len(skill_db.get(key, [1]))
                # scale to 100 but cap it — don't over-reward a small category
                cat_scores.append(min(round((found / max(total, 1)) * 300, 1), 100.0))

            fig_radar = create_radar_chart(categories, cat_scores, "Skill Category Proficiency")
            st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tab 2: Job Matching ───────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Target Roles & Company Alignment")

        j1, j2 = st.columns(2)
        with j1:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown("##### Recommended Job Titles")
            for role in copilot_report.get("best_roles", []):
                st.markdown(f"- **{role}**")
            st.markdown("</div>", unsafe_allow_html=True)

        with j2:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown("##### Priority Skill Gaps to Fill")
            for gap in copilot_report.get("skill_gaps", []):
                st.markdown(f"- <font color='#F87171'>**{gap}**</font>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Company Targets")
        comp1, comp2 = st.columns(2)
        with comp1:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown("##### Tier-1 Tech MNCs")
            for c in copilot_report.get("mncs", []):
                st.markdown(f"- {c}")
            st.markdown("</div>", unsafe_allow_html=True)

        with comp2:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown("##### High-Growth AI & SaaS Startups")
            for c in copilot_report.get("startups", []):
                st.markdown(f"- {c}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 3: Career Advisor ─────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Cover Letter & Interview Prep")

        st.markdown("##### Tailored Cover Letter Draft")
        st.text_area("Copy and edit as needed:", value=copilot_report.get("cover_letter", ""), height=250)

        st.markdown("---")
        st.markdown("##### Practice Interview Questions")
        for i, q in enumerate(copilot_report.get("mock_questions", []), 1):
            st.markdown(f"**Q{i}:** {q}")


# ── Recruiter Suite ───────────────────────────────────────────────────────────

def render_recruiter_suite(weights: dict, min_threshold: int, anonymize: bool):
    """
    Batch resume screening mode for HR.
    Processes multiple resumes, ranks candidates, and exports reports.
    """
    st.markdown("### Recruiter Suite")
    st.caption(
        "Batch-process candidate resumes against a job description. "
        "Results are ranked by hybrid NLP score with XAI breakdowns."
    )

    col1, col2 = st.columns([1, 1])
    sample_jds = load_sample_jds()

    with col1:
        st.markdown("#### 1. Job Description")
        jd_source  = st.segmented_control(
            "JD Input:", ["Preset Roles", "Custom Text", "Upload File"], default="Preset Roles"
        )
        jd_text    = ""
        jd_title   = "Target Role"
        min_exp    = 0.0

        if jd_source == "Preset Roles":
            selected = st.selectbox("Select Position:", list(sample_jds.keys()))
            if selected:
                info     = sample_jds[selected]
                jd_title = info.get("title", selected)
                jd_text  = info.get("description", "")
                min_exp  = float(info.get("min_experience_years", 0))
                st.info(f"Min experience: {min_exp} yrs | Required: {', '.join(info.get('required_skills', []))}")

        elif jd_source == "Custom Text":
            jd_title = st.text_input("Position Title", value="Software Development Engineer")
            jd_text  = st.text_area("Job Description:", height=180, placeholder="Paste full JD here...")
            min_exp  = st.number_input("Minimum Experience (Years)", 0.0, 15.0, 2.0, 0.5)

        else:
            uploaded_jd = st.file_uploader("Upload JD File", type=["pdf", "docx", "txt"])
            if uploaded_jd:
                jd_text  = extract_text_from_file(uploaded_jd, uploaded_jd.name)
                jd_title = os.path.splitext(uploaded_jd.name)[0]
                st.success(f"Loaded: {uploaded_jd.name}")

    with col2:
        st.markdown("#### 2. Candidate Resumes")
        res_source = st.segmented_control(
            "Resume Source:", ["Upload Resumes", "Sample Dataset"], default="Sample Dataset"
        )
        resume_data = []   # list of (filename, text, file_obj_or_None)

        if res_source == "Upload Resumes":
            uploaded_resumes = st.file_uploader(
                "Upload Resumes (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
            )
            if uploaded_resumes:
                for f in uploaded_resumes:
                    text = extract_text_from_file(f, f.name)
                    if text:
                        resume_data.append((f.name, text, f))
                    else:
                        st.warning(f"Could not extract text from {f.name}")
        else:
            samples = load_sample_resumes()
            st.info(f"Loaded {len(samples)} sample candidate resumes.")
            for fname, rtxt in samples:
                resume_data.append((fname, rtxt, None))

    st.markdown("---")

    if st.button("Run Candidate Screening", use_container_width=True, type="primary"):
        if not jd_text.strip():
            st.error("Please provide a job description first.")
        elif not resume_data:
            st.error("Please upload resumes or select the sample dataset.")
        else:
            with st.spinner("Scoring candidates — SBERT + TF-IDF + GitHub audit in progress..."):
                results = []
                prog_bar = st.progress(0)   # show batch progress during demo

                for idx, (fname, rtext, fobj) in enumerate(resume_data, 1):
                    try:
                        _debug_log(f"Processing resume {idx}/{len(resume_data)}: {fname}")
                        analysis = analyze_candidate(
                            resume_text=rtext,
                            jd_text=jd_text,
                            min_experience_years=min_exp,
                            weights=weights,
                            anonymize=anonymize,
                        )
                        display_name = f"Candidate #{100 + idx}" if anonymize else fname
                        analysis["filename"]          = display_name
                        analysis["original_filename"] = fname
                        analysis["raw_text"]          = rtext
                        analysis["fraud_info"]        = (
                            detect_ats_fraud(fobj, fname) if fobj else
                            {"fraud_detected": False, "reasons": []}
                        )
                        results.append(analysis)
                    except Exception as e:
                        st.error(f"Error processing {fname}: {e}")
                        _debug_log(f"ERROR on {fname}: {e}")

                    prog_bar.progress(int((idx / len(resume_data)) * 100))

                results.sort(key=lambda x: x["overall_score"], reverse=True)
                st.session_state["screening_results"]  = results
                st.session_state["target_jd_title"]    = jd_title
                _debug_log(f"Screening done. {len(results)} candidates ranked.")

    _render_debug_panel()

    # ── Leaderboard ───────────────────────────────────────────────────────────
    if not st.session_state.get("screening_results"):
        return

    results   = st.session_state["screening_results"]
    jd_title  = st.session_state.get("target_jd_title", "Position")

    st.markdown("### Candidate Leaderboard")

    # ── KPI summary row ───────────────────────────────────────────────────────
    try:
        total      = len(results)
        qualified  = len([c for c in results if c["overall_score"] >= min_threshold])
        top_score  = results[0]["overall_score"] if results else 0
        avg_score  = round(sum(c["overall_score"] for c in results) / max(total, 1), 1)

        k1, k2, k3, k4 = st.columns(4)
        with k1: _metric_card("Total Submissions",              total,            "#818CF8")
        with k2: _metric_card(f"Qualified (≥ {min_threshold}%)", qualified,       "#34D399")
        with k3: _metric_card("Top Score",                     f"{top_score}%",   "#3B82F6")
        with k4: _metric_card("Batch Average",                 f"{avg_score}%",   "#FBBF24")
    except Exception as e:
        st.warning(f"Error rendering KPI cards: {e}")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Leaderboard table + 4D radar ─────────────────────────────────────────
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("#### Ranked Candidate Summary")
        try:
            rows = []
            for i, c in enumerate(results, 1):
                fraud_label  = "⚠️ Flagged" if c.get("fraud_info", {}).get("fraud_detected") else "✅ Passed"
                github_idx   = c.get("github_audit", {}).get("credibility_index", "Unverified")
                rows.append({
                    "Rank":             i,
                    "Candidate":        c["filename"],
                    "Score (%)":        f"{c['overall_score']}%",
                    "Tier":             c["tier"],
                    "GitHub Index":     github_idx,
                    "Fraud Check":      fraud_label,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                csv_data = generate_csv_report(results)
                st.download_button(
                    "⬇️ Download CSV Audit Report",
                    data=csv_data,
                    file_name=f"screening_{jd_title.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                if results:
                    top_pdf = generate_pdf_report(results[0], jd_title)
                    st.download_button(
                        "⬇️ Export Top Candidate PDF",
                        data=top_pdf,
                        file_name=f"Report_{results[0]['filename']}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Leaderboard render error: {e}")

    with right:
        st.markdown("#### 4-Dimension Evaluation Radar")
        try:
            fig_4d = create_4d_candidate_radar_chart(results)
            st.plotly_chart(fig_4d, use_container_width=True)
        except Exception as e:
            st.warning(f"Radar chart error: {e}")

    # ── Per-candidate XAI profiles ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Candidate Profiles & XAI Breakdown")

    for c in results:
        if c["overall_score"] < min_threshold:
            continue  # skip below-threshold candidates

        rank_label = f"Rank #{results.index(c) + 1}"
        with st.expander(f"{rank_label}: {c['filename']} — {c['overall_score']}% ({c['tier']})"):

            if c.get("fraud_info", {}).get("fraud_detected"):
                st.error(f"ATS Fraud: {', '.join(c['fraud_info'].get('reasons', []))}")

            # XAI card
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown("#### Explainable AI (XAI) Score Breakdown")
            xai = c.get("xai_explanation", {})
            st.info(f"**Verdict:** {xai.get('justification_summary', 'N/A')}")

            xc1, xc2 = st.columns(2)
            with xc1:
                st.markdown("**Positive Contributors:**")
                for pos in xai.get("positive_contributors", []):
                    st.markdown(f"- <font color='#34D399'>{pos}</font>", unsafe_allow_html=True)
            with xc2:
                st.markdown("**Score Deductions:**")
                for ded in xai.get("deductions", []):
                    st.markdown(f"- <font color='#F87171'>{ded}</font>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Charts + contact details
            d1, d2 = st.columns([1, 1])
            with d1:
                try:
                    fig_g = create_score_gauge(c["overall_score"], f"{c['filename']}")
                    st.plotly_chart(fig_g, use_container_width=True)
                except Exception as e:
                    st.warning(f"Gauge error: {e}")

                st.markdown(f"**Email:** `{c['contact_info']['email']}` | **Phone:** `{c['contact_info']['phone']}`")
                st.markdown(f"**Education:** {c['education']} | **Experience:** {c['experience_years']} yrs")
                gh = c.get("github_audit", {})
                st.markdown(f"**GitHub Credibility:** `{gh.get('credibility_index')}` ({gh.get('public_repos', 0)} public repos)")

            with d2:
                try:
                    fig_bar = create_score_breakdown_bar(c)
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.warning(f"Breakdown chart error: {e}")

            # Interview questions for this candidate's gaps
            with st.expander("Technical Interview Questions (Recruiter Guide)", expanded=False):
                questions = c.get("interview_questions", [])
                if questions:
                    for qi, qt in enumerate(questions, 1):
                        st.markdown(f"**Q{qi}:** {qt}")
                else:
                    st.write("No specific gap questions generated.")

            # Skill gap chart
            gap1, gap2 = st.columns([1, 1])
            with gap1:
                try:
                    fig_gap = create_skill_gap_chart(c["matched_skills"], c["missing_skills"])
                    st.plotly_chart(fig_gap, use_container_width=True)
                except Exception as e:
                    st.warning(f"Skill gap chart error: {e}")

            with gap2:
                st.markdown("**Matched Skills:**")
                if c["matched_skills"]:
                    pills = "".join([f'<span class="skill-pill">✓ {s}</span>' for s in c["matched_skills"]])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.write("None identified.")

                st.markdown("<br/>**Missing Skills:**", unsafe_allow_html=True)
                if c["missing_skills"]:
                    pills = "".join([f'<span class="skill-pill skill-pill-missing">✗ {s}</span>' for s in c["missing_skills"]])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.write("None — all target skills present!")


# ── System Analytics & Reference ──────────────────────────────────────────────

def render_system_analytics():
    """Skill taxonomy explorer + architecture/algorithm reference panel."""
    st.markdown("### System Analytics & Reference")

    tab_a, tab_b = st.tabs(["Skill Taxonomy Explorer", "Architecture & Algorithm Reference"])

    with tab_a:
        st.caption("Browse the 500+ skill taxonomy powering the NLP matcher.")
        try:
            skill_db = load_skill_db()
            total    = sum(len(v) for v in skill_db.values())
            st.markdown(
                f'<div style="margin-bottom:1rem;"><span class="badge badge-recommended" '
                f'style="font-size:0.95rem;">Taxonomy: {total} unique skills across {len(skill_db)} categories</span></div>',
                unsafe_allow_html=True,
            )

            selected_cat = st.segmented_control(
                "Category:", list(skill_db.keys()), default=list(skill_db.keys())[0]
            )
            if selected_cat:
                skills = skill_db[selected_cat]
                st.markdown(f"#### `{selected_cat.upper().replace('_', ' ')}` — {len(skills)} skills")
                pills = "".join([f'<span class="skill-pill" style="font-size:0.9rem;">{s}</span>' for s in skills])
                st.markdown(pills, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Distribution Across All Categories")
            fig_tax = create_interactive_taxonomy_chart(skill_db, selected_cat)
            st.plotly_chart(fig_tax, use_container_width=True)
        except Exception as e:
            st.error(f"Taxonomy render error: {e}")

    with tab_b:
        st.caption("Technical specification: NLP pipeline, models, and scoring equations.")
        st.markdown(r"""
        #### Data Flow Architecture
        ```
        ┌──────────────────────┐     ┌──────────────────────┐
        │  Resume (PDF/DOCX)   │     │  Job Description     │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   ▼                             ▼
        ┌──────────────────────┐     ┌──────────────────────┐
        │  extractor.py        │     │  ATS Fraud Inspector  │
        │  (pdfplumber / docx) │     │  (white text / fonts) │
        └──────────┬───────────┘     └──────────┬───────────┘
                   │                             │
                   ▼                             ▼
        ┌──────────────────────────────────────────────────────┐
        │               nlp_processor.py                       │
        │  normalize_text → tokenize → stopword filter → lemma │
        └─────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────────┐
        │              scoring_engine.py                       │
        │  1. compute_sbert_score  (all-MiniLM-L6-v2)         │
        │  2. compute_tfidf_score  (unigram + bigram)          │
        │  3. Skill overlap matrix (regex × skill_db.json)     │
        │  4. Experience score  (regex date parsing)           │
        │  5. GitHub credibility (github_verifier.py)          │
        │  6. Weighted hybrid final score                      │
        │  7. XAI card + interview question generation         │
        └─────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────────┐
        │        Streamlit Dashboard (app.py)                  │
        │  Leaderboard ▸ Radar/Gauge charts ▸ PDF/CSV export  │
        └──────────────────────────────────────────────────────┘
        ```

        ---

        #### Mathematical Formulations

        ##### 1. Sentence-BERT Cosine Similarity
        $$\text{Sim}_{SBERT}(R, JD) = \frac{\mathbf{e}_R \cdot \mathbf{e}_{JD}}{\|\mathbf{e}_R\| \|\mathbf{e}_{JD}\|}$$

        ##### 2. TF-IDF Weighting
        $$\text{TF-IDF}(t, d) = \log(1 + tf_{t,d}) \times \log\!\left(\frac{|D|}{1 + df_t}\right)$$

        ##### 3. Hybrid Weighted Final Score
        $$\text{Score} = w_{sbert} \cdot S_{SBERT} + w_{tfidf} \cdot S_{TF\text{-}IDF} + w_{skills} \cdot S_{Skills} + w_{exp} \cdot S_{Exp}$$

        where $\sum w_i = 1.0$ (normalized from sidebar sliders at runtime).
        """)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_header()

    app_mode, weights, min_threshold, anonymize = render_sidebar()

    # context-aware AI copilot chatbot in sidebar
    render_copilot_chatbot(app_mode, weights, min_threshold, anonymize)

    if app_mode == "Candidate Workspace":
        render_candidate_workspace(weights)
    elif app_mode == "Recruiter Suite":
        render_recruiter_suite(weights, min_threshold, anonymize)
    else:
        render_system_analytics()


if __name__ == "__main__":
    main()
