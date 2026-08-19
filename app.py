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
    create_interactive_taxonomy_chart
)
from modules.extractor import extract_text_from_file, detect_ats_fraud
from modules.nlp_processor import load_skill_db, extract_skills
from modules.scoring_engine import analyze_candidate
from modules.ats_analyzer import analyze_ats_compliance
from modules.report_generator import generate_csv_report, generate_pdf_report
from modules.career_advisor import (
    generate_career_coaching_report,
    generate_executive_cover_letter,
    generate_copilot_unified_report
)
from components.chatbot import render_copilot_chatbot

# Page Configuration
st.set_page_config(
    page_title="Resume Screening & Candidate Analytics Engine",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Glassmorphism Design System CSS
inject_custom_css()


def load_application_css():
    """Inject enterprise styling rules."""
    inject_custom_css()



@st.cache_data
def load_sample_job_descriptions() -> dict:
    """Load JSON database of sample job descriptions."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        jd_path = os.path.join(base_dir, "data", "sample_jds.json")
        if os.path.exists(jd_path):
            with open(jd_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load sample JDs: {e}")
    return {}


def load_sample_resumes() -> list:
    """Load pre-configured sample candidate resumes."""
    samples = []
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        res_dir = os.path.join(base_dir, "data", "sample_resumes")
        if os.path.exists(res_dir):
            for fname in os.listdir(res_dir):
                if fname.endswith(".txt"):
                    fpath = os.path.join(res_dir, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    samples.append((fname, content))
    except Exception as e:
        st.warning(f"Could not load sample resumes: {e}")
    return samples


def render_header():
    """Render clean, professional SaaS-style header bar."""
    st.markdown("""
        <div class="enterprise-header">
            <h1 class="header-title">Resume Screening &amp; Candidate Analytics Engine</h1>
            <p class="header-subtitle">
                Enterprise talent evaluation platform powered by Sentence-BERT, TF-IDF, and Explainable AI.
            </p>
            <div style="margin-top: 0.75rem;">
                <span class="hero-tech-badge">Sentence-BERT</span>
                <span class="hero-tech-badge">TF-IDF Vectorizer</span>
                <span class="hero-tech-badge">Explainable AI</span>
                <span class="hero-tech-badge">ATS Inspector</span>
            </div>
        </div>
    """, unsafe_allow_html=True)



def render_sidebar():
    """Render clean enterprise sidebar navigation and evaluation configuration controls."""
    st.sidebar.markdown("### Navigation Workspace")
    app_mode = st.sidebar.radio(
        "Select Portal Mode",
        [
            "Candidate Workspace",
            "Recruiter Suite",
            "System Analytics & Reference"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Privacy & Anti-Bias")
    anonymize_toggle = st.sidebar.toggle(
        "Enable Candidate Anonymization",
        value=False,
        help="Mask candidate name, email, phone, and profile links for unbiased evaluation."
    )

    st.sidebar.markdown("---")
    with st.sidebar.expander("Scoring Algorithm Weights", expanded=False):
        st.caption("Customize weights for hybrid scoring engine:")
        w_sbert = st.slider("Sentence-BERT Weight", 0.0, 1.0, 0.35, 0.05)
        w_tfidf = st.slider("TF-IDF Keyword Weight", 0.0, 1.0, 0.25, 0.05)
        w_skills = st.slider("Hard Skill Match Weight", 0.0, 1.0, 0.30, 0.05)
        w_exp = st.slider("Experience Match Weight", 0.0, 1.0, 0.10, 0.05)

        total_w = w_sbert + w_tfidf + w_skills + w_exp
        if total_w > 0:
            weights = {
                "w_sbert": w_sbert / total_w,
                "w_tfidf": w_tfidf / total_w,
                "w_skills": w_skills / total_w,
                "w_exp": w_exp / total_w
            }
        else:
            weights = {"w_sbert": 0.35, "w_tfidf": 0.25, "w_skills": 0.30, "w_exp": 0.10}

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Qualification Threshold")
    min_match_threshold = st.sidebar.slider("Minimum Score Threshold (%)", 0, 100, 40, 5)

    return app_mode, weights, min_match_threshold, anonymize_toggle


def render_candidate_hub(weights: dict):
    """Render unified Candidate Workspace using Streamlit Tabs without emoji clutter."""
    st.markdown("### Candidate Workspace")
    st.caption("Process your resume to generate ATS compliance scores, job match suggestions, tailored cover letters, and interview prep.")

    c_col1, c_col2 = st.columns([1, 1])

    with c_col1:
        st.markdown("#### 1. Upload Resume Document")
        uploaded_resume_file = st.file_uploader(
            "Upload Resume (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="candidate_hub_resume_upload"
        )
        cand_resume_text = ""
        if st.checkbox("Use Benchmark Sample Resume", key="cand_hub_bench"):
            samples = load_sample_resumes()
            if samples:
                cand_resume_text = samples[0][1]
                st.info(f"Loaded benchmark resume: {samples[0][0]}")
        elif uploaded_resume_file:
            cand_resume_text = extract_text_from_file(uploaded_resume_file, uploaded_resume_file.name)

    with c_col2:
        st.markdown("#### 2. Target Job & Company (Optional)")
        sample_jds = load_sample_job_descriptions()

        ats_jd_mode = st.segmented_control(
            "Target JD Source:",
            ["Preset Position", "Custom Role"],
            default="Preset Position"
        )

        target_jd_text = ""
        target_role_title = "Software Engineer"
        target_company_name = ""

        if ats_jd_mode == "Preset Position":
            target_role_select = st.selectbox("Select Target Position:", list(sample_jds.keys()))
            if target_role_select:
                target_jd_text = sample_jds[target_role_select].get("description", "")
                target_role_title = sample_jds[target_role_select].get("title", target_role_select)
        else:
            target_role_title = st.text_input("Target Position Title", value="AI / Machine Learning Engineer")
            target_company_name = st.text_input("Target Company Name", placeholder="e.g. Google DeepMind / Microsoft AI")
            target_jd_text = st.text_area("Target Job Requirements:", height=120, placeholder="Paste job qualifications...")

    st.markdown("---")

    if st.button("Generate Comprehensive Candidate Analysis", use_container_width=True):
        if not cand_resume_text.strip():
            st.error("Please upload a resume file or select a benchmark sample resume.")
        else:
            if not target_jd_text.strip() and sample_jds:
                target_jd_text = list(sample_jds.values())[0].get("description", "")

            with st.spinner("Analyzing document compliance, running Sentence-BERT embeddings, and generating career insights..."):
                try:
                    analysis = analyze_candidate(
                        resume_text=cand_resume_text,
                        jd_text=target_jd_text,
                        weights=weights,
                        anonymize=False
                    )
                    ats_audit = analyze_ats_compliance(cand_resume_text, analysis)
                    copilot_report = generate_copilot_unified_report(
                        cand_resume_text,
                        target_jd_text=target_jd_text,
                        target_role=target_role_title,
                        company_name=target_company_name
                    )

                    if uploaded_resume_file is not None:
                        fraud_info = detect_ats_fraud(uploaded_resume_file, uploaded_resume_file.name)
                    else:
                        fraud_info = {"fraud_detected": False, "reasons": []}

                    st.session_state["cand_analysis"] = analysis
                    st.session_state["cand_ats_audit"] = ats_audit
                    st.session_state["cand_copilot_report"] = copilot_report
                    st.session_state["cand_fraud_info"] = fraud_info
                    st.session_state["cand_resume_text"] = cand_resume_text
                except Exception as e:
                    st.error(f"Error analyzing candidate profile: {e}")

    # Display Clean Output using st.tabs
    if "cand_analysis" in st.session_state:
        analysis = st.session_state["cand_analysis"]
        ats_audit = st.session_state["cand_ats_audit"]
        copilot_report = st.session_state["cand_copilot_report"]
        fraud_info = st.session_state["cand_fraud_info"]
        cand_resume_text = st.session_state["cand_resume_text"]

        st.markdown("### Candidate Comprehensive Insights")

        # Render 3 Clean Tabs without Emojis
        tab1, tab2, tab3 = st.tabs([
            "ATS Audit & Resume Optimization",
            "Job Matching & Company Strategy",
            "Career Advisor & Interview Prep"
        ])

        # TAB 1: ATS Audit & Resume Optimization
        with tab1:
            st.markdown("#### ATS Readability & Document Compliance Audit")

            if fraud_info.get("fraud_detected"):
                st.error(f"Anti-Fraud Compliance Warning: FLAGGED (Hidden text or font manipulation detected: {', '.join(fraud_info.get('reasons', []))}).")
            else:
                st.success("Anti-Fraud Compliance Check: PASSED (No hidden text or font manipulation detected).")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">ATS Compatibility</div><div class="metric-value" style="color:#818CF8">{ats_audit["ats_score"]}%</div></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Hybrid Match Score</div><div class="metric-value" style="color:#34D399">{analysis["overall_score"]}%</div></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Word Count</div><div class="metric-value" style="color:#38BDF8">{ats_audit["word_count"]}</div></div>', unsafe_allow_html=True)
            with r4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Skill Overlap</div><div class="metric-value" style="color:#FBBF24">{analysis["skill_score"]}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.markdown('<div class="card-box">', unsafe_allow_html=True)
                st.markdown("##### Matched Skills")
                if analysis["matched_skills"]:
                    pills = "".join([f'<span class="skill-pill">✓ {s}</span>' for s in analysis["matched_skills"]])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.write("No direct skills matched.")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_a2:
                st.markdown('<div class="card-box">', unsafe_allow_html=True)
                st.markdown("##### Missing Critical Skills")
                if analysis["missing_skills"]:
                    pills_m = "".join([f'<span class="skill-pill skill-pill-missing">✗ {s}</span>' for s in analysis["missing_skills"]])
                    st.markdown(pills_m, unsafe_allow_html=True)
                else:
                    st.write("All target skills present.")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_a3:
                st.markdown('<div class="card-box">', unsafe_allow_html=True)
                st.markdown("##### Metric-Driven Bullet Improvements")
                for fix in copilot_report["bullet_fixes"]:
                    st.markdown(f"- *Current:* \"{fix['current']}\"")
                    st.markdown(f"  - *Improved:* <font color='#34D399'>**\"{fix['improved']}\"**</font>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            c_aud1, c_aud2 = st.columns([1, 1])
            with c_aud1:
                fig_ats_g = create_score_gauge(ats_audit["ats_score"], "ATS Readability Score")
                st.plotly_chart(fig_ats_g, use_container_width=True)

            with c_aud2:
                categories = ["Languages", "Frameworks", "Data/AI", "Databases", "Cloud/DevOps", "CS Core"]
                skill_db = load_skill_db()
                skills_grouped = extract_skills(cand_resume_text, skill_db)["by_category"]

                cat_scores = []
                for cat_key in ["programming_languages", "web_frameworks", "data_science_ml", "databases", "cloud_devops", "core_cs"]:
                    cat_found = len(skills_grouped.get(cat_key, []))
                    cat_total = len(skill_db.get(cat_key, [1]))
                    score = min(round((cat_found / max(cat_total, 1)) * 300, 1), 100.0)
                    cat_scores.append(score)

                fig_radar = create_radar_chart(categories, cat_scores, "Skill Category Proficiency Radar")
                st.plotly_chart(fig_radar, use_container_width=True)

        # TAB 2: Job Matching & Company Strategy
        with tab2:
            st.markdown("#### Targeted Roles & Company Alignment")

            j_col1, j_col2 = st.columns(2)
            with j_col1:
                st.markdown('<div class="card-box">', unsafe_allow_html=True)
                st.markdown("##### Recommended Job Titles")
                for role in copilot_report["best_roles"]:
                    st.markdown(f"- **{role}**")
                st.markdown('</div>', unsafe_allow_html=True)

            with j_col2:
                st.markdown('<div class="card-box">', unsafe_allow_html=True)
                st.markdown("##### Key Skill Gaps to Learn Next")
                for sg in copilot_report["skill_gaps"]:
                    st.markdown(f"- <font color='#F87171'>**{sg}**</font>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### Recommended Company Targets")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown('<div class="card-box">', unsafe_allow_html=True)
                st.markdown("##### Tier-1 Tech MNCs")
                for mnc in copilot_report["mncs"]:
                    st.markdown(f"- {mnc}")
                st.markdown('</div>', unsafe_allow_html=True)

            with comp_col2:
                st.markdown('<div class="card-box">', unsafe_allow_html=True)
                st.markdown("##### High-Growth AI & SaaS Startups")
                for st_comp in copilot_report["startups"]:
                    st.markdown(f"- {st_comp}")
                st.markdown('</div>', unsafe_allow_html=True)

        # TAB 3: Career Advisor & Interview Prep
        with tab3:
            st.markdown("#### Tailored Cover Letter & Interview Preparation")

            st.markdown("##### Customized Executive Cover Letter")
            st.text_area("Copy-Paste Cover Letter Draft:", value=copilot_report["cover_letter"], height=250)

            st.markdown("---")
            st.markdown("##### Recommended Technical & Behavioral Practice Questions")
            for q_idx, q_txt in enumerate(copilot_report["mock_questions"], 1):
                st.markdown(f"**Q{q_idx}:** {q_txt}")


def render_recruiter_dashboard(weights: dict, min_threshold: int, anonymize: bool):
    """Render HR Recruiter Suite workspace."""
    st.markdown("### Recruiter Suite")
    st.caption("Process and rank candidate resumes against target job descriptions using hybrid NLP scoring and XAI breakdowns.")

    col1, col2 = st.columns([1, 1])
    sample_jds = load_sample_job_descriptions()

    with col1:
        st.markdown("#### 1. Job Description Specification")
        jd_input_option = st.segmented_control(
            "JD Input Source:",
            ["Preset Roles", "Custom Text", "Upload File"],
            default="Preset Roles"
        )

        jd_text = ""
        jd_title = "Target Role"
        min_exp_req = 0.0

        if jd_input_option == "Preset Roles":
            selected_role = st.selectbox("Select Target Position:", list(sample_jds.keys()))
            if selected_role:
                jd_info = sample_jds[selected_role]
                jd_title = jd_info.get("title", selected_role)
                jd_text = jd_info.get("description", "")
                min_exp_req = float(jd_info.get("min_experience_years", 0))
                st.info(f"Required Experience: {min_exp_req} Years | Required Skills: {', '.join(jd_info.get('required_skills', []))}")

        elif jd_input_option == "Custom Text":
            jd_title = st.text_input("Position Title", value="Software Development Engineer")
            jd_text = st.text_area("Job Description Requirements:", height=180, placeholder="Paste job qualifications, technical requirements, responsibilities...")
            min_exp_req = st.number_input("Minimum Required Experience (Years)", 0.0, 15.0, 2.0, 0.5)

        else:
            uploaded_jd = st.file_uploader("Upload Job Description File", type=["pdf", "docx", "txt"])
            if uploaded_jd:
                jd_text = extract_text_from_file(uploaded_jd, uploaded_jd.name)
                jd_title = os.path.splitext(uploaded_jd.name)[0]
                st.success(f"Loaded Job Description: {uploaded_jd.name}")

    with col2:
        st.markdown("#### 2. Candidate Submissions")
        resume_source = st.segmented_control(
            "Resume Source:",
            ["Upload Resumes", "Sample Resumes"],
            default="Sample Resumes"
        )

        resume_files_data = []

        if resume_source == "Upload Resumes":
            uploaded_resumes = st.file_uploader(
                "Upload Candidate Resumes (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True
            )
            if uploaded_resumes:
                for f in uploaded_resumes:
                    extracted_txt = extract_text_from_file(f, f.name)
                    if extracted_txt:
                        resume_files_data.append((f.name, extracted_txt, f))
                    else:
                        st.warning(f"Could not extract text from {f.name}")
        else:
            sample_resumes = load_sample_resumes()
            st.info(f"Loaded {len(sample_resumes)} candidate benchmark resumes.")
            for fname, rtxt in sample_resumes:
                resume_files_data.append((fname, rtxt, None))

    st.markdown("---")

    if st.button("Execute Candidate Evaluation & Ranking", use_container_width=True):
        if not jd_text.strip():
            st.error("Please specify a valid Job Description before executing evaluation.")
        elif not resume_files_data:
            st.error("Please upload candidate resumes or select benchmark sample files.")
        else:
            with st.spinner("Calculating Sentence-BERT Embeddings, GitHub Audits, and XAI Explanations..."):
                results = []
                for idx, (fname, r_text, f_obj) in enumerate(resume_files_data, 1):
                    try:
                        analysis = analyze_candidate(
                            resume_text=r_text,
                            jd_text=jd_text,
                            min_experience_years=min_exp_req,
                            weights=weights,
                            anonymize=anonymize
                        )

                        display_name = f"Candidate #{100+idx}" if anonymize else fname
                        analysis["filename"] = display_name
                        analysis["original_filename"] = fname
                        analysis["raw_text"] = r_text

                        if f_obj is not None:
                            fraud_info = detect_ats_fraud(f_obj, fname)
                        else:
                            fraud_info = {"fraud_detected": False, "reasons": []}

                        analysis["fraud_info"] = fraud_info
                        results.append(analysis)
                    except Exception as e:
                        st.error(f"Error processing {fname}: {e}")

                results.sort(key=lambda x: x["overall_score"], reverse=True)
                st.session_state["screening_results"] = results
                st.session_state["target_jd_title"] = jd_title

    # Render Screening Results
    if "screening_results" in st.session_state and st.session_state["screening_results"]:
        results = st.session_state["screening_results"]
        jd_title = st.session_state.get("target_jd_title", "Position")

        st.markdown("### Candidate Screening Leaderboard & Visual Analytics")

        try:
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            total_cand = len(results)
            qualified_cand = len([c for c in results if c["overall_score"] >= min_threshold])
            top_score = results[0]["overall_score"] if results else 0
            avg_score = round(sum([c["overall_score"] for c in results]) / max(total_cand, 1), 1)

            with kpi1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Submissions</div><div class="metric-value">{total_cand}</div></div>', unsafe_allow_html=True)
            with kpi2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Qualified (≥ {min_threshold}%)</div><div class="metric-value" style="color:#34D399">{qualified_cand}</div></div>', unsafe_allow_html=True)
            with kpi3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Top Score</div><div class="metric-value" style="color:#3B82F6">{top_score}%</div></div>', unsafe_allow_html=True)
            with kpi4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Average Score</div><div class="metric-value" style="color:#FBBF24">{avg_score}%</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Error rendering metric cards: {e}")

        st.markdown("<br/>", unsafe_allow_html=True)

        c_left, c_right = st.columns([1.1, 0.9])

        with c_left:
            st.markdown("#### Candidate Leaderboard Summary")
            try:
                df_summary = []
                for i, c in enumerate(results, 1):
                    fraud_status = "Flagged (Hidden Text)" if c.get("fraud_info", {}).get("fraud_detected") else "Passed"
                    github_index = c.get("github_audit", {}).get("credibility_index", "Unverified")

                    df_summary.append({
                        "Rank": i,
                        "Candidate Name": c["filename"],
                        "Match Score (%)": f"{c['overall_score']}%",
                        "GitHub Index": github_index,
                        "Anti-Fraud Status": fraud_status
                    })
                st.dataframe(pd.DataFrame(df_summary), use_container_width=True)

                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    csv_data = generate_csv_report(results)
                    st.download_button(
                        label="Download Full Screening Audit (CSV)",
                        data=csv_data,
                        file_name=f"screening_audit_{jd_title.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with exp_col2:
                    if results:
                        top_pdf_bytes = generate_pdf_report(results[0], jd_title)
                        st.download_button(
                            label="Export Top Candidate PDF Report",
                            data=top_pdf_bytes,
                            file_name=f"Report_{results[0]['filename']}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error rendering leaderboard table: {e}")

        with c_right:
            st.markdown("#### 4-Dimension Candidate Evaluation Radar")
            try:
                fig_radar_4d = create_4d_candidate_radar_chart(results)
                st.plotly_chart(fig_radar_4d, use_container_width=True)
            except Exception as e:
                st.warning(f"Error rendering 4D candidate radar chart: {e}")

        st.markdown("---")
        st.markdown("### Candidate Evaluation Profiles & XAI Breakdown")

        for c in results:
            if c["overall_score"] < min_threshold:
                continue

            with st.expander(f"Rank #{results.index(c)+1}: {c['filename']} — Score: {c['overall_score']}% ({c['tier']})"):
                if c.get("fraud_info", {}).get("fraud_detected"):
                    reasons = c["fraud_info"].get("reasons", [])
                    st.error(f"ATS Fraud Warning: Hidden text or font manipulation detected ({', '.join(reasons)}).")

                st.markdown('<div class="card-box">', unsafe_allow_html=True)
                st.markdown("#### Explainable AI (XAI) Match Justification Card")
                xai = c.get("xai_explanation", {})
                st.info(f"**Natural Language Summary:** {xai.get('justification_summary', 'N/A')}")

                xai_col1, xai_col2 = st.columns(2)
                with xai_col1:
                    st.markdown("**Positive Score Contributors:**")
                    for pos in xai.get("positive_contributors", []):
                        st.markdown(f"- <font color='#34D399'>{pos}</font>", unsafe_allow_html=True)
                with xai_col2:
                    st.markdown("**Score Deductions:**")
                    for ded in xai.get("deductions", []):
                        st.markdown(f"- <font color='#F87171'>{ded}</font>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                d_col1, d_col2 = st.columns([1, 1])

                with d_col1:
                    try:
                        fig_g = create_score_gauge(c['overall_score'], f"{c['filename']} Score")
                        st.plotly_chart(fig_g, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error loading gauge: {e}")

                    st.markdown(f"**Email:** `{c['contact_info']['email']}` | **Phone:** `{c['contact_info']['phone']}`")
                    st.markdown(f"**Education:** {c['education']} | **Extracted Experience:** {c['experience_years']} Years")

                    gh = c.get("github_audit", {})
                    st.markdown(f"**GitHub Credibility:** `{gh.get('credibility_index')}` ({gh.get('public_repos', 0)} public repos)")

                with d_col2:
                    try:
                        fig_break = create_score_breakdown_bar(c)
                        st.plotly_chart(fig_break, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error loading breakdown chart: {e}")

                with st.expander("Suggested Technical Interview Questions for Recruiter", expanded=False):
                    questions = c.get("interview_questions", [])
                    if questions:
                        for q_idx, q_txt in enumerate(questions, 1):
                            st.markdown(f"**Q{q_idx}:** {q_txt}")
                    else:
                        st.write("No missing skill questions generated.")

                gap_col1, gap_col2 = st.columns([1, 1])
                with gap_col1:
                    try:
                        fig_gap = create_skill_gap_chart(c["matched_skills"], c["missing_skills"])
                        st.plotly_chart(fig_gap, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error loading skill gap chart: {e}")

                with gap_col2:
                    st.markdown("**Matched Skills:**")
                    if c["matched_skills"]:
                        pills_html = "".join([f'<span class="skill-pill">✓ {s}</span>' for s in c["matched_skills"]])
                        st.markdown(pills_html, unsafe_allow_html=True)
                    else:
                        st.write("None identified.")

                    st.markdown("<br/>**Missing Skills:**", unsafe_allow_html=True)
                    if c["missing_skills"]:
                        pills_missing = "".join([f'<span class="skill-pill skill-pill-missing">✗ {s}</span>' for s in c["missing_skills"]])
                        st.markdown(pills_missing, unsafe_allow_html=True)
                    else:
                        st.write("None! All target skills present.")


def render_system_analytics_and_reference():
    """Render System Analytics and Architecture Reference using Tabs."""
    st.markdown("### System Analytics & Reference")

    tab_sec1, tab_sec2 = st.tabs([
        "Skill Taxonomy Analytics",
        "Architecture & Technical Reference"
    ])

    with tab_sec1:
        st.caption("Explore the categorized skill taxonomy containing 500+ technical and domain competencies.")
        try:
            skill_db = load_skill_db()
            total_skills_count = sum(len(skills) for skills in skill_db.values())
            st.markdown(f'<div style="margin-bottom:1rem;"><span class="badge badge-recommended" style="font-size:0.95rem;">Total Skills in Taxonomy Database: {total_skills_count} Unique Skills</span></div>', unsafe_allow_html=True)

            category_key = st.segmented_control(
                "Domain Category:",
                list(skill_db.keys()),
                default=list(skill_db.keys())[0]
            )

            if category_key:
                skills = skill_db[category_key]
                st.markdown(f"#### Domain: `{category_key.upper().replace('_', ' ')}` ({len(skills)} Skills)")
                pills_html = "".join([f'<span class="skill-pill" style="font-size:0.9rem; padding: 0.35rem 0.75rem;">{s}</span>' for s in skills])
                st.markdown(pills_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Skill Distribution Across Categories")
            fig_interactive_tax = create_interactive_taxonomy_chart(skill_db, category_key)
            st.plotly_chart(fig_interactive_tax, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering taxonomy analytics: {e}")

    with tab_sec2:
        st.caption("Technical specification of NLP pipeline, embedding models, and scoring equations.")
        st.markdown(r"""
        #### Architecture & Data Pipeline
        ```
        +-----------------------+     +-----------------------+
        | Candidate Resume File |     | Job Description (JD)  |
        | (PDF / DOCX / TXT)    |     | (Text / File Input)   |
        +-----------+-----------+     +-----------+-----------+
                    |                             |
                    v                             v
        +-----------------------+     +-----------------------+
        |  Document Parser      |     |  ATS Fraud Inspector  |
        |  (pdfplumber / docx)  |     |  (White text/Fonts)   |
        +-----------+-----------+     +-----------+-----------+
                    |                             |
                    v                             v
        +-----------------------------------------------------+
        |              NLP Preprocessing Pipeline             |
        | (Lowercasing, RegEx URL Strip, Stop-words, Lemmatization) |
        +--------------------------+--------------------------+
                                   |
                                   v
        +-----------------------------------------------------+
        |           Hybrid Scoring & Analytics Engine         |
        | 1. Sentence-BERT Embeddings ('all-MiniLM-L6-v2')    |
        | 2. TF-IDF N-Gram Vectorization & Cosine Similarity  |
        | 3. GitHub Profile API Credibility Inspector          |
        | 4. Explainable AI (XAI) Match Justification Engine  |
        | 5. Candidate Workspace Copilot Engine               |
        | 6. Anti-Bias Candidate Anonymizer Engine            |
        +--------------------------+--------------------------+
                                   |
                                   v
        +-----------------------------------------------------+
        |        Interactive Dashboard & Reporting            |
        |  (Leaderboard, Plotly Gauges/Radar, PDF/CSV Export) |
        +-----------------------------------------------------+
        ```

        ---

        #### Mathematical Formulations

        ##### 1. Sentence-BERT Dense Embedding Cosine Similarity
        $$\text{Sim}_{\text{SBERT}}(R, JD) = \frac{\mathbf{e}_R \cdot \mathbf{e}_{JD}}{\|\mathbf{e}_R\| \|\mathbf{e}_{JD}\|}$$

        ##### 2. TF-IDF Representation
        $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$

        ##### 3. Hybrid Scoring Equation
        $$\text{Final Score} = w_{\text{sbert}} \cdot S_{\text{SBERT}} + w_{\text{tfidf}} \cdot S_{\text{TF-IDF}} + w_{\text{skills}} \cdot S_{\text{Skills}} + w_{\text{exp}} \cdot S_{\text{Exp}}$$
        """)


def main():
    """Application Entry Point."""
    load_application_css()
    render_header()

    app_mode, weights, min_threshold, anonymize = render_sidebar()

    # Render AI Co-Pilot chatbot in sidebar (context-aware)
    render_copilot_chatbot(app_mode, weights, min_threshold, anonymize)

    if app_mode == "Candidate Workspace":
        render_candidate_hub(weights)
    elif app_mode == "Recruiter Suite":
        render_recruiter_dashboard(weights, min_threshold, anonymize)
    else:
        render_system_analytics_and_reference()


if __name__ == "__main__":
    main()
