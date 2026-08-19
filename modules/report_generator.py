import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_csv_report(candidate_results: list) -> str:
    """
    Generate CSV audit report string containing all candidate evaluation metrics, fraud status, GitHub credibility, and missing skills.
    """
    data = []
    for c in candidate_results:
        fraud_status = "Flagged (Hidden Text)" if c.get("fraud_info", {}).get("fraud_detected") else "Passed"
        github_cred = c.get("github_audit", {}).get("credibility_index", "Unverified")

        data.append({
            "Candidate Name / File": c.get("filename", "Candidate"),
            "Overall Match Score (%)": c.get("overall_score", 0),
            "Recommendation Tier": c.get("tier", "N/A"),
            "Sentence-BERT Semantic (%)": c.get("sbert_score", 0),
            "TF-IDF Keyword Similarity (%)": c.get("tfidf_score", 0),
            "Skill Match Score (%)": c.get("skill_score", 0),
            "Experience (Years)": c.get("experience_years", 0),
            "Highest Education": c.get("education", "N/A"),
            "Anti-Fraud Audit": fraud_status,
            "GitHub Credibility": github_cred,
            "Email": c.get("contact_info", {}).get("email", "N/A"),
            "Phone": c.get("contact_info", {}).get("phone", "N/A"),
            "Matched Skills": ", ".join(c.get("matched_skills", [])),
            "Missing Skills": ", ".join(c.get("missing_skills", []))
        })
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def generate_pdf_report(candidate_data: dict, jd_title: str) -> bytes:
    """
    Generate formal PDF Evaluation Audit Report containing XAI Justifications, GitHub Credibility, and Technical Interview Questions.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontSize=20, leading=24, textColor=colors.HexColor('#0F172A'), spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontSize=11, leading=15, textColor=colors.HexColor('#64748B'), spaceAfter=12
    )
    section_style = ParagraphStyle(
        'SectionHeader', parent=styles['Heading2'],
        fontSize=13, leading=17, textColor=colors.HexColor('#1E293B'), spaceBefore=10, spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyTextCustom', parent=styles['Normal'],
        fontSize=9.5, leading=13.5, textColor=colors.HexColor('#334155')
    )

    story = []

    # Header
    filename = candidate_data.get("filename", "Candidate")
    story.append(Paragraph("Candidate Evaluation Audit Report", title_style))
    story.append(Paragraph(f"Target Position: {jd_title} &nbsp;|&nbsp; Candidate: {filename}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=12))

    # Core Metric Table
    fraud_text = "FLAGGED ⚠️" if candidate_data.get("fraud_info", {}).get("fraud_detected") else "PASSED ✓"
    github_cred = candidate_data.get("github_audit", {}).get("credibility_index", "Unverified")

    metrics_table_data = [
        [
            Paragraph("<b>Overall Match Score</b>", body_style),
            Paragraph(f"<b><font color='#2563EB'>{candidate_data.get('overall_score', 0)}%</font></b>", body_style),
            Paragraph("<b>Recommendation Tier</b>", body_style),
            Paragraph(f"<b>{candidate_data.get('tier', 'N/A')}</b>", body_style)
        ],
        [
            Paragraph("SBERT Semantic Score", body_style),
            Paragraph(f"{candidate_data.get('sbert_score', 0)}%", body_style),
            Paragraph("Hard Skill Match Score", body_style),
            Paragraph(f"{candidate_data.get('skill_score', 0)}%", body_style)
        ],
        [
            Paragraph("Anti-Fraud Audit", body_style),
            Paragraph(f"<b>{fraud_text}</b>", body_style),
            Paragraph("GitHub Credibility", body_style),
            Paragraph(f"<b>{github_cred}</b>", body_style)
        ]
    ]

    t = Table(metrics_table_data, colWidths=[130, 130, 130, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # XAI Justification Summary
    xai = candidate_data.get("xai_explanation", {})
    story.append(Paragraph("Explainable AI (XAI) Match Justification", section_style))
    summary_txt = xai.get("justification_summary", "Candidate meets technical requirements.")
    story.append(Paragraph(f"<i>{summary_txt}</i>", body_style))
    story.append(Spacer(1, 8))

    pos_contributors = "<br/>• ".join(xai.get("positive_contributors", []))
    if pos_contributors:
        story.append(Paragraph(f"<b>Positive Score Contributors:</b><br/>• {pos_contributors}", body_style))
        story.append(Spacer(1, 8))

    # Matched & Missing Skills Table
    story.append(Paragraph("Skill Alignment Breakdown", section_style))
    matched = ", ".join(candidate_data.get("matched_skills", [])) or "None"
    missing = ", ".join(candidate_data.get("missing_skills", [])) or "None! All required skills present."

    skill_table_data = [
        [Paragraph("<b>Matched Skills</b>", body_style), Paragraph(matched, body_style)],
        [Paragraph("<b>Missing Critical Skills</b>", body_style), Paragraph(missing, body_style)]
    ]
    st_table = Table(skill_table_data, colWidths=[140, 380])
    st_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(st_table)
    story.append(Spacer(1, 12))

    # Technical Interview Questions
    questions = candidate_data.get("interview_questions", [])
    if questions:
        story.append(Paragraph("Suggested Candidate Technical Interview Questions", section_style))
        for idx, q in enumerate(questions, 1):
            story.append(Paragraph(f"<b>Q{idx}:</b> {q}", body_style))
            story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=8))
    story.append(Paragraph("<i>Generated automatically by AI-Resume Screening & Candidate Analytics Engine</i>", ParagraphStyle('Footer', parent=body_style, fontSize=8, textColor=colors.HexColor('#94A3B8'))))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
