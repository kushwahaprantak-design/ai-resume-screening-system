import re

def analyze_ats_compliance(resume_text: str, analysis_result: dict = None) -> dict:
    """
    Analyze resume formatting, section completeness, word count, and keyword density for ATS optimization.
    """
    text_lower = resume_text.lower()
    words = resume_text.split()
    word_count = len(words)
    
    # 1. Section Detection
    essential_sections = {
        "Summary / Objective": [r'\bsummary\b', r'\bobjective\b', r'\bprofile\b', r'\babout me\b'],
        "Work Experience": [r'\bexperience\b', r'\bwork history\b', r'\bemployment\b'],
        "Education": [r'\beducation\b', r'\bacademics\b', r'\bqualifications\b'],
        "Skills": [r'\bskills\b', r'\btechnical skills\b', r'\bexpertise\b', r'\btechnologies\b'],
        "Projects": [r'\bprojects\b', r'\bpersonal projects\b', r'\bacademic projects\b']
    }
    
    section_checks = {}
    found_sections_count = 0
    for section_name, patterns in essential_sections.items():
        found = any(re.search(p, text_lower) for p in patterns)
        section_checks[section_name] = found
        if found:
            found_sections_count += 1
            
    section_score = round((found_sections_count / len(essential_sections)) * 100, 2)
    
    # 2. Word Count Health Check
    if 250 <= word_count <= 1000:
        word_count_status = "Optimal Length (1-2 pages)"
        word_count_score = 100.0
    elif 150 <= word_count < 250:
        word_count_status = "Slightly Short"
        word_count_score = 75.0
    elif word_count > 1000:
        word_count_status = "Too Long (> 2 pages)"
        word_count_score = 60.0
    else:
        word_count_status = "Critically Short (< 150 words)"
        word_count_score = 40.0

    # 3. Contact Info Presence
    contact_info = analysis_result.get("contact_info", {}) if analysis_result else {}
    has_email = contact_info.get("email") != "N/A"
    has_phone = contact_info.get("phone") != "N/A"
    has_linkedin = contact_info.get("linkedin") != "N/A"
    
    contact_score = 0
    if has_email: contact_score += 40
    if has_phone: contact_score += 40
    if has_linkedin: contact_score += 20
    
    # 4. Overall ATS Score Calculation
    overall_ats_score = round(
        (section_score * 0.40) + 
        (contact_score * 0.30) + 
        (word_count_score * 0.30), 
        2
    )

    # 5. Improvement Suggestions
    suggestions = []
    
    if not section_checks.get("Summary / Objective"):
        suggestions.append("➕ Add a clear 'Professional Summary' or 'Objective' section at the top.")
    if not section_checks.get("Projects"):
        suggestions.append("➕ Add a 'Projects' section to showcase practical coding & problem-solving experience.")
    if not has_linkedin:
        suggestions.append("🔗 Add your LinkedIn profile URL for recruiter verification.")
    if word_count < 250:
        suggestions.append("📝 Expand bullet points with measurable achievements (e.g. 'Increased accuracy by 15%').")
    if analysis_result and analysis_result.get("missing_skills"):
        missing = analysis_result.get("missing_skills")[:5]
        suggestions.append(f"🔑 Add key target skills missing from your resume: {', '.join(missing)}")
        
    if not suggestions:
        suggestions.append("✅ Great job! Your resume format is highly ATS-compliant.")

    return {
        "ats_score": overall_ats_score,
        "word_count": word_count,
        "word_count_status": word_count_status,
        "section_checks": section_checks,
        "contact_completeness": {
            "has_email": has_email,
            "has_phone": has_phone,
            "has_linkedin": has_linkedin
        },
        "suggestions": suggestions
    }
