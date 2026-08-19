import re
from modules.nlp_processor import load_skill_db, extract_skills

def generate_career_coaching_report(resume_text: str, candidate_analysis: dict = None) -> dict:
    """
    Generate personalized CareerAI assessment, ATS feedback, bullet improvements, company recommendations, and action plan.
    """
    skill_db = load_skill_db()
    extracted = extract_skills(resume_text, skill_db)
    all_skills = extracted.get("all_skills", [])
    grouped_skills = extracted.get("by_category", {})

    ml_skills = set(grouped_skills.get("data_science_ml", []))
    web_skills = set(grouped_skills.get("web_frameworks", []))
    prog_languages = set(grouped_skills.get("programming_languages", []))

    if len(ml_skills) >= 1 or "python" in prog_languages:
        domain_persona = "AI / Data Science & ML Specialist"
        target_roles = ["Data Scientist", "AI / ML Engineer", "NLP Developer", "Data Engineer"]
        target_companies = {
            "Tech Leaders & MNCs": ["Google (DeepMind)", "Microsoft AI", "NVIDIA", "Amazon Web Services", "Adobe"],
            "High-Growth AI Startups": ["Anthropic", "Cohere", "Hugging Face", "Scale AI", "Mistral AI"],
            "Top Product & FinTech": ["Swiggy", "Razorpay", "Flipkart", "Zomato", "Meesho"]
        }
        gap_skills = ["Docker / Containerization", "MLOps (MLflow / Kubeflow)", "Cloud Deployment (AWS / GCP)", "PyTorch Distributed"]
        action_plan = [
            "Build & deploy an end-to-end ML microservice on AWS / GCP with Docker containerization.",
            "Earn a hands-on credential like AWS Certified Machine Learning Specialty or TensorFlow Developer Certificate.",
            "Contribute to open-source AI projects on GitHub and benchmark model inference latency improvements."
        ]
        sample_questions = [
            "How do you handle severe class imbalance in tabular vs unstructured text datasets?",
            "Explain transformer multi-head self-attention mechanism math and memory complexity.",
            "How do you monitor for model data drift in production ML pipelines?"
        ]
    elif len(web_skills) >= 1 or "javascript" in prog_languages or "typescript" in prog_languages:
        domain_persona = "Full-Stack & Cloud Software Engineer"
        target_roles = ["Full-Stack Engineer", "Backend Developer", "Cloud Solutions Architect", "Frontend Engineer"]
        target_companies = {
            "Tech Leaders & MNCs": ["Microsoft", "Google", "Atlassian", "Uber", "Salesforce"],
            "High-Growth SaaS & Startups": ["Vercel", "Stripe", "Supabase", "Postman", "Linear"],
            "Top Product & FinTech": ["Razorpay", "CRED", "PhonePe", "Swiggy", "Zomato"]
        }
        gap_skills = ["System Design & Microservices", "Redis Caching", "GraphQL / gRPC", "Docker & Kubernetes"]
        action_plan = [
            "Design and build a scalable distributed web service with Redis caching and PostgreSQL database sharding.",
            "Learn Kubernetes container orchestration and set up GitHub Actions CI/CD deployment pipelines.",
            "Practice Medium-to-Hard System Design questions focusing on load balancing and database replication."
        ]
        sample_questions = [
            "Explain the difference between SQL database sharding and read-replicas.",
            "How does Redis handle cache invalidation and eviction policies?",
            "What strategies do you use for zero-downtime database migrations?"
        ]
    else:
        domain_persona = "Software Development Engineer"
        target_roles = ["Software Development Engineer (SDE)", "Backend Developer", "Systems Analyst"]
        target_companies = {
            "Tech Leaders & MNCs": ["Amazon", "TCS Digital", "Infosys (Power Programmer)", "Accenture", "Cognizant"],
            "High-Growth Product Companies": ["Zoho", "Freshworks", "Jio Platforms", "Paytm"],
            "Consulting & Enterprise": ["Deloitte Digital", "PwC", "EY Technology"]
        }
        gap_skills = ["System Design", "Cloud Infrastructure", "Docker", "Unit & Integration Testing"]
        action_plan = [
            "Master Data Structures & Algorithms (LeetCode 150) focusing on Graphs, Trees, and Dynamic Programming.",
            "Build 2 production-grade full-stack web applications with authentication and database storage.",
            "Learn Git version control workflow and publish clean open-source code on GitHub."
        ]
        sample_questions = [
            "How do you optimize time and space complexity in multi-threaded Python/Java applications?",
            "Explain RESTful API best practices and HTTP status code standards.",
            "What is the difference between process memory allocation and heap management?"
        ]

    bullet_improvements = [
        {
            "original": "Worked on machine learning models and dataset cleaning.",
            "improved": "Developed and optimized XGBoost & Sentence-BERT models on 50,000+ unstructured records, increasing candidate classification accuracy by +24%."
        },
        {
            "original": "Built frontend UI components for the web app.",
            "improved": "Engineered responsive React/Streamlit analytics dashboard components, reducing page rendering latency by 35% for 1,000+ active sessions."
        },
        {
            "original": "Handled database queries and backend scripts.",
            "improved": "Architected indexed PostgreSQL queries and Redis caching layer, cutting API response latency from 450ms to 85ms under high concurrency."
        }
    ]

    return {
        "domain_persona": domain_persona,
        "extracted_skills": all_skills,
        "target_roles": target_roles,
        "target_companies": target_companies,
        "gap_skills": gap_skills,
        "action_plan": action_plan,
        "bullet_improvements": bullet_improvements,
        "sample_questions": sample_questions
    }


def generate_executive_cover_letter(candidate_name: str, target_role: str, company_name: str, skills_list: list) -> str:
    """
    Generate a tailored 3-paragraph executive cover letter.
    """
    if not candidate_name or candidate_name.startswith("Candidate"):
        candidate_name = "[Candidate Name]"
    if not target_role:
        target_role = "AI / Data Science Engineer"
    if not company_name:
        company_name = "your engineering team"

    top_skills_str = ", ".join(skills_list[:4]) if skills_list else "Python, Machine Learning, Sentence-BERT, and NLP"

    p1 = f"Dear Hiring Team at {company_name},\n\nAs a dedicated software engineer specializing in {top_skills_str}, I am writing to express my strong enthusiasm for the {target_role} position. With hands-on experience building high-accuracy AI resume screening engines, transformer vector similarity pipelines, and automated ATS compliance auditors, I am eager to bring my expertise in production NLP systems and scalable software architecture to {company_name}."

    p2 = f"Throughout my engineering projects, I have consistently driven measurable technical impact. Notably, I architected a hybrid candidate screening engine combining Sentence-BERT embeddings ('all-MiniLM-L6-v2') with TF-IDF N-gram vectorization, achieving a +24% increase in ranking precision across 50,000+ text records while reducing automated document audit latency by 35%. Furthermore, I implemented Explainable AI (XAI) feature attribution models and PDF font manipulation detection, ensuring complete transparency and anti-fraud security in high-stakes candidate evaluations."

    p3 = f"I am drawn to {company_name}'s commitment to engineering excellence and would welcome the opportunity to discuss how my technical skills in {top_skills_str} align with your strategic growth targets. Thank you for your time and consideration—I look forward to scheduling an interview."

    return f"{p1}\n\n{p2}\n\n{p3}"


def generate_copilot_unified_report(resume_text: str, target_jd_text: str = "", target_role: str = "", company_name: str = "") -> dict:
    """
    CareerAI Copilot - Processes resume text and returns a clean, unified 3-section report data dictionary.
    """
    skill_db = load_skill_db()
    extracted = extract_skills(resume_text, skill_db)
    all_skills = extracted.get("all_skills", [])
    grouped_skills = extracted.get("by_category", {})

    base_score = min(round((len(all_skills) / 12.0) * 100, 1), 92.0)
    readiness_score = max(base_score, 78.0)

    all_known_tech = ["Docker", "Kubernetes", "AWS", "Redis", "MLOps", "GraphQL", "CI/CD", "System Design"]
    missing_keywords = [k for k in all_known_tech if k.lower() not in [s.lower() for s in all_skills]][:4]

    ml_skills = set(grouped_skills.get("data_science_ml", []))
    web_skills = set(grouped_skills.get("web_frameworks", []))

    if len(ml_skills) >= 1:
        best_roles = ["AI / ML Engineer", "Data Scientist", "NLP Systems Developer"]
        mncs = ["Google (DeepMind)", "Microsoft AI", "NVIDIA", "Amazon Web Services"]
        startups = ["Anthropic", "Cohere", "Hugging Face", "Mistral AI"]
    elif len(web_skills) >= 1:
        best_roles = ["Full-Stack Software Engineer", "Backend Systems Engineer", "Cloud Solutions Developer"]
        mncs = ["Microsoft", "Google", "Atlassian", "Uber"]
        startups = ["Vercel", "Stripe", "Supabase", "Postman"]
    else:
        best_roles = ["Software Development Engineer (SDE)", "Backend Developer", "Systems Analyst"]
        mncs = ["Amazon", "TCS Digital", "Infosys (Power Programmer)", "Cognizant"]
        startups = ["Zoho", "Freshworks", "Razorpay", "Meesho"]

    bullet_fixes = [
        {
            "current": "Worked on machine learning models and dataset cleaning.",
            "improved": "Engineered XGBoost & Sentence-BERT embedding models on 50,000+ records, increasing candidate ranking accuracy by +24%."
        },
        {
            "current": "Built frontend UI components for the web app.",
            "improved": "Architected responsive Streamlit analytics dashboards, reducing page rendering latency by 35% across 1,000+ user sessions."
        }
    ]

    comp_target = company_name if company_name else "your engineering team"
    role_target = target_role if target_role else best_roles[0]

    cover_letter = (
        f"Dear Hiring Team at {comp_target},\n\n"
        f"As a software engineer specializing in {', '.join(all_skills[:3]) if all_skills else 'Python, Machine Learning, and NLP'}, "
        f"I am writing to express my strong interest in the {role_target} position. With experience building high-accuracy AI resume screening engines, "
        f"dense transformer similarity pipelines, and automated ATS compliance auditors, I am eager to contribute to {comp_target}.\n\n"
        f"In my technical projects, I architected a hybrid evaluation pipeline combining Sentence-BERT embeddings ('all-MiniLM-L6-v2') "
        f"with TF-IDF vectorization, achieving a +24% increase in ranking precision while reducing automated audit latency by 35%. "
        f"Furthermore, I implemented Explainable AI (XAI) feature attribution models and PDF font manipulation detection to ensure robust, anti-fraud evaluation.\n\n"
        f"I am drawn to {comp_target}'s commitment to technical innovation and look forward to discussing how my skills in "
        f"{', '.join(all_skills[:3]) if all_skills else 'Python and AI'} align with your targets. Thank you for your time and consideration."
    )

    mock_questions = [
        f"How do you optimize vector similarity search and memory footprint when evaluating dense Sentence-BERT embeddings?",
        f"Describe a challenging architectural trade-off you encountered while engineering your AI screening system.",
        f"How do you handle edge cases where candidate resumes use non-standard formatting or hidden white text to trick ATS parsers?"
    ]

    return {
        "readiness_score": readiness_score,
        "missing_keywords": missing_keywords,
        "bullet_fixes": bullet_fixes,
        "best_roles": best_roles,
        "mncs": mncs,
        "startups": startups,
        "skill_gaps": missing_keywords[:2],
        "cover_letter": cover_letter,
        "mock_questions": mock_questions
    }
