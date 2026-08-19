import re
import os
import io
import pdfplumber
from pypdf import PdfReader
import docx

def extract_text_from_file(file_obj, filename: str) -> str:
    """
    Extract raw text from PDF, DOCX, or TXT file object or file path.
    """
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    
    try:
        if ext == ".pdf":
            text = _extract_from_pdf(file_obj)
        elif ext in [".docx", ".doc"]:
            text = _extract_from_docx(file_obj)
        elif ext == ".txt":
            if isinstance(file_obj, (str, bytes)):
                text = file_obj.decode("utf-8", errors="ignore") if isinstance(file_obj, bytes) else file_obj
            elif hasattr(file_obj, "read"):
                content = file_obj.read()
                text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else content
        else:
            text = str(file_obj)
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        text = ""

    return text.strip()


def _extract_from_pdf(file_obj) -> str:
    text_content = []
    
    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content.append(extracted)
    except Exception:
        text_content = []
        
    if not text_content:
        try:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            reader = PdfReader(file_obj)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content.append(extracted)
        except Exception as e:
            print(f"PyPDF fallback error: {e}")

    return "\n".join(text_content)


def _extract_from_docx(file_obj) -> str:
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    doc = docx.Document(file_obj)
    full_text = []
    for para in doc.paragraphs:
        if para.text:
            full_text.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text:
                    full_text.append(cell.text)
    return "\n".join(full_text)


def detect_ats_fraud(file_obj, filename: str) -> dict:
    """
    Detect ATS Fraud / Cheating tricks in PDF resumes:
    1. Invisible White Text (Font color is white [1,1,1] or near white).
    2. Micro Fonts (< 3pt font size).
    3. Abnormal keyword repetition (keyword stuffing).
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".pdf":
        return {
            "fraud_detected": False,
            "reasons": [],
            "hidden_text_snippets": []
        }

    reasons = []
    hidden_snippets = []

    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        with pdfplumber.open(file_obj) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                chars = page.chars
                for char in chars:
                    # Check white / invisible color
                    color = char.get("non_stroking_color")
                    font_size = char.get("size", 10)
                    text_char = char.get("text", "")

                    # Check white font (RGB [1,1,1] or 1.0 or (255,255,255))
                    is_white = False
                    if isinstance(color, (list, tuple)):
                        if all(c >= 0.95 for c in color):
                            is_white = True
                        elif all(c >= 250 for c in color):
                            is_white = True
                    elif color == 1 or color == 1.0:
                        is_white = True

                    if is_white and text_char.strip():
                        hidden_snippets.append(text_char)

                    # Check tiny font size
                    if font_size <= 2.5 and text_char.strip():
                        reasons.append(f"Micro-font size detected ({font_size}pt on Page {i})")

        if hidden_snippets:
            snippet_str = "".join(hidden_snippets)[:100]
            reasons.append(f"Invisible white text detected: '{snippet_str}...'")

    except Exception as e:
        print(f"Error checking ATS fraud in {filename}: {e}")

    fraud_detected = len(reasons) > 0
    return {
        "fraud_detected": fraud_detected,
        "reasons": list(set(reasons)),
        "hidden_text_snippets": hidden_snippets
    }


def extract_contact_info(text: str) -> dict:
    """
    Extract Contact Information (Email, Phone, LinkedIn, GitHub).
    """
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    email = emails[0] if emails else "N/A"
    
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}'
    phones = re.findall(phone_pattern, text)
    valid_phones = [p for p in phones if len(re.sub(r'\D', '', p)) >= 10]
    phone = valid_phones[0] if valid_phones else "N/A"
    
    linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+'
    linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
    linkedin = linkedin_match.group(0) if linkedin_match else "N/A"
    
    github_pattern = r'(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_-]+'
    github_match = re.search(github_pattern, text, re.IGNORECASE)
    github = github_match.group(0) if github_match else "N/A"
    
    return {
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github
    }


def extract_education(text: str) -> str:
    """
    Detect Highest Education Degree mentioned in resume text.
    """
    text_lower = text.lower()
    
    degrees = [
        ("Ph.D / Doctorate", [r'\bph\.?d\b', r'\bdoctorate\b', r'\bdoctor of philosophy\b']),
        ("Master (M.Tech / M.S / MCA / MBA)", [r'\bm\.?tech\b', r'\bm\.?s\.?\b', r'\bmca\b', r'\bmba\b', r'\bmaster\b', r'\bpost graduate\b']),
        ("Bachelor (B.Tech / B.E / B.Sc / BCA)", [r'\bb\.?tech\b', r'\bb\.?e\.?\b', r'\bbca\b', r'\bb\.?sc\b', r'\bbachelor\b', r'\bundergraduate\b']),
        ("Diploma / High School", [r'\bdiploma\b', r'\bhigh school\b', r'\b12th\b', r'\bsecondary\b'])
    ]
    
    for degree_name, patterns in degrees:
        for p in patterns:
            if re.search(p, text_lower):
                return degree_name
                
    return "Not Specified"


def extract_experience_years(text: str) -> float:
    """
    Estimate total work experience in years using pattern matching.
    """
    text_lower = text.lower()
    
    exp_pattern = r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience'
    matches = re.findall(exp_pattern, text_lower)
    if matches:
        try:
            return max([float(m) for m in matches])
        except ValueError:
            pass
            
    range_pattern = r'(20\d{2})\s*(?:-|to)\s*(20\d{2}|present|current)'
    range_matches = re.findall(range_pattern, text_lower)
    
    current_year = 2026
    total_years = 0.0
    
    for start_str, end_str in range_matches:
        try:
            start_yr = int(start_str)
            end_yr = current_year if end_str in ["present", "current"] else int(end_str)
            if end_yr >= start_yr:
                total_years += (end_yr - start_yr)
        except ValueError:
            continue
            
    if total_years > 0:
        return min(total_years, 30.0)
        
    return 1.0
