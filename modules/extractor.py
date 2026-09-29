"""
extractor.py
------------
Handles all document parsing and regex-based info extraction:
  - PDF (pdfplumber primary, pypdf fallback)
  - DOCX (python-docx)
  - TXT
  - ATS fraud detection (hidden white text, micro fonts)
  - Contact info, education level, experience years

NOTE: pdfplumber is more accurate than pypdf for layout-heavy PDFs, so we
try it first. The pypdf fallback handles encrypted/older PDF versions that
pdfplumber chokes on. If both fail, we return an empty string and log it.

TODO: add OCR fallback (pytesseract) for scanned image-based PDFs — currently
      these return empty text and get a 0 score which is misleading
"""

import re
import os
import io
import pdfplumber
from pypdf import PdfReader
import docx


# ── Public entry point ────────────────────────────────────────────────────────

def extract_text_from_file(file_obj, filename: str) -> str:
    """
    Route the file to the correct parser based on extension.
    Returns raw text string, or empty string on failure (never raises).
    """
    ext = os.path.splitext(filename)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            text = _parse_pdf(file_obj)
        elif ext in [".docx", ".doc"]:
            text = _parse_docx(file_obj)
        elif ext == ".txt":
            text = _parse_txt(file_obj)
        else:
            # unknown format — try converting to string (last resort)
            text = str(file_obj)
    except Exception as e:
        # log but don't crash — UI will show a warning
        print(f"[ERROR] Text extraction failed for {filename}: {e}")
        text = ""

    return text.strip()


# ── PDF parsing ───────────────────────────────────────────────────────────────

def _parse_pdf(file_obj) -> str:
    """
    Try pdfplumber first, then pypdf as fallback.

    Edge case: some PDFs are scanned images — both libraries return empty text.
    In that case the caller gets "" and should show a 'could not extract text'
    warning (OCR would be needed but that's a future enhancement).
    """
    pages_text = []

    # ── Primary: pdfplumber ───────────────────────────────────────────────────
    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
    except Exception as e:
        print(f"[WARN] pdfplumber failed: {e} — trying pypdf fallback")
        pages_text = []

    # ── Fallback: pypdf ───────────────────────────────────────────────────────
    if not pages_text:
        try:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            reader = PdfReader(file_obj)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
        except Exception as e:
            print(f"[ERROR] pypdf fallback also failed: {e}")

    return "\n".join(pages_text)


# ── DOCX parsing ──────────────────────────────────────────────────────────────

def _parse_docx(file_obj) -> str:
    """
    Extract text from paragraphs AND table cells.
    Tables often contain experience/skills in two-column layouts — easy to miss.
    """
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    doc = docx.Document(file_obj)
    parts = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    # don't skip table cells — many resume templates put skills/dates in tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)

    return "\n".join(parts)


# ── TXT parsing ───────────────────────────────────────────────────────────────

def _parse_txt(file_obj) -> str:
    """Read plain text file — handles both bytes and string file objects."""
    if isinstance(file_obj, bytes):
        return file_obj.decode("utf-8", errors="ignore")
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "read"):
        content = file_obj.read()
        return content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else content
    return ""


# ── ATS fraud detection ───────────────────────────────────────────────────────

def detect_ats_fraud(file_obj, filename: str) -> dict:
    """
    Scan a PDF for common ATS-cheating tricks:
      1. Invisible white text (font color ~white = RGB [1,1,1] or 0xFFFFFF)
      2. Micro-fonts (< 3pt) — invisible to humans but parsed by ATS bots
      3. (planned) keyword stuffing detection — checking repetition ratio

    Non-PDF files are returned as fraud_detected=False immediately.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".pdf":
        return {"fraud_detected": False, "reasons": [], "hidden_text_snippets": []}

    reasons = []
    hidden_snippets = []

    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)

        with pdfplumber.open(file_obj) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for char in page.chars:
                    color     = char.get("non_stroking_color")
                    font_size = char.get("size", 10)
                    glyph     = char.get("text", "")

                    # ── check invisible/white fill color ─────────────────────
                    is_white = False
                    if isinstance(color, (list, tuple)):
                        # PDF colors are 0.0–1.0 floats; near-white = all >= 0.95
                        if all(c >= 0.95 for c in color):
                            is_white = True
                        # some PDFs encode as 0–255 integers
                        elif all(c >= 242 for c in color):
                            is_white = True
                    elif color in (1, 1.0):
                        is_white = True

                    if is_white and glyph.strip():
                        hidden_snippets.append(glyph)

                    # ── check micro font (< 2.5pt is human-invisible) ─────────
                    if font_size <= 2.5 and glyph.strip():
                        reasons.append(
                            f"Micro-font ({font_size:.1f}pt) on Page {page_num}"
                        )

        if hidden_snippets:
            preview = "".join(hidden_snippets)[:120]
            reasons.append(f"White/invisible text detected: '{preview}...'")

    except Exception as e:
        print(f"[ERROR] ATS fraud check error for {filename}: {e}")

    # deduplicate reasons (same micro-font size can appear many times)
    return {
        "fraud_detected": len(reasons) > 0,
        "reasons": list(set(reasons)),
        "hidden_text_snippets": hidden_snippets,
    }


# ── Contact info extraction ───────────────────────────────────────────────────

def extract_contact_info(text: str) -> dict:
    """
    Pull email, phone, LinkedIn, and GitHub URLs from raw resume text using regex.

    Phone regex is intentionally broad to catch Indian (10-digit), US (+1 xxx),
    and international formats. We then filter by digit count >= 10.
    """
    # email
    emails = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    email  = emails[0] if emails else "N/A"

    # phone — matches various separators: spaces, dots, dashes, parentheses
    raw_phones   = re.findall(r"(?:\+?\d{1,3}[\-.\s]?)?\(?\d{3,5}\)?[\-.\s]?\d{3,5}[\-.\s]?\d{3,5}", text)
    valid_phones = [p for p in raw_phones if len(re.sub(r"\D", "", p)) >= 10]
    phone        = valid_phones[0] if valid_phones else "N/A"

    # LinkedIn — handles with or without https://
    linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_\-]+", text, re.I)
    linkedin       = linkedin_match.group(0) if linkedin_match else "N/A"

    # GitHub — same pattern
    github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_\-]+", text, re.I)
    github       = github_match.group(0) if github_match else "N/A"

    return {
        "email":    email,
        "phone":    phone,
        "linkedin": linkedin,
        "github":   github,
    }


# ── Education detection ───────────────────────────────────────────────────────

def extract_education(text: str) -> str:
    """
    Detect the highest education degree mentioned in the resume.
    Ordered from highest to lowest so first match wins.
    """
    text_lower = text.lower()

    degree_patterns = [
        ("Ph.D / Doctorate", [
            r"\bph\.?d\b", r"\bdoctorate\b", r"\bdoctor of philosophy\b"
        ]),
        ("Master (M.Tech / M.S / MCA / MBA)", [
            r"\bm\.?tech\b", r"\bm\.?s\.?\b", r"\bmca\b", r"\bmba\b",
            r"\bmaster\b", r"\bpost.?graduate\b"
        ]),
        ("Bachelor (B.Tech / B.E / B.Sc / BCA)", [
            r"\bb\.?tech\b", r"\bb\.?e\.?\b", r"\bbca\b", r"\bb\.?sc\b",
            r"\bbachelor\b", r"\bundergraduate\b"
        ]),
        ("Diploma / High School", [
            r"\bdiploma\b", r"\bhigh school\b", r"\b12th\b", r"\bsecondary\b"
        ]),
    ]

    for degree_label, patterns in degree_patterns:
        for pat in patterns:
            if re.search(pat, text_lower):
                return degree_label

    return "Not Specified"


# ── Experience years extraction ───────────────────────────────────────────────

def extract_experience_years(text: str) -> float:
    """
    Estimate total work experience in years from the resume text.

    Two strategies:
      1. Direct mention: "3+ years of experience" → parse the number
      2. Date range:  "2021 - 2024" or "2022 - Present" → compute duration

    Returns 1.0 as a safe default if neither strategy finds anything.
    """
    text_lower = text.lower()

    # ── Strategy 1: explicit "X years" mentions ───────────────────────────────
    direct_pattern = r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience"
    direct_matches = re.findall(direct_pattern, text_lower)
    if direct_matches:
        try:
            return max(float(m) for m in direct_matches)
        except ValueError:
            pass

    # ── Strategy 2: job tenure date ranges ───────────────────────────────────
    # handles "2020 - 2023", "2019 to Present", "2022–Current"
    range_pattern = r"(20\d{2})\s*(?:[-\u2013\u2014]|to)\s*(20\d{2}|present|current)"
    range_matches = re.findall(range_pattern, text_lower)

    current_year  = 2026
    total_years   = 0.0

    for start_str, end_str in range_matches:
        try:
            start_yr = int(start_str)
            end_yr   = current_year if end_str in ("present", "current") else int(end_str)
            if end_yr >= start_yr:
                total_years += (end_yr - start_yr)
        except ValueError:
            continue

    if total_years > 0:
        return min(total_years, 30.0)  # cap at 30 to avoid bogus date pairs

    # default — assume at least 1 year (freshers with internship experience)
    return 1.0
