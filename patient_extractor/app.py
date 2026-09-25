"""
Patient Detail Extractor – High-Accuracy Multi-Patient Streamlit App
=====================================================================
- Handles single or multiple patients in one PDF / image
- Hybrid native-text + multi-config OCR + aggressive preprocessing
- Intelligent block splitting for multi-patient documents
- Label-aware + regex + validation for maximum accuracy
- Easy-to-copy tabular output + per-patient cards + JSON/CSV/TSV export
"""

import streamlit as st
import re
import io
import json
import csv
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from copy import deepcopy

import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import numpy as np

# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Patient Detail Extractor – Multi-Patient",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Early check for Tesseract (important on Streamlit Cloud)
def _check_tesseract():
    try:
        ver = pytesseract.get_tesseract_version()
        return True, str(ver)
    except Exception as e:
        return False, str(e)

_TESS_OK, _TESS_MSG = _check_tesseract()


# -----------------------------------------------------------------------------
# Field definitions (OCR-tolerant label variants)
# -----------------------------------------------------------------------------
FIELD_LABELS: Dict[str, List[str]] = {
    "patient_name": [
        r"pt'?s?\s*name", r"patient\s*(?:'?s?\s*)?name", r"name\s*of\s*patient",
        r"patient'?s?\s*full\s*name", r"full\s*name", r"^name\s*[:\-]",
        r"pt\s*name", r"client\s*name", r"member\s*name", r"subscriber\s*name",
    ],
    "phone": [
        r"pt'?s?\s*phone", r"patient\s*phone", r"phone\s*(?:number|#)?",
        r"telephone", r"mobile", r"cell\s*phone", r"contact\s*(?:number|phone)",
        r"ph\s*[:\-#]", r"tel\s*[:\-]", r"home\s*phone", r"work\s*phone",
    ],
    "dob": [
        r"pt'?s?\s*dob", r"date\s*of\s*birth", r"d\.?o\.?b\.?", r"birth\s*date",
        r"born\s*on", r"dob\s*[:\-]", r"birthdate", r"date\s*born",
    ],
    "email": [
        r"pt'?s?\s*email", r"patient\s*email", r"e-?mail", r"email\s*address",
        r"email\s*[:\-]", r"e\s*mail",
    ],
    "diagnosis": [
        r"pt'?s?\s*diagnosis", r"diagnosis", r"primary\s*diagnosis",
        r"dx\s*[:\-]", r"clinical\s*diagnosis", r"impression",
        r"assessment", r"condition", r"principal\s*diagnosis",
        r"working\s*diagnosis", r"final\s*diagnosis",
    ],
    "referrer": [
        r"referrer", r"referring\s*(?:physician|doctor|provider|md|dr|clinician)",
        r"referred\s*by", r"referral\s*(?:from|source)", r"ordering\s*provider",
        r"referring\s*md", r"referring\s*doctor", r"ordered\s*by",
    ],
    "address": [
        r"address", r"pt'?s?\s*address", r"patient\s*address", r"home\s*address",
        r"residential\s*address", r"street\s*address", r"mailing\s*address",
        r"physical\s*address",
    ],
    "insurance": [
        r"insurance", r"insurer", r"insurance\s*(?:company|provider|carrier|plan)",
        r"primary\s*insurance", r"health\s*plan", r"payer", r"policy\s*holder",
        r"coverage", r"ins\s*\.?\s*co",
    ],
    "height": [
        r"height", r"ht\s*[:\-]", r"ht\.", r"height\s*\(?(?:cm|in|ft|inches)?\)?",
        r"ht\s*\(",
    ],
    "weight": [
        r"weight", r"wt\s*[:\-]", r"wt\.", r"weight\s*\(?(?:kg|lbs|lb|pounds)?\)?",
        r"wt\s*\(",
    ],
    "bmi": [
        r"bmi", r"body\s*mass\s*index", r"bmi\s*[:\-]", r"body\s*mass",
    ],
}

# Structured value patterns
PHONE_RE = re.compile(
    r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    r"|\d{3}[-.\s]\d{3}[-.\s]\d{4}"
)
EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)
DATE_RE = re.compile(
    r"(?:\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
    r"|(?:\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})"
    r"|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})"
    r"|(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})",
    re.IGNORECASE,
)
HEIGHT_RE = re.compile(
    r"(\d{1,2}(?:\.\d+)?)\s*(?:'|′|ft|feet|foot)\s*(?:(\d{1,2}(?:\.\d+)?)\s*(?:\"|″|in|inches|inch))?"
    r"|(\d{1,3}(?:\.\d+)?)\s*(?:cm|centimeters?)"
    r"|(\d{1,2}(?:\.\d+)?)\s*(?:in|inches|inch|\"|″)",
    re.IGNORECASE,
)
WEIGHT_RE = re.compile(
    r"(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs|kilograms?)"
    r"|(\d{2,3}(?:\.\d+)?)\s*(?:lbs?|pounds?|lb)",
    re.IGNORECASE,
)
BMI_RE = re.compile(
    r"(?:bmi|body\s*mass\s*index)\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

# Strong patient-start signals
PATIENT_START_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*(?:patient|pt|member|client)\s*(?:name|#|id|no)?\s*[:\-]",
        r"^\s*name\s*[:\-]",
        r"^\s*patient\s*information",
        r"^\s*patient\s*demographics",
        r"^\s*new\s*patient",
        r"^\s*patient\s*#?\s*\d+",
        r"^\s*chart\s*(?:#|no|number)",
    ]
]

DISPLAY_ORDER = [
    "patient_name", "phone", "dob", "email", "diagnosis",
    "referrer", "address", "insurance", "height", "weight", "bmi",
]

DISPLAY_LABELS = {
    "patient_name": "PT's Name",
    "phone": "PT's Phone Number",
    "dob": "PT's DOB",
    "email": "PT's Email",
    "diagnosis": "PT's Diagnosis",
    "referrer": "Referrer",
    "address": "Address",
    "insurance": "Insurance",
    "height": "Height",
    "weight": "Weight",
    "bmi": "BMI",
}


# -----------------------------------------------------------------------------
# Image Preprocessing
# -----------------------------------------------------------------------------
def preprocess_image(img: Image.Image) -> Image.Image:
    """Aggressive but safe preprocessing for maximum OCR accuracy."""
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    if img.mode != "L":
        img = img.convert("L")

    # Upscale small images (helps OCR)
    w, h = img.size
    if max(w, h) < 1200:
        scale = 1500 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.7)

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)

    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.autocontrast(img, cutoff=1.5)

    return img


def ocr_image(img: Image.Image, lang: str = "eng") -> str:
    """Run multiple Tesseract configs and keep the best result."""
    pre = preprocess_image(img)

    configs = [
        "--oem 3 --psm 6",   # uniform block
        "--oem 3 --psm 4",   # single column
        "--oem 3 --psm 3",   # fully automatic
        "--oem 3 --psm 11",  # sparse text
        "--oem 3 --psm 1",   # automatic with OSD
    ]

    best_text, best_score = "", -1

    for cfg in configs:
        try:
            text = pytesseract.image_to_string(pre, lang=lang, config=cfg)
            score = len(re.findall(r"[A-Za-z0-9]", text))
            # Bonus for medical keywords
            lower = text.lower()
            for kw in ("patient", "name", "dob", "phone", "diagnosis", "referr", "insurance", "height", "weight"):
                if kw in lower:
                    score += 40
            if score > best_score:
                best_score = score
                best_text = text
        except Exception:
            continue

    # Fallback on original if preprocessed is weak
    if best_score < 100:
        try:
            text = pytesseract.image_to_string(img, lang=lang, config="--oem 3 --psm 6")
            score = len(re.findall(r"[A-Za-z0-9]", text))
            if score > best_score:
                best_text = text
        except Exception:
            pass

    return best_text.strip()


# -----------------------------------------------------------------------------
# PDF / Image → Text
# -----------------------------------------------------------------------------
def _render_pdf_pages_pypdfium2(file_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    """Reliable pure-Python PDF → PIL images via pypdfium2 (no poppler needed)."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(file_bytes)
    scale = dpi / 72.0
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        images.append(pil_image)
        page.close()
    pdf.close()
    return images


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, List[Image.Image], List[str]]:
    """
    Returns (full_text, page_images, page_texts)
    page_texts keeps per-page text for better multi-patient splitting.
    Robust: tries pdfplumber → pdf2image → pypdfium2 fallback.
    """
    page_texts: List[str] = []
    images: List[Image.Image] = []

    # 1. Native text layer via pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                page_texts.append(page_text)
                try:
                    im = page.to_image(resolution=200).original
                    images.append(im)
                except Exception:
                    pass
    except Exception as e:
        st.warning(f"pdfplumber text extraction issue: {e}")

    full_text = "\n\n".join(page_texts).strip()
    alpha = len(re.findall(r"[A-Za-z0-9]", full_text))

    # 2. If sparse text → treat as scanned and OCR
    if alpha < 150:
        st.info("📄 Scanned / image-based PDF detected → running high-quality OCR…")
        ocr_images: List[Image.Image] = []

        # Try A: pdf2image (poppler)
        try:
            ocr_images = convert_from_bytes(file_bytes, dpi=300)
        except Exception as e1:
            st.warning(f"pdf2image/poppler path failed ({e1}). Trying alternative renderer…")
            # Try B: pypdfium2 (pure Python, very reliable)
            try:
                ocr_images = _render_pdf_pages_pypdfium2(file_bytes, dpi=300)
            except Exception as e2:
                st.error(f"All PDF→image methods failed.\n• pdf2image: {e1}\n• pypdfium2: {e2}")
                # Last resort: keep whatever images pdfplumber managed to give us
                ocr_images = images

        if ocr_images:
            images = ocr_images
            page_texts = []
            progress = st.progress(0.0, text="OCR in progress…")
            for i, img in enumerate(ocr_images):
                page_texts.append(ocr_image(img))
                progress.progress((i + 1) / len(ocr_images), text=f"OCR page {i+1}/{len(ocr_images)}")
            progress.empty()
            full_text = "\n\n".join(page_texts)

    return full_text, images, page_texts


def extract_text_from_image(file_bytes: bytes) -> Tuple[str, List[Image.Image], List[str]]:
    img = Image.open(io.BytesIO(file_bytes))
    texts = []
    images = []

    if hasattr(img, "n_frames") and img.n_frames > 1:
        for i in range(img.n_frames):
            img.seek(i)
            frame = img.copy()
            images.append(frame)
            texts.append(ocr_image(frame))
    else:
        images.append(img)
        texts.append(ocr_image(img))

    return "\n\n".join(texts), images, texts


# -----------------------------------------------------------------------------
# Text cleaning & utilities
# -----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    replacements = {
        "|": "I", "—": "-", "–": "-", "‘": "'", "’": "'",
        "“": '"', "”": '"', "´": "'", "`": "'",
    }
    for o, n in replacements.items():
        text = text.replace(o, n)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def calculate_bmi(height_str: str, weight_str: str) -> Optional[str]:
    if not height_str or not weight_str:
        return None
    h = height_str.lower()
    w = weight_str.lower()
    meters = None
    kg = None

    try:
        if "cm" in h:
            cm = float(re.search(r"[\d.]+", h).group())
            meters = cm / 100.0
        elif "'" in h or "ft" in h or "′" in h:
            feet_m = re.search(r"(\d+(?:\.\d+)?)\s*['′ft]", h)
            inch_m = re.search(r"(\d+(?:\.\d+)?)\s*[\"″in]", h)
            feet = float(feet_m.group(1)) if feet_m else 0.0
            inches = float(inch_m.group(1)) if inch_m else 0.0
            meters = (feet * 12 + inches) * 0.0254
        elif "in" in h or '"' in h or "″" in h:
            inches = float(re.search(r"[\d.]+", h).group())
            meters = inches * 0.0254

        if "kg" in w:
            kg = float(re.search(r"[\d.]+", w).group())
        elif "lb" in w or "pound" in w:
            lbs = float(re.search(r"[\d.]+", w).group())
            kg = lbs * 0.45359237

        if meters and kg and meters > 0.5:
            bmi = kg / (meters ** 2)
            if 10 < bmi < 70:
                return f"{bmi:.1f}"
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# Core field extraction (single patient block)
# -----------------------------------------------------------------------------
def find_value_near_label(text: str, label_patterns: List[str], max_chars: int = 140) -> Optional[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for pat in label_patterns:
            m = re.search(pat, line, re.IGNORECASE)
            if not m:
                continue
            after = line[m.end():].strip(" :-–—\t")
            if after and len(after) > 1:
                # Stop before next obvious label
                after = re.split(
                    r"\s{2,}|\t|(?=\b(?:Name|Phone|DOB|Email|Diagnosis|Address|Insurance|Height|Weight|BMI|Referr)\b)",
                    after, maxsplit=1
                )[0]
                candidate = after.strip(" :-.,;")
                if candidate and not re.match(r"^[:\-\s]*$", candidate):
                    return candidate[:max_chars]

            # Next 1–3 lines
            for j in range(1, 4):
                if i + j >= len(lines):
                    break
                nxt = lines[i + j].strip()
                if not nxt:
                    continue
                # Skip if it looks like another label
                if any(re.search(p, nxt, re.IGNORECASE) for p in sum(FIELD_LABELS.values(), [])):
                    break
                return nxt[:max_chars]
    return None


def extract_fields_from_block(text: str) -> Dict[str, Optional[str]]:
    """High-accuracy extraction from one patient text block."""
    text = normalize_text(text)
    results: Dict[str, Optional[str]] = {k: None for k in FIELD_LABELS}

    # ---- Name ----
    name = find_value_near_label(text, FIELD_LABELS["patient_name"], max_chars=80)
    if name:
        name = re.sub(r"^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Miss|Prof\.?)\s+", "", name, flags=re.I)
        name = re.sub(r"[^\w\s\-\.'']", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        # Validation: 1–6 tokens, mostly letters, reasonable length
        tokens = name.split()
        if 1 <= len(tokens) <= 6 and len(name) >= 3 and sum(c.isalpha() for c in name) / max(len(name), 1) > 0.6:
            results["patient_name"] = name.title()

    # ---- Phone ----
    near = find_value_near_label(text, FIELD_LABELS["phone"], max_chars=40)
    m = PHONE_RE.search(near or text)
    if m:
        digits = re.sub(r"\D", "", m.group())
        if len(digits) == 10:
            results["phone"] = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits.startswith("1"):
            results["phone"] = f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            results["phone"] = m.group().strip()

    # ---- DOB ----
    near = find_value_near_label(text, FIELD_LABELS["dob"], max_chars=35)
    m = DATE_RE.search(near or text)
    if m:
        results["dob"] = m.group().strip()

    # ---- Email ----
    near = find_value_near_label(text, FIELD_LABELS["email"], max_chars=70)
    m = EMAIL_RE.search(near or text)
    if m:
        results["email"] = m.group().lower().strip()

    # ---- Diagnosis ----
    diag = find_value_near_label(text, FIELD_LABELS["diagnosis"], max_chars=220)
    if diag:
        diag = re.split(r"\n{2,}|(?=\b(?:Referrer|Address|Insurance|Height|Weight|BMI)\b)", diag)[0]
        results["diagnosis"] = diag.strip()[:220]

    # ---- Referrer ----
    ref = find_value_near_label(text, FIELD_LABELS["referrer"], max_chars=90)
    if ref:
        results["referrer"] = ref.strip()[:90]

    # ---- Address ----
    addr = find_value_near_label(text, FIELD_LABELS["address"], max_chars=160)
    if addr:
        results["address"] = addr.strip()[:160]

    # ---- Insurance ----
    ins = find_value_near_label(text, FIELD_LABELS["insurance"], max_chars=90)
    if ins:
        results["insurance"] = ins.strip()[:90]

    # ---- Height ----
    near = find_value_near_label(text, FIELD_LABELS["height"], max_chars=35)
    m = HEIGHT_RE.search(near or text)
    if m:
        g = m.groups()
        if g[0]:
            feet, inches = g[0], g[1] or "0"
            results["height"] = f"{feet}'{inches}\""
        elif g[2]:
            results["height"] = f"{g[2]} cm"
        elif g[3]:
            results["height"] = f'{g[3]}"'

    # ---- Weight ----
    near = find_value_near_label(text, FIELD_LABELS["weight"], max_chars=35)
    m = WEIGHT_RE.search(near or text)
    if m:
        if m.group(1):
            results["weight"] = f"{m.group(1)} kg"
        elif m.group(2):
            results["weight"] = f"{m.group(2)} lbs"

    # ---- BMI ----
    near = find_value_near_label(text, FIELD_LABELS["bmi"], max_chars=20)
    # Prefer value near an explicit BMI label; avoid matching random numbers
    search_space = near if near else text
    m = BMI_RE.search(search_space)
    if m:
        try:
            val = float(m.group(1))
            if 10 < val < 70:
                results["bmi"] = m.group(1)
        except ValueError:
            pass

    # Auto-calculate BMI if missing
    if not results["bmi"] and results["height"] and results["weight"]:
        calc = calculate_bmi(results["height"], results["weight"])
        if calc:
            results["bmi"] = calc

    return results


# -----------------------------------------------------------------------------
# Multi-patient block detection (the intelligent part)
# -----------------------------------------------------------------------------
def split_into_patient_blocks(full_text: str, page_texts: List[str]) -> List[str]:
    """
    Intelligently split a multi-patient document into individual patient blocks.
    Strategies (in order of preference):
    1. Explicit patient-start markers (Name:, Patient Information, etc.)
    2. Page-level separation when each page looks like one patient
    3. Fallback: treat entire document as one patient
    """
    text = normalize_text(full_text)
    lines = text.splitlines()

    # Strategy 1 – find strong start markers
    start_indices = []
    for i, line in enumerate(lines):
        for pat in PATIENT_START_PATTERNS:
            if pat.search(line):
                start_indices.append(i)
                break

    # Deduplicate close starts
    filtered_starts = []
    for idx in start_indices:
        if not filtered_starts or idx - filtered_starts[-1] > 4:
            filtered_starts.append(idx)

    if len(filtered_starts) >= 2:
        blocks = []
        for k, start in enumerate(filtered_starts):
            end = filtered_starts[k + 1] if k + 1 < len(filtered_starts) else len(lines)
            block = "\n".join(lines[start:end]).strip()
            if len(re.findall(r"[A-Za-z0-9]", block)) > 40:
                blocks.append(block)
        if blocks:
            return blocks

    # Strategy 2 – page-based (common for multi-patient packets)
    if len(page_texts) >= 2:
        page_blocks = []
        for pt in page_texts:
            cleaned = normalize_text(pt)
            if len(re.findall(r"[A-Za-z0-9]", cleaned)) > 60:
                # Does this page contain a name-like field?
                if any(re.search(p, cleaned, re.I) for p in FIELD_LABELS["patient_name"]):
                    page_blocks.append(cleaned)
        if len(page_blocks) >= 2:
            return page_blocks

    # Strategy 3 – single patient
    return [text] if text.strip() else []


def extract_all_patients(full_text: str, page_texts: List[str]) -> List[Dict[str, Optional[str]]]:
    blocks = split_into_patient_blocks(full_text, page_texts)
    patients = []
    for block in blocks:
        fields = extract_fields_from_block(block)
        # Only keep blocks that produced at least a name or a couple of key fields
        filled = sum(1 for v in fields.values() if v)
        if fields.get("patient_name") or filled >= 3:
            patients.append(fields)
    # Final fallback
    if not patients and full_text.strip():
        patients.append(extract_fields_from_block(full_text))
    return patients


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def patient_to_display_dict(p: Dict[str, Optional[str]], idx: int) -> Dict[str, Any]:
    d = {"#": idx + 1}
    for k in DISPLAY_ORDER:
        d[DISPLAY_LABELS[k]] = p.get(k) or ""
    return d


def main():
    st.title("🏥 Patient Detail Extractor")
    st.markdown(
        """
        **Ultra-accurate multi-patient extraction** from PDFs & images.  
        Handles one or many patients in a single file. All processing is **local**.
        """
    )

    if not _TESS_OK:
        st.error(
            f"**Tesseract OCR is not available** on this server.\n\n"
            f"Error: `{_TESS_MSG}`\n\n"
            "On **Streamlit Community Cloud** make sure you have a `packages.txt` file "
            "in the repository root containing:\n"
            "```\ntesseract-ocr\ntesseract-ocr-eng\npoppler-utils\n```\n"
            "Then reboot the app. Text-based (non-scanned) PDFs will still work."
        )

    with st.sidebar:
        st.header("⚙️ Options")
        show_raw = st.checkbox("Show raw OCR / text", value=False)
        show_images = st.checkbox("Show page previews", value=False)
        force_single = st.checkbox("Force single-patient mode", value=False,
                                   help="Disable multi-patient splitting")
        st.markdown("---")
        st.markdown(
            """
            **Supported**  
            PDF · PNG · JPG · TIFF · BMP

            **Best accuracy**  
            • ≥ 200–300 DPI scans  
            • Clear printed text  
            • Good contrast  

            Always review results — medical data is critical.
            """
        )
        st.caption("Privacy: nothing leaves your machine.")

    uploaded = st.file_uploader(
        "Upload PDF or Image (supports multiple patients)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"],
        accept_multiple_files=False,
    )

    if not uploaded:
        st.info("👆 Upload a document to begin.")
        st.markdown(
            """
            ### What this app extracts
            | Field | Field | Field |
            |-------|-------|-------|
            | PT's Name | PT's Phone | PT's DOB |
            | PT's Email | PT's Diagnosis | Referrer |
            | Address | Insurance | Height / Weight / BMI |
            """
        )
        return

    file_bytes = uploaded.read()
    is_pdf = (uploaded.type == "application/pdf") or uploaded.name.lower().endswith(".pdf")

    with st.spinner("Extracting text (OCR if needed)…"):
        if is_pdf:
            full_text, page_images, page_texts = extract_text_from_pdf(file_bytes)
        else:
            full_text, page_images, page_texts = extract_text_from_image(file_bytes)

    if not full_text or len(re.findall(r"[A-Za-z0-9]", full_text)) < 25:
        st.error("No meaningful text could be extracted. Try a clearer scan.")
        if show_raw:
            st.text_area("Raw text", full_text or "(empty)", height=200)
        return

    # Extract patients
    with st.spinner("Intelligently detecting patients & extracting fields…"):
        if force_single:
            patients = [extract_fields_from_block(full_text)]
        else:
            patients = extract_all_patients(full_text, page_texts)

    n = len(patients)
    if n == 0:
        st.warning("No patient records detected.")
        return

    st.success(f"✅ Detected **{n}** patient record{'s' if n != 1 else ''}")

    # ------------------------------------------------------------------
    # Easy-to-copy main table
    # ------------------------------------------------------------------
    st.subheader("📋 Extracted Data (easy to select & copy)")

    display_rows = [patient_to_display_dict(p, i) for i, p in enumerate(patients)]
    df_data = display_rows

    # Streamlit dataframe is highly copy-friendly (select cells → Ctrl+C)
    st.dataframe(
        df_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
        },
    )

    st.caption("💡 Tip: Click any cell and drag to select → Ctrl/Cmd + C to copy. Or use the download buttons below.")

    # ------------------------------------------------------------------
    # Per-patient editable cards (for correction)
    # ------------------------------------------------------------------
    st.subheader("✏️ Review & Correct (optional)")

    edited_patients = []
    for i, p in enumerate(patients):
        with st.expander(f"Patient #{i+1}: {p.get('patient_name') or 'Unknown'}", expanded=(n <= 2)):
            cols = st.columns(2)
            edited = {}
            with cols[0]:
                edited["patient_name"] = st.text_input("PT's Name", value=p.get("patient_name") or "", key=f"name_{i}")
                edited["phone"] = st.text_input("PT's Phone Number", value=p.get("phone") or "", key=f"phone_{i}")
                edited["dob"] = st.text_input("PT's DOB", value=p.get("dob") or "", key=f"dob_{i}")
                edited["email"] = st.text_input("PT's Email", value=p.get("email") or "", key=f"email_{i}")
                edited["diagnosis"] = st.text_area("PT's Diagnosis", value=p.get("diagnosis") or "", height=70, key=f"dx_{i}")
                edited["referrer"] = st.text_input("Referrer", value=p.get("referrer") or "", key=f"ref_{i}")
            with cols[1]:
                edited["address"] = st.text_area("Address", value=p.get("address") or "", height=70, key=f"addr_{i}")
                edited["insurance"] = st.text_input("Insurance", value=p.get("insurance") or "", key=f"ins_{i}")
                edited["height"] = st.text_input("Height", value=p.get("height") or "", key=f"ht_{i}")
                edited["weight"] = st.text_input("Weight", value=p.get("weight") or "", key=f"wt_{i}")
                edited["bmi"] = st.text_input("BMI", value=p.get("bmi") or "", key=f"bmi_{i}")

                if st.button("🔄 Recalc BMI", key=f"recalc_{i}"):
                    calc = calculate_bmi(edited["height"], edited["weight"])
                    if calc:
                        edited["bmi"] = calc
                        st.success(f"BMI → {calc}")
                    else:
                        st.warning("Could not parse height/weight")

            edited_patients.append(edited)

            # Easy copy for this single patient
            lines = []
            for k in DISPLAY_ORDER:
                label = DISPLAY_LABELS[k]
                val = edited.get(k) or ""
                lines.append(f"{label}: {val}")
            single_text = "\n".join(lines)
            st.code(single_text, language=None)
            st.caption("↑ Select the block above and copy (Ctrl/Cmd+C)")

    # ------------------------------------------------------------------
    # Bulk export – easy copy formats
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📤 Export / Copy All")

    # Prepare clean final data
    final_list = []
    for i, ep in enumerate(edited_patients):
        row = {
            "PT's name": (ep.get("patient_name") or "").strip() or None,
            "PT's phone number": (ep.get("phone") or "").strip() or None,
            "PT's DOB": (ep.get("dob") or "").strip() or None,
            "PT's Email": (ep.get("email") or "").strip() or None,
            "PT's Diagnosis": (ep.get("diagnosis") or "").strip() or None,
            "Referrer": (ep.get("referrer") or "").strip() or None,
            "Address": (ep.get("address") or "").strip() or None,
            "Insurance": (ep.get("insurance") or "").strip() or None,
            "Height": (ep.get("height") or "").strip() or None,
            "Weight": (ep.get("weight") or "").strip() or None,
            "BMI": (ep.get("bmi") or "").strip() or None,
        }
        final_list.append(row)

    # 1. JSON
    json_str = json.dumps(final_list, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Download JSON",
        data=json_str,
        file_name=f"patients_{Path(uploaded.name).stem}.json",
        mime="application/json",
        use_container_width=True,
    )

    # 2. CSV
    if final_list:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(final_list[0].keys()))
        writer.writeheader()
        writer.writerows(final_list)
        csv_str = output.getvalue()
        st.download_button(
            "⬇️ Download CSV",
            data=csv_str,
            file_name=f"patients_{Path(uploaded.name).stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 3. TSV (perfect for pasting into Excel / Google Sheets)
    if final_list:
        tsv_lines = ["\t".join(final_list[0].keys())]
        for row in final_list:
            tsv_lines.append("\t".join(str(v or "") for v in row.values()))
        tsv_str = "\n".join(tsv_lines)
        st.download_button(
            "⬇️ Download TSV (Excel-friendly)",
            data=tsv_str,
            file_name=f"patients_{Path(uploaded.name).stem}.tsv",
            mime="text/tab-separated-values",
            use_container_width=True,
        )

    # 4. Plain text block – easiest to select & copy everything
    st.markdown("#### 📋 All patients – plain text (select & copy)")
    plain_blocks = []
    for i, row in enumerate(final_list):
        block = [f"=== Patient #{i+1} ==="]
        for k, v in row.items():
            block.append(f"{k}: {v or ''}")
        plain_blocks.append("\n".join(block))
    plain_all = "\n\n".join(plain_blocks)
    st.code(plain_all, language=None)
    st.caption("Select the entire block above → Ctrl/Cmd + C")

    # ------------------------------------------------------------------
    # Debug views
    # ------------------------------------------------------------------
    if show_raw:
        with st.expander("🔍 Raw extracted text", expanded=False):
            st.text_area("Full text", full_text, height=350)

    if show_images and page_images:
        with st.expander("🖼️ Page / Image previews", expanded=False):
            for i, img in enumerate(page_images[:8]):
                st.image(img, caption=f"Page {i+1}", use_container_width=True)


if __name__ == "__main__":
    main()
