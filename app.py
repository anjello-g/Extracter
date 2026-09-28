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
from concurrent.futures import ThreadPoolExecutor
import hashlib

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
        r"patient\s*information", r"^patient\s*[:\-]",
    ],
    "phone": [
        r"pt'?s?\s*phone", r"patient\s*phone", r"phone\s*(?:number|#)?",
        r"telephone", r"mobile", r"cell\s*phone", r"contact\s*(?:number|phone|:)",
        r"ph\s*[:\-#]", r"tel\s*[:\-]", r"home\s*phone", r"work\s*phone",
        r"contact\s*[:\-]",
    ],
    "dob": [
        r"pt'?s?\s*dob", r"date\s*of\s*birth", r"d\.?o\.?b\.?", r"birth\s*date",
        r"born\s*on", r"dob\s*[:\-]", r"birthdate", r"date\s*born",
        r"date\s*of\s*birth\s*[:\-]",
    ],
    "email": [
        r"pt'?s?\s*email", r"patient\s*email", r"e-?mail", r"email\s*address",
        r"email\s*[:\-]", r"e\s*mail",
    ],
    "diagnosis": [
        r"pt'?s?\s*diagnosis", r"primary\s*/?\s*billing\s*diagnosis",
        r"primary\s*diagnosis", r"billing\s*diagnosis", r"diagnosis",
        r"dx\s*[:\-]", r"clinical\s*diagnosis", r"impression",
        r"assessment", r"condition", r"principal\s*diagnosis",
        r"working\s*diagnosis", r"final\s*diagnosis",
        r"other\s*assessments?\s*at\s*time\s*of\s*order",
    ],
    "referrer": [
        r"referrer", r"referring\s*(?:physician|doctor|provider|md|dr|clinician)",
        r"referred\s*by", r"referral\s*(?:from|source)", r"ordering\s*provider",
        r"referring\s*md", r"referring\s*doctor", r"ordered\s*by",
        r"ordering\s*provider\s*[:\-]", r"pcp\s*[:\-]", r"primary\s*care\s*provider",
    ],
    "address": [
        r"address", r"pt'?s?\s*address", r"patient\s*address", r"home\s*address",
        r"residential\s*address", r"street\s*address", r"mailing\s*address",
        r"physical\s*address",
    ],
    "insurance": [
        r"insurance\s*(?:company|provider|carrier|plan)", r"primary\s*insurance",
        r"insurance\s*[:\-]", r"insurer", r"health\s*plan", r"payer",
        r"policy\s*holder", r"coverage", r"ins\s*\.?\s*co", r"insurance",
    ],
    "height": [
        r"height", r"ht\s*[:\-]", r"ht\.", r"height\s*\(?(?:cm|in|ft|inches)?\)?",
        r"ht\s*\(", r"height\s*/?\s*bsa",
    ],
    "weight": [
        r"weight", r"wt\s*[:\-]", r"wt\.", r"weight\s*\(?(?:kg|lbs|lb|pounds)?\)?",
        r"wt\s*\(", r"weight\s*/?\s*bsa\s*/?\s*bmi",
    ],
    "bmi": [
        r"bmi", r"body\s*mass\s*index", r"bmi\s*[:\-]", r"body\s*mass",
        r"bmi\s*/?\s*kg",
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
    # 5'10" or 5 ft 10 in or 5.0 ft 3.00 in
    r"(\d{1,2}(?:\.\d+)?)\s*(?:'|′|ft|feet|foot)\s*(?:(\d{1,2}(?:\.\d+)?)\s*(?:\"|″|in|inches|inch))?"
    # table style: 5.0   3.00   160.02  (ft  in  cm)
    r"|(\d{1,2}(?:\.\d+)?)\s+(\d{1,2}(?:\.\d+)?)\s+(\d{2,3}(?:\.\d+)?)"
    # pure cm
    r"|(\d{1,3}(?:\.\d+)?)\s*(?:cm|centimeters?)"
    # pure inches
    r"|(\d{1,2}(?:\.\d+)?)\s*(?:in|inches|inch|\"|″)",
    re.IGNORECASE,
)
WEIGHT_RE = re.compile(
    r"(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs|kilograms?)\b"
    r"|(\d{2,3}(?:\.\d+)?)\s*(?:lbs?|pounds?|lb)\b",
    re.IGNORECASE,
)
BMI_RE = re.compile(
    r"(?:bmi|body\s*mass\s*index|bmi\s*/?\s*kg/?m2?)\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?)"
    r"|(?<![A-Za-z])(\d{2}\.\d{1,2})\s*(?:kg/?m2|bmi)?",  # standalone plausible BMI near vitals
    # (?<![A-Za-z]) keeps this from matching the decimal part of an ICD code like "E11.9"
    re.IGNORECASE,
)

# ICD-10-CM style codes (e.g. E11.9, I10, M54.5, Z00.00) and ICD-9-CM style
# codes (e.g. 250.00, 401.9) used to tag diagnoses.
DIAGNOSIS_CODE_RE = re.compile(
    r"\b[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4})?\b"   # ICD-10-CM
    r"|\b[0-9]{3}\.[0-9]{1,2}\b"                      # ICD-9-CM
)

# Labels that precede a member / subscriber / policy identifier
MEMBER_ID_LABELS = [
    r"member\s*id", r"member\s*#", r"member\s*no", r"member\s*number",
    r"subscriber\s*id", r"subscriber\s*#", r"subscriber\s*no", r"subscriber\s*number",
    r"insurance\s*id", r"policy\s*(?:#|no\.?|number)", r"id\s*#",
    r"group\s*(?:#|no\.?|number)",
]
# A plausible member/subscriber/policy ID: alphanumeric, 5-20 chars, at least one digit
MEMBER_ID_VALUE_RE = re.compile(r"\b(?=[A-Za-z0-9\-]{5,20}\b)[A-Za-z0-9\-]*\d[A-Za-z0-9\-]*\b")

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


# Config sets: "fast" tries the two most reliable layouts and stops early
# once it gets a good result; "full" (accuracy mode) tries every layout.
OCR_CONFIGS_FAST = [
    "--oem 3 --psm 6",   # uniform block (best general-purpose default)
    "--oem 3 --psm 4",   # single column
]
OCR_CONFIGS_FULL = OCR_CONFIGS_FAST + [
    "--oem 3 --psm 3",   # fully automatic
    "--oem 3 --psm 11",  # sparse text
    "--oem 3 --psm 1",   # automatic with OSD
]

_MED_KEYWORDS = ("patient", "name", "dob", "phone", "diagnosis", "referr", "insurance", "height", "weight")


def _score_text(text: str) -> int:
    score = len(re.findall(r"[A-Za-z0-9]", text))
    lower = text.lower()
    for kw in _MED_KEYWORDS:
        if kw in lower:
            score += 40
    return score


def ocr_image(img: Image.Image, lang: str = "eng", fast_mode: bool = True, early_exit_score: int = 400) -> str:
    """
    Run Tesseract with one or more configs and keep the best-scoring result.
    fast_mode uses fewer configs and exits as soon as a config scores well
    (usually the first one), which is the biggest lever for OCR speed.
    """
    pre = preprocess_image(img)
    configs = OCR_CONFIGS_FAST if fast_mode else OCR_CONFIGS_FULL

    best_text, best_score = "", -1
    for cfg in configs:
        try:
            text = pytesseract.image_to_string(pre, lang=lang, config=cfg)
            score = _score_text(text)
            if score > best_score:
                best_score = score
                best_text = text
            if best_score >= early_exit_score:
                break  # good enough — skip remaining configs
        except Exception:
            continue

    # Fallback on original (un-preprocessed) image only if everything was weak
    if best_score < 100:
        try:
            text = pytesseract.image_to_string(img, lang=lang, config="--oem 3 --psm 6")
            score = _score_text(text)
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


def _ocr_pages_parallel(ocr_images: List[Image.Image], fast_mode: bool, max_workers: int = 4) -> List[str]:
    """OCR multiple page/frame images concurrently (Tesseract runs as a
    subprocess per call, so threads give real wall-clock speedup here)."""
    if len(ocr_images) <= 1:
        return [ocr_image(im, fast_mode=fast_mode) for im in ocr_images]

    results: List[Optional[str]] = [None] * len(ocr_images)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(ocr_images))) as ex:
        futures = {ex.submit(ocr_image, im, "eng", fast_mode): i for i, im in enumerate(ocr_images)}
        done = 0
        progress = st.progress(0.0, text="OCR in progress…")
        from concurrent.futures import as_completed
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = ""
            done += 1
            progress.progress(done / len(ocr_images), text=f"OCR page {done}/{len(ocr_images)}")
        progress.empty()
    return [r or "" for r in results]


@st.cache_data(show_spinner=False, max_entries=8)
def extract_text_from_pdf(file_bytes: bytes, fast_mode: bool = True, dpi: int = 250) -> Tuple[str, List[Image.Image], List[str]]:
    """
    Returns (full_text, page_images, page_texts)
    page_texts keeps per-page text for better multi-patient splitting.
    Robust: tries pdfplumber → pdf2image → pypdfium2 fallback.

    Cached by (file bytes, fast_mode, dpi) so re-running the Streamlit script
    on every widget interaction does NOT re-OCR the document — this is the
    single biggest speed win, since Streamlit reruns main() on every click.
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
                    im = page.to_image(resolution=150).original
                    images.append(im)
                except Exception:
                    pass
    except Exception as e:
        st.warning(f"pdfplumber text extraction issue: {e}")

    full_text = "\n\n".join(page_texts).strip()
    alpha = len(re.findall(r"[A-Za-z0-9]", full_text))

    # 2. If sparse text → treat as scanned and OCR
    if alpha < 150:
        st.info("📄 Scanned / image-based PDF detected → running OCR…")
        ocr_images: List[Image.Image] = []

        # Try A: pdf2image (poppler)
        try:
            ocr_images = convert_from_bytes(file_bytes, dpi=dpi)
        except Exception as e1:
            st.warning(f"pdf2image/poppler path failed ({e1}). Trying alternative renderer…")
            # Try B: pypdfium2 (pure Python, very reliable)
            try:
                ocr_images = _render_pdf_pages_pypdfium2(file_bytes, dpi=dpi)
            except Exception as e2:
                st.error(f"All PDF→image methods failed.\n• pdf2image: {e1}\n• pypdfium2: {e2}")
                # Last resort: keep whatever images pdfplumber managed to give us
                ocr_images = images

        if ocr_images:
            images = ocr_images
            page_texts = _ocr_pages_parallel(ocr_images, fast_mode=fast_mode)
            full_text = "\n\n".join(page_texts)

    return full_text, images, page_texts


@st.cache_data(show_spinner=False, max_entries=8)
def extract_text_from_image(file_bytes: bytes, fast_mode: bool = True) -> Tuple[str, List[Image.Image], List[str]]:
    img = Image.open(io.BytesIO(file_bytes))
    frames = []

    if hasattr(img, "n_frames") and img.n_frames > 1:
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(img.copy())
    else:
        frames.append(img)

    texts = _ocr_pages_parallel(frames, fast_mode=fast_mode)
    return "\n\n".join(texts), frames, texts


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


def find_section_block(
    text: str,
    label_patterns: List[str],
    max_lines: int = 12,
    stop_labels: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Capture a *multi-line* block following a label (unlike find_value_near_label,
    which only grabs a single value). Stops at the next recognized field label,
    two consecutive blank lines, or max_lines.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for pat in label_patterns:
            m = re.search(pat, line, re.IGNORECASE)
            if not m:
                continue
            block_lines = []
            after = line[m.end():].strip(" :-–—\t")
            if after:
                block_lines.append(after)
            j = i + 1
            blank_run = 0
            while j < len(lines) and len(block_lines) < max_lines:
                nxt = lines[j].strip()
                if not nxt:
                    blank_run += 1
                    if blank_run >= 2:
                        break
                    j += 1
                    continue
                blank_run = 0
                if stop_labels and any(re.search(p, nxt, re.IGNORECASE) for p in stop_labels):
                    break
                block_lines.append(nxt)
                j += 1
            if block_lines:
                return "\n".join(block_lines)
    return None


def extract_diagnoses(text: str) -> Optional[str]:
    """
    Find every present diagnosis mentioned in/near the diagnosis section and
    return them as 'Description (Code)' entries joined by '; '.
    Falls back gracefully when codes or descriptions are missing.
    """
    # Everything that is NOT a diagnosis label acts as a stop marker for the block
    stop_labels: List[str] = []
    for key, pats in FIELD_LABELS.items():
        if key != "diagnosis":
            stop_labels.extend(pats)

    block = find_section_block(text, FIELD_LABELS["diagnosis"], max_lines=15, stop_labels=stop_labels)
    if not block:
        return None

    # Strip bullet / numbering prefixes so each line becomes a clean entry
    block = re.sub(r"^\s*[\-\u2022\*]\s*", "", block, flags=re.MULTILINE)
    block = re.sub(r"^\s*\d{1,2}[\.\)]\s*", "", block, flags=re.MULTILINE)

    # Diagnoses are typically separated by newlines or semicolons; commas are
    # ambiguous (can appear inside a single diagnosis description) so we don't split on them.
    raw_entries = re.split(r"[\n;]+", block)

    entries: List[str] = []
    seen = set()
    for raw in raw_entries:
        raw = raw.strip(" .,:-")
        if not raw or len(raw) < 2:
            continue
        # Skip stray lines that are actually a different field leaking in
        if any(re.search(p, raw, re.IGNORECASE) for p in stop_labels):
            continue

        code_match = DIAGNOSIS_CODE_RE.search(raw)
        code = code_match.group(0).upper() if code_match else None
        desc = DIAGNOSIS_CODE_RE.sub("", raw).strip(" -–—:(),")
        desc = re.sub(r"\s{2,}", " ", desc)

        if code and desc:
            entry = f"{desc} ({code})"
        elif code:
            entry = code
        elif desc and len(desc) > 2:
            entry = desc
        else:
            continue

        key = entry.lower()
        if key not in seen:
            seen.add(key)
            entries.append(entry)

    if not entries:
        return None
    return "; ".join(entries[:15])


def extract_insurance_with_id(text: str) -> Optional[str]:
    """
    Return 'Insurance Name - Member/Subscriber ID' when both are found,
    or just whichever piece is available.
    """
    name = None
    ins = find_value_near_label(text, FIELD_LABELS["insurance"], max_chars=90)
    if ins:
        ins = re.sub(r"^[/\\|]+\s*", "", ins)
        ins = re.sub(r"\b(?:Authorization|Information|Policy\s*#?|Provider).*$", "", ins, flags=re.I)
        ins = ins.strip(" :/-")
        if ins and len(ins) > 2 and not re.match(r"^(Information|Authorization)$", ins, re.I):
            name = ins[:90]

    if not name:
        payer_re = re.compile(
            r"(?:Insurance|Insurer|Payer|Plan)\s*[:\-]?\s*((?:BCBS|Blue\s*Cross|Aetna|United|Cigna|Medicare|Medicaid|Humana|Tricare|Kaiser|Anthem|Federal)[^\n]{0,40})",
            re.IGNORECASE,
        )
        m = payer_re.search(text)
        if m:
            name = m.group(1).strip()[:90]

    member_id = None
    id_near = find_value_near_label(text, MEMBER_ID_LABELS, max_chars=40)
    if id_near:
        m = MEMBER_ID_VALUE_RE.search(id_near)
        if m:
            member_id = m.group(0).strip(" -")

    if name and member_id:
        return f"{name} - {member_id}"
    elif name:
        return name
    elif member_id:
        return member_id
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

    # ---- Diagnosis (all present diagnoses + codes, separated by "; ") ----
    multi_diag = extract_diagnoses(text)
    if multi_diag:
        results["diagnosis"] = multi_diag[:500]
    else:
        # Fallback to the original single-value grab if no section block was found
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
    else:
        # Common pattern: name followed by street / city / state zip (no "Address:" label)
        # Look for a line that looks like a street address near the patient name
        street_re = re.compile(
            r"^\s*\d{1,6}\s+[A-Za-z0-9\.\-'\s]+(?:St|Street|Rd|Road|Ave|Avenue|Blvd|Lane|Ln|Dr|Drive|Ct|Court|Way|Pl|Place)\b.*",
            re.IGNORECASE,
        )
        for line in text.splitlines():
            if street_re.match(line.strip()):
                # grab this line + possible next city/state line
                idx = text.splitlines().index(line)
                parts = [line.strip()]
                lines = text.splitlines()
                if idx + 1 < len(lines):
                    nxt = lines[idx + 1].strip()
                    if re.search(r"[A-Za-z]+,?\s*[A-Z]{2}\s*\d{5}", nxt) or re.search(r"\b[A-Z]{2}\s*\d{5}", nxt):
                        parts.append(nxt)
                results["address"] = ", ".join(parts)[:160]
                break

    # ---- Insurance (Name - Member/Subscriber ID) ----
    ins_combined = extract_insurance_with_id(text)
    if ins_combined:
        results["insurance"] = ins_combined[:140]

    # ---- Height ----
    # Prefer context near "Height" label; also scan whole text for vital tables
    near = find_value_near_label(text, FIELD_LABELS["height"], max_chars=80)
    search_h = near or text
    m = HEIGHT_RE.search(search_h)
    if m:
        g = m.groups()
        # groups roughly: (ft_classic, in_classic, ft_table, in_table, cm_table, pure_cm, pure_in)
        if g[0]:  # classic 5'10" or 5 ft 10
            feet, inches = g[0], g[1] or "0"
            results["height"] = f"{feet}'{inches}\""
        elif g[2] is not None and g[3] is not None:  # table 5.0  3.00  160
            feet, inches = g[2], g[3]
            results["height"] = f"{feet}'{inches}\""
        elif g[4]:  # pure cm
            results["height"] = f"{g[4]} cm"
        elif g[5]:  # pure inches
            results["height"] = f'{g[5]}"'

    # Extra vital-table pattern: "5.0  3.00  160.02"
    if not results["height"]:
        vitals_h = re.search(
            r"(?:Height|Ht).*?(?:ft|in|cm)?.*?(\d{1,2}(?:\.\d+)?)\s+(\d{1,2}(?:\.\d+)?)\s+(\d{2,3}(?:\.\d+)?)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if vitals_h:
            results["height"] = f"{vitals_h.group(1)}'{vitals_h.group(2)}\""

    # ---- Weight ----
    near = find_value_near_label(text, FIELD_LABELS["weight"], max_chars=80)
    search_w = near or text
    m = WEIGHT_RE.search(search_w)
    if m:
        if m.group(1):
            results["weight"] = f"{m.group(1)} kg"
        elif m.group(2):
            results["weight"] = f"{m.group(2)} lbs"
        elif m.group(3):
            results["weight"] = f"{m.group(3)} lbs"

    # Extra vital-table pattern: look for plausible adult weight near an actual
    # Weight header. Unlike before, we do NOT fall back to scanning the whole
    # document — that produced false positives on unrelated numbers (IDs, phone
    # digits, etc.) whenever no explicit "Weight:" label existed.
    if not results["weight"]:
        section = re.search(
            r"(?:Weight|Wt|Weight/BSA|Weight\s*/\s*BMI)\b.{0,200}",
            text, re.IGNORECASE | re.DOTALL,
        )
        if section:
            candidates = []
            for m in re.finditer(r"\b(\d{2,3}(?:\.\d{1,2})?)\b", section.group(0)):
                try:
                    wval = float(m.group(1))
                    if 80 <= wval <= 450:
                        candidates.append((wval, m.group(1)))
                except ValueError:
                    pass
            if candidates:
                # Prefer the first plausible weight (usually the lb value)
                results["weight"] = f"{candidates[0][1]} lbs"

    # ---- BMI ----
    near = find_value_near_label(text, FIELD_LABELS["bmi"], max_chars=40)
    search_space = near or text
    m = BMI_RE.search(search_space)
    if m:
        try:
            val_str = m.group(1) or m.group(2)
            val = float(val_str)
            if 10 < val < 70:
                results["bmi"] = f"{val:.1f}" if "." in str(val_str) else str(val_str)
        except (ValueError, TypeError):
            pass

    # Extra: look for BMI value in vitals section (prefer numbers after BMI header)
    if not results["bmi"]:
        section = re.search(
            r"(?:BMI|bmi\s*kg/?m2?).{0,120}",
            text, re.IGNORECASE | re.DOTALL,
        )
        if section:
            for m in re.finditer(r"\b(\d{2}\.\d{1,2})\b", section.group(0)):
                try:
                    val = float(m.group(1))
                    if 10 < val < 70:
                        results["bmi"] = f"{val:.1f}"
                        break
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
# AI Extraction (free tiers: Groq + Google Gemini)
# -----------------------------------------------------------------------------
AI_SYSTEM_PROMPT = """You are a medical document extraction specialist.
Extract patient information from the OCR text of clinical referral / chart documents.
Return ONLY valid JSON (no markdown, no commentary).

If the document contains multiple distinct patients, return a JSON array of objects.
If only one patient, still return a JSON array with one object.

Each object must use exactly these keys (string or null):
{
  "patient_name": null,
  "phone": null,
  "dob": null,
  "email": null,
  "diagnosis": null,
  "referrer": null,
  "address": null,
  "insurance": null,
  "height": null,
  "weight": null,
  "bmi": null
}

Rules:
- patient_name: full name only (no titles unless part of the name)
- phone: format as (XXX) XXX-XXXX when possible
- dob: keep original format found (MM/DD/YYYY preferred)
- diagnosis: list EVERY diagnosis present in the document (not just the primary one),
  each formatted as "Description (Code)" using its ICD-9/ICD-10 code when one is given,
  or just the description or just the code if only one is present. Separate multiple
  diagnoses with "; ". Example: "Type 2 diabetes mellitus (E11.9); Essential hypertension (I10)"
- referrer: ordering provider, referring physician, or PCP name
- address: full street address
- insurance: format as "Insurance Name - Member ID" (use the subscriber ID if no
  member ID is given). If only the name or only the ID is present, return just that.
  Example: "Blue Cross Blue Shield - ABC123456789"
- height: prefer ft'in" (e.g. 5'3") or cm
- weight: include unit (lbs or kg)
- bmi: numeric value only
- Use null when a field is truly absent
- Never invent data that is not present in the text
"""

def _normalize_ai_patient(obj: dict) -> Dict[str, Optional[str]]:
    """Map AI JSON keys into our internal schema and clean values."""
    key_map = {
        "patient_name": "patient_name",
        "name": "patient_name",
        "pt_name": "patient_name",
        "phone": "phone",
        "phone_number": "phone",
        "dob": "dob",
        "date_of_birth": "dob",
        "email": "email",
        "diagnosis": "diagnosis",
        "referrer": "referrer",
        "referring_provider": "referrer",
        "ordering_provider": "referrer",
        "address": "address",
        "insurance": "insurance",
        "height": "height",
        "weight": "weight",
        "bmi": "bmi",
    }
    out = {k: None for k in FIELD_LABELS}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        canon = key_map.get(str(k).lower().strip())
        if canon and v is not None and str(v).strip().lower() not in ("null", "none", "n/a", ""):
            cap = 400 if canon == "diagnosis" else 220
            out[canon] = str(v).strip()[:cap]
    return out


def extract_with_groq(text: str, api_key: str, model: str = "llama-3.1-8b-instant") -> List[Dict[str, Optional[str]]]:
    """Call Groq free API for structured extraction."""
    try:
        from groq import Groq
    except ImportError:
        st.error("groq package not installed. Add `groq` to requirements.txt")
        return []

    client = Groq(api_key=api_key)
    # Truncate very long OCR to stay within context (keep start + end)
    if len(text) > 28000:
        text = text[:14000] + "\n\n[... middle truncated ...]\n\n" + text[-14000:]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": f"OCR text of the medical document:\n\n{text}"},
        ],
        temperature=0.1,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)

    # Accept either {"patients": [...]} or a bare list or a single object
    if isinstance(data, dict):
        if "patients" in data and isinstance(data["patients"], list):
            items = data["patients"]
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    return [_normalize_ai_patient(item) for item in items if isinstance(item, dict)]


def extract_with_gemini(text: str, api_key: str, model: str = "gemini-3.5-flash-lite") -> List[Dict[str, Optional[str]]]:
    """"""
    try:
        import google.generativeai as genai
    except ImportError:
        st.error("google-generativeai package not installed. Add it to requirements.txt and redeploy.")
        return []

    genai.configure(api_key=api_key)
    if len(text) > 28000:
        text = text[:14000] + "\n\n[... middle truncated ...]\n\n" + text[-14000:]

    # Try current free-tier friendly models in order
    # Flash-Lite models have the highest free daily quota (~500/day); fall through to the next on 429/quota errors
    candidate_models = [model, "gemini-3.1-flash-lite", "gemini-3.8-flash"]
    last_err = None
    raw = None

    for model_name in candidate_models:
        try:
            model_obj = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=AI_SYSTEM_PROMPT,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json",
                },
            )
            response = model_obj.generate_content(
                f"OCR text of the medical document:\n\n{text}"
            )
            raw = response.text
            break
        except Exception as e:
            last_err = e
            continue

    if raw is None:
        raise RuntimeError(f"Gemini models failed. Last error: {last_err}")

    data = json.loads(raw)

    if isinstance(data, dict):
        if "patients" in data and isinstance(data["patients"], list):
            items = data["patients"]
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    return [_normalize_ai_patient(item) for item in items if isinstance(item, dict)]


@st.cache_data(show_spinner=False, max_entries=16)
def _extract_with_ai_cached(full_text: str, provider: str, api_key: str) -> List[Dict[str, Optional[str]]]:
    """Cached so Streamlit reruns (every widget click) don't burn API quota.
    Exceptions are NOT cached, so a failed/quota-limited call can be retried."""
    if provider == "groq":
        return extract_with_groq(full_text, api_key)
    elif provider in ("gemini", "google"):
        return extract_with_gemini(full_text, api_key)
    raise ValueError(f"Unknown AI provider: {provider}")


def extract_with_ai(full_text: str, provider: str, api_key: str) -> List[Dict[str, Optional[str]]]:
    """Dispatch to the selected free AI provider (cached per document)."""
    if not api_key or not api_key.strip():
        return []
    try:
        return _extract_with_ai_cached(full_text, (provider or "").lower(), api_key.strip())
    except Exception as e:
        st.error(f"AI extraction failed ({provider}): {e}")
        return []


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
        **
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
        fast_mode = st.checkbox(
            "⚡ Fast mode", value=True,
            help="Fewer OCR passes + early exit + parallel pages. Turn off for maximum "
                 "accuracy on very poor-quality scans (slower)."
        )

        st.markdown("---")
        st.header("")
        st.caption("")

        # Prefer Streamlit secrets if present (recommended on Cloud)
        # Default provider is Gemini (user preference)
        default_provider = "Gemini"
        default_key = ""
        try:
            if "GEMINI_API_KEY" in st.secrets or "GOOGLE_API_KEY" in st.secrets:
                default_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
                default_provider = "Gemini"
            elif "GROQ_API_KEY" in st.secrets:
                default_key = st.secrets["GROQ_API_KEY"]
                default_provider = "Groq"
        except Exception:
            pass

        ai_provider = st.selectbox(
            "AI Provider",
            options=["Gemini", "Groq", "None (rules only)"],
            index=0 if default_provider == "Gemini" else (1 if default_provider == "Groq" else 2),
            help="Gemini (Google) is recommended. Free API key from Google AI Studio.",
        )
        ai_api_key = st.text_input(
            "Gemini / Groq API Key",
            value=default_key,
            type="password",
            help="Get a free Gemini key: https://aistudio.google.com/apikey",
            placeholder="Paste your free Gemini API key here",
        )
        use_ai = ai_provider != "None (rules only)" and bool(ai_api_key.strip())

        st.markdown("---")
        st.markdown(
            """
            
            """
        )
        st.caption("")

    uploaded = st.file_uploader(
        "Upload PDF or Image (supports multiple patients)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"],
        accept_multiple_files=False,
    )

    if not uploaded:
        st.info("👆 Upload a document to begin.")
        st.markdown(
            """
            
            """
        )
        return

    file_bytes = uploaded.read()
    is_pdf = (uploaded.type == "application/pdf") or uploaded.name.lower().endswith(".pdf")

    with st.spinner("Extracting text"):
        if is_pdf:
            full_text, page_images, page_texts = extract_text_from_pdf(file_bytes, fast_mode=fast_mode)
        else:
            full_text, page_images, page_texts = extract_text_from_image(file_bytes, fast_mode=fast_mode)

    if not full_text or len(re.findall(r"[A-Za-z0-9]", full_text)) < 25:
        st.error("No meaningful text could be extracted. Try a clearer scan.")
        if show_raw:
            st.text_area("Raw text", full_text or "(empty)", height=200)
        return

    # Extract patients — prefer free AI when configured, else rules engine
    patients: List[Dict[str, Optional[str]]] = []
    extraction_method = "rules"

    if use_ai:
        with st.spinner(f"🤖 AI extraction via {ai_provider}…"):
            ai_patients = extract_with_ai(full_text, ai_provider, ai_api_key)
            if ai_patients:
                # Keep only records that have at least a name or several fields
                for p in ai_patients:
                    filled = sum(1 for v in p.values() if v)
                    if p.get("patient_name") or filled >= 3:
                        patients.append(p)
                if patients:
                    extraction_method = f"AI ({ai_provider})"

    if not patients:
        with st.spinner("Intelligently detecting patients & extracting fields (rules engine)…"):
            if force_single:
                patients = [extract_fields_from_block(full_text)]
            else:
                patients = extract_all_patients(full_text, page_texts)
            extraction_method = "rules"

    n = len(patients)
    if n == 0:
        st.warning("No patient records detected.")
        return

    st.success(f"✅ Detected **{n}** patient record{'s' if n != 1 else ''}  ·  method: **{extraction_method}**")

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
