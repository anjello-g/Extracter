# Patient Detail Extractor (Multi-Patient)

Ultra-accurate Streamlit app that extracts critical patient information from **PDFs** (text or scanned) and **images**.  
Supports **multiple patients** in a single file.

## Deploy on Streamlit Community Cloud (GitHub)

1. Create a new GitHub repository and push these files:
   ```
   app.py
   requirements.txt
   packages.txt          ← required for Tesseract + Poppler
   README.md
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
   - Repository: your GitHub repo
   - Branch: `main` (or whatever you use)
   - Main file path: `app.py`

3. Click **Deploy**. Streamlit Cloud will install the system packages listed in `packages.txt` and the Python packages from `requirements.txt`.

4. After the first deploy finishes, **reboot the app** once (⋮ menu → Reboot) so Tesseract is fully available.

### Required files for Cloud

**`packages.txt`** (system packages – already included):
```
tesseract-ocr
tesseract-ocr-eng
poppler-utils
```

**`requirements.txt`**:
```
streamlit>=1.28.0
pdfplumber>=0.10.0
pdf2image>=1.16.0
pytesseract>=0.3.10
Pillow>=10.0.0
numpy>=1.24.0
pypdfium2>=4.0.0
```

## Local run

```bash
# System deps (Ubuntu/Debian)
sudo apt-get install -y tesseract-ocr poppler-utils

# Python
pip install -r requirements.txt
streamlit run app.py
```

## Features

- **Multi-patient** detection & separation
- Hybrid native-text + multi-config OCR
- Aggressive image preprocessing for accuracy
- Label-aware extraction + validation
- Auto BMI calculation
- Easy-to-copy table, plain-text blocks, JSON / CSV / TSV export
- Fully local processing (no external APIs)

## Extracted fields

PT's Name · Phone · DOB · Email · Diagnosis · Referrer · Address · Insurance · Height · Weight · BMI

## Notes for Streamlit Cloud

- Scanned PDFs use OCR → can be slower / more memory-intensive on the free tier.
- Prefer ≤ 10–15 page documents when possible.
- If you see “Tesseract is not available”, confirm `packages.txt` is in the **root** of the repo and reboot the app.
- Text-layer PDFs (not scanned) work even without Tesseract.

Always review extracted values — this is medical data.
