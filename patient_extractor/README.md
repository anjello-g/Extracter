# Patient Detail Extractor (Multi-Patient)

Ultra-accurate Streamlit app that extracts critical patient information from **PDFs** (text or scanned) and **images**.

## Key Features

- **Multi-patient support** – automatically detects and separates multiple patients in one file
- **Hybrid extraction** – native PDF text first, high-quality multi-config OCR fallback
- **Aggressive image preprocessing** – upscaling, contrast, sharpening, denoising
- **Label-aware + regex + validation** for maximum field accuracy
- **Auto BMI calculation** from height + weight
- **Easy-to-copy output**:
  - Interactive table (select cells → Ctrl/Cmd+C)
  - Per-patient plain-text blocks
  - One-click JSON / CSV / TSV downloads
- **100% local** – no data leaves your machine

## Extracted Fields

| Field              | Field              | Field          |
|--------------------|--------------------|----------------|
| PT's Name          | PT's Phone Number  | PT's DOB       |
| PT's Email         | PT's Diagnosis     | Referrer       |
| Address            | Insurance          | Height         |
| Weight             | BMI                |                |

## Requirements

### System
```bash
# Ubuntu / Debian
sudo apt-get install -y tesseract-ocr poppler-utils
```

### Python
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Accuracy Tips
1. Prefer ≥ 200–300 DPI scans
2. Clear printed text, good contrast, minimal skew
3. Always review results — this is medical data
4. Use “Force single-patient mode” in the sidebar if splitting is unwanted
5. Use “Show raw OCR / text” to verify extraction quality

## Privacy
All processing is local. No external APIs or cloud services are used.
