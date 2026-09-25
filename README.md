# Patient Detail Extractor (Multi-Patient + Gemini AI)

Ultra-accurate Streamlit app that extracts critical patient information from **PDFs** (text or scanned) and **images**.  
Optional **free Google Gemini AI** for best accuracy across varied clinical formats.

## Quick setup with Gemini (recommended)

1. Get a **free** Gemini API key:  
   → [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

2. Push these files to the **root** of your GitHub repo:
   ```
   app.py
   requirements.txt
   packages.txt
   README.md
   ```

3. Deploy on [share.streamlit.io](https://share.streamlit.io) (Main file: `app.py`)

4. Add the key in **App settings → Secrets**:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```

5. **Reboot** the app once (so Tesseract + packages load).

6. In the sidebar, keep **AI Provider = Gemini**. Upload a document.

### `packages.txt` (required on Streamlit Cloud)
```
tesseract-ocr
tesseract-ocr-eng
poppler-utils
```

### `requirements.txt`
```
streamlit>=1.28.0
pdfplumber>=0.10.0
pdf2image>=1.16.0
pytesseract>=0.3.10
Pillow>=10.0.0
numpy>=1.24.0
pypdfium2>=4.0.0
groq>=0.9.0
google-generativeai>=0.7.0
```

## How it works

1. PDF/image is OCR’d **locally** (Tesseract).
2. OCR text is sent to **Gemini** (only when you enable AI + provide a key).
3. Gemini returns clean structured JSON for one or many patients.
4. You can edit any field before exporting JSON / CSV / TSV.
5. If no key is set, a strong rules engine still runs as fallback.

## Extracted fields

PT's Name · Phone · DOB · Email · Diagnosis · Referrer · Address · Insurance · Height · Weight · BMI

## Local run

```bash
sudo apt-get install -y tesseract-ocr poppler-utils
pip install -r requirements.txt
streamlit run app.py
```

Paste your Gemini key in the sidebar (or set `GEMINI_API_KEY` in secrets).

## Privacy

- OCR runs on the Streamlit server.
- Only the extracted text is sent to Google Gemini when AI is enabled.
- Always review results — this is medical data.
