# SEO Content Quality & Duplicate Detector

## Project Overview
End-to-end pipeline to parse pre-scraped HTML, extract NLP features, detect near-duplicates, and score content quality. Core deliverable is a reproducible Jupyter notebook; optional real-time URL analysis is included.

## Setup Instructions
- Clone and install:
  - git clone https://github.com/yourusername/seo-content-detector
  - cd seo-content-detector
  - pip install -r requirements.txt
- Launch notebook:
  - jupyter notebook notebooks/seo_pipeline.ipynb

## Quick Start
- Ensure `data/data.csv` exists (provided; columns: `url`, `html_content`).
- Run all cells in `notebooks/seo_pipeline.ipynb`.
- Outputs (saved under `data/`):
  - `extracted_content.csv` (url, title, body_text, word_count)
  - `features.csv` (core features, thin flag, labels)
  - `duplicates.csv` (duplicate URL pairs)
  - Model saved to `models/quality_model.pkl`

## Key Decisions
- Parsing: BeautifulSoup with priority on `<main>`, `<article>`, then `<p>` fallback; scripts/styles removed.
- Features: TF‑IDF (1–2 grams) with SVD (50D) as embeddings; Flesch Reading Ease via textstat.
- Similarity threshold: 0.80 on cosine(SVD embeddings) to flag near-duplicates.
- Model: RandomForestClassifier using word_count, sentence_count, readability (+ simple text stats). Baseline uses word_count-only rules.

## Results Summary
- Reported in notebook: accuracy, F1, confusion matrix, top features.
- Duplicate statistics and thin-content counts printed and saved.
- Example `analyze_url(url)` returns quality label and similar pages.

## Limitations
- Readability scores can be noisy for very short texts.
- TF‑IDF vocabulary depends on this dataset; unseen pages may have weaker embeddings.
- Simple content extraction may miss highly dynamic sites.

## Streamlit (Optional Bonus)
- Local run:
  - pip install -r requirements.txt
  - streamlit run streamlit_app/app.py
- Deployed URL: (https://seoduplicatechecker-biisof8ujafcu7jovmtkzx.streamlit.app/)
