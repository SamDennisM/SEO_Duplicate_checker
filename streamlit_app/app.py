import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from utils.parser import fetch_url, parse_html, clean_text, sentence_count
from utils.features import build_vectorizer_and_svd, compute_readability, avg_word_len
from utils.scorer import load_model, predict_label

st.set_page_config(page_title="SEO Content Quality & Duplicate Detector", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

@st.cache_data
def load_dataset():
    # Prefer features.csv if present; else fall back to extracted_content.csv
    feat_path = DATA_DIR / 'features.csv'
    if feat_path.exists():
        df = pd.read_csv(feat_path)
        if 'clean_text' not in df.columns:
            df['clean_text'] = df['body_text'].astype(str).str.lower().map(clean_text)
        # sanitize
        df['clean_text'] = df['clean_text'].fillna('').astype(str)
        return df
    ext_path = DATA_DIR / 'extracted_content.csv'
    df = pd.read_csv(ext_path)
    df['clean_text'] = df['body_text'].astype(str).str.lower().map(clean_text)
    df['clean_text'] = df['clean_text'].fillna('').astype(str)
    df['flesch_reading_ease'] = df['clean_text'].map(compute_readability)
    df['is_thin'] = df['word_count'] < 500
    return df

@st.cache_resource
def build_search_index(corpus: pd.Series):
    vec, svd, X_emb = build_vectorizer_and_svd(corpus)
    return vec, svd, X_emb

@st.cache_resource
def load_quality_model():
    return load_model(MODELS_DIR)


def analyze_url(url: str, threshold: float = 0.80, top_k: int = 5):
    html = fetch_url(url)
    title, body = parse_html(html)
    clean = body.lower().strip()
    wc = len(clean.split())
    sc = sentence_count(clean)
    read = compute_readability(clean)
    awl = avg_word_len(clean)

    model, feature_names = load_quality_model()
    Xq = np.array([[wc, sc, read, awl]])
    label = predict_label(model, Xq)
    thin = wc < 500

    vec, svd, X_emb = build_search_index(df['clean_text'])
    Xq_emb = svd.transform(vec.transform([clean]))
    sims = cosine_similarity(Xq_emb, X_emb)[0]
    idxs = np.argsort(sims)[::-1]
    similar = []
    for i in idxs[:max(top_k, 20)]:
        s = float(sims[i])
        if s >= threshold:
            similar.append({"url": df.loc[i, 'url'], "similarity": round(s, 4)})
        if len(similar) >= top_k:
            break

    return {
        "url": url,
        "title": title,
        "word_count": int(wc),
        "readability": float(read),
        "quality_label": label,
        "is_thin": bool(thin),
        "similar_to": similar,
    }

# Load dataset and build index
with st.spinner('Loading dataset and building search index...'):
    df = load_dataset()
    vec, svd, X_emb = build_search_index(df['clean_text'])

st.title("SEO Content Quality & Duplicate Detector")
st.write("Enter a URL to analyze its content quality and find near-duplicates in the dataset.")

col1, col2 = st.columns([3, 1])
with col1:
    url = st.text_input("URL", placeholder="https://example.com/article")
with col2:
    threshold = st.slider("Duplicate threshold", 0.5, 0.95, 0.80, 0.01)

if st.button("Analyze", type="primary"):
    if not url:
        st.warning("Please enter a URL.")
    else:
        with st.spinner('Analyzing URL...'):
            res = analyze_url(url, threshold=threshold)
        st.subheader("Quality Summary")
        met1, met2, met3, met4 = st.columns(4)
        met1.metric("Word Count", res['word_count'])
        met2.metric("Readability (FRE)", f"{res['readability']:.1f}")
        met3.metric("Quality Label", res['quality_label'])
        met4.metric("Thin Content", "Yes" if res['is_thin'] else "No")
        
        st.subheader("Near-Duplicate Matches")
        if res['similar_to']:
            st.table(pd.DataFrame(res['similar_to']))
        else:
            st.info("No near-duplicates above the threshold.")

st.markdown("---")
st.caption("Model trained from the included dataset. Similarity computed using TF-IDF + SVD embeddings.")
