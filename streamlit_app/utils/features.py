from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import textstat


def build_vectorizer_and_svd(corpus: pd.Series):
    # Ensure no NaNs and all strings
    corpus = corpus.fillna('').astype(str)
    vec = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(corpus)
    n_docs = X.shape[0]
    n_comp = int(min(50, max(2, n_docs - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_emb = svd.fit_transform(X)
    return vec, svd, X_emb


def compute_readability(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return float(textstat.flesch_reading_ease(text))


def avg_word_len(text: str) -> float:
    toks = text.split()
    return float(np.mean([len(t) for t in toks])) if toks else 0.0
