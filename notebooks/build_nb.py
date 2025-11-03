import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""
# SEO Content Quality & Duplicate Detector

Reproducible pipeline: parse HTML, extract features, detect duplicates, and score content quality.
"""))

cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import re, json
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import textstat
import requests
from tqdm.auto import tqdm
import joblib

tqdm.pandas()

DATA_DIR = '../data'
MODELS_DIR = '../models'
"""))

cells.append(nbf.v4.new_code_cell("""
# Load dataset (expects columns: url, html_content)
df_raw = pd.read_csv(f"{DATA_DIR}/data.csv")
(len(df_raw), df_raw.columns.tolist())
"""))

cells.append(nbf.v4.new_code_cell("""
# HTML parsing utilities
def clean_text(s):
    if not isinstance(s, str):
        return ''
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_title(soup):
    if soup.title and soup.title.string:
        return clean_text(soup.title.string)
    h1 = soup.find('h1')
    return clean_text(h1.get_text(separator=' ')) if h1 else ''


def extract_main_text(soup):
    # Priority: <main>, <article>, then role=main, else fallback to <p> tags
    # Try explicit tags first
    for selector in ['main', 'article']:
        node = soup.find(selector)
        if node:
            for bad in node.find_all(['script','style','noscript','svg','nav','footer','header','aside']):
                bad.decompose()
            txt = clean_text(node.get_text(separator=' '))
            if len(txt.split()) > 50:
                return txt
    # Try role=main
    node = soup.find(attrs={'role': 'main'})
    if node:
        for bad in node.find_all(['script','style','noscript','svg','nav','footer','header','aside']):
            bad.decompose()
        txt = clean_text(node.get_text(separator=' '))
        if len(txt.split()) > 50:
            return txt

    # Fallback to <p>
    for bad in soup.find_all(['script','style','noscript','svg']):
        bad.decompose()
    ps = [p.get_text(separator=' ') for p in soup.find_all('p')]
    txt = clean_text(' '.join(ps)) if ps else clean_text(soup.get_text(separator=' '))
    return txt


def parse_html(html):
    try:
        soup = BeautifulSoup(html, 'lxml')
        title = extract_title(soup)
        body = extract_main_text(soup)
        return title, body
    except Exception:
        return '', ''


def sentence_count(text):
    if not text:
        return 0
    # Simple split on sentence-ending punctuation
    parts = re.split(r'[.!?]+', text)
    return len([p for p in parts if p.strip()])
"""))

cells.append(nbf.v4.new_code_cell("""
# Parse HTML -> extract title, body_text, counts
rows = []
for i, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
    url = row.get('url', '')
    html = row.get('html_content', '')
    title, body = parse_html(html)
    wc = len(body.split())
    sc = sentence_count(body)
    rows.append({'url': url, 'title': title, 'body_text': body, 'word_count': wc, 'sentence_count': sc})

df = pd.DataFrame(rows)
df.to_csv(f"{DATA_DIR}/extracted_content.csv", index=False)
df.head()
"""))

cells.append(nbf.v4.new_code_cell("""
# Feature engineering: readability, TF-IDF, SVD embeddings, thin flag
df_feat = df.copy()
df_feat['clean_text'] = df_feat['body_text'].str.lower().map(clean_text)
df_feat['flesch_reading_ease'] = df_feat['clean_text'].apply(lambda x: textstat.flesch_reading_ease(x) if isinstance(x,str) and x.strip() else 0.0)

vectorizer = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2), min_df=2)
X_tfidf = vectorizer.fit_transform(df_feat['clean_text'])
feature_names = np.array(vectorizer.get_feature_names_out())


def top_k_keywords(row, k=5):
    idx = row.indices
    data = row.data
    if len(data) == 0:
        return ''
    top_idx = np.argsort(data)[-k:][::-1]
    return '|'.join(feature_names[idx[top_idx]])


df_feat['top_keywords'] = [top_k_keywords(X_tfidf.getrow(i), k=5) for i in range(X_tfidf.shape[0])]

n_docs = X_tfidf.shape[0]
n_comp = int(min(50, max(2, n_docs - 1)))
svd = TruncatedSVD(n_components=n_comp, random_state=42)
X_emb = svd.fit_transform(X_tfidf)
df_feat['embedding'] = [json.dumps(vec.tolist()) for vec in X_emb]

df_feat['is_thin'] = (df_feat['word_count'] < 500)

df_feat.to_csv(f"{DATA_DIR}/features.csv", index=False)
df_feat.head()
"""))

cells.append(nbf.v4.new_code_cell("""
# Duplicate detection via cosine similarity on SVD embeddings
sim_matrix = cosine_similarity(X_emb)
pairs = []
threshold = 0.80
urls = df_feat['url'].tolist()
for i in range(n_docs):
    for j in range(i+1, n_docs):
        s = float(sim_matrix[i, j])
        if s >= threshold:
            pairs.append({'url1': urls[i], 'url2': urls[j], 'similarity': round(s, 4)})

dup_df = pd.DataFrame(pairs)
dup_df.to_csv(f"{DATA_DIR}/duplicates.csv", index=False)
print(f"Total pages: {n_docs}")
print(f"Duplicate pairs (>= {threshold}): {len(dup_df)}")
print(f"Thin content pages: {df_feat['is_thin'].sum()} ({df_feat['is_thin'].mean():.0%})")
dup_df.head()
"""))

cells.append(nbf.v4.new_code_cell("""
# Quality labels (synthetic) and model training
def rule_label(row):
    wc = row['word_count']
    r = row['flesch_reading_ease']
    if wc > 1500 and 50 <= r <= 70:
        return 'High'
    if wc < 500 or r < 30:
        return 'Low'
    return 'Medium'


df_feat['label'] = df_feat.apply(rule_label, axis=1)


def avg_word_len(text):
    toks = text.split()
    return float(np.mean([len(t) for t in toks])) if toks else 0.0


df_feat['avg_word_len'] = df_feat['clean_text'].apply(avg_word_len)

feat_cols = ['word_count', 'sentence_count', 'flesch_reading_ease', 'avg_word_len']
X = df_feat[feat_cols].values
y = df_feat['label'].values
indices = np.arange(len(df_feat))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.30, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced_subsample')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Model Performance:')
print(classification_report(y_test, y_pred, digits=3))
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Overall Accuracy: {acc:.3f} | Weighted F1: {f1:.3f}")

# Baseline (word_count-only rules)
def baseline_label(wc):
    if wc > 1500:
        return 'High'
    if wc < 500:
        return 'Low'
    return 'Medium'


y_base = [baseline_label(df_feat.loc[i, 'word_count']) for i in idx_test]
print('Baseline (word_count-only) Accuracy:', accuracy_score(y_test, y_base))

# Top features
importances = clf.feature_importances_
top_idx = np.argsort(importances)[::-1]
print('Top Features:')
for k in top_idx[:3]:
    print(f"- {feat_cols[k]} (importance: {importances[k]:.3f})")

# Save model
joblib.dump({'model': clf, 'features': feat_cols}, f"{MODELS_DIR}/quality_model.pkl")
"""))

cells.append(nbf.v4.new_code_cell("""
# Real-time analysis: analyze_url(url)
HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; SEO-Detector/1.0; +https://example.com/bot)'}

def fetch_url(url, timeout=15):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return ''


def analyze_url(url, top_k=5, sim_threshold=0.75):
    html = fetch_url(url)
    title, body = parse_html(html)
    clean = body.lower().strip()
    wc = len(clean.split())
    sc = sentence_count(clean)
    read = textstat.flesch_reading_ease(clean) if clean else 0.0
    awl = float(np.mean([len(t) for t in clean.split()])) if clean else 0.0
    # Predict label
    Xq = np.array([[wc, sc, read, awl]])
    label = clf.predict(Xq)[0]
    thin = wc < 500
    # Similarity vs dataset
    Xq_tfidf = vectorizer.transform([clean])
    Xq_emb = svd.transform(Xq_tfidf)
    sims = cosine_similarity(Xq_emb, X_emb)[0]
    idxs = np.argsort(sims)[::-1]
    similar = []
    urls_local = df_feat['url'].tolist()
    for i in idxs[:top_k]:
        if sims[i] >= sim_threshold:
            similar.append({'url': urls_local[i], 'similarity': float(round(sims[i], 4))})
    return {
        'url': url,
        'title': title,
        'word_count': int(wc),
        'readability': float(read),
        'quality_label': str(label),
        'is_thin': bool(thin),
        'similar_to': similar
    }

# Example:
# result = analyze_url(df_feat['url'].iloc[0])
# import json; print(json.dumps(result, indent=2))
"""))

nb['cells'] = cells

# Set kernelspec to this venv kernel
nb['metadata'] = nb.get('metadata', {})
nb['metadata']['kernelspec'] = {
    'name': 'seo-detector-venv',
    'display_name': 'Python (seo-detector-venv)',
    'language': 'python'
}
nb['metadata']['language_info'] = {'name': 'python', 'version': '3'}

with open(r'F:\LeadWalnut\seo-content-detector\notebooks\seo_pipeline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
