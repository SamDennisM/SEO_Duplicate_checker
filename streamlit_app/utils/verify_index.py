import pandas as pd
from pathlib import Path
import sys

# Ensure we can import from streamlit_app
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.features import build_vectorizer_and_svd

data_dir = Path(__file__).resolve().parents[2] / 'data'
df = pd.read_csv(data_dir / 'features.csv')
if 'clean_text' not in df.columns:
    raise SystemExit('clean_text not found')
vec, svd, X_emb = build_vectorizer_and_svd(df['clean_text'])
print('OK', X_emb.shape)
