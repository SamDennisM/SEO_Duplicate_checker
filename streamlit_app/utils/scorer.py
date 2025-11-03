from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from typing import Dict, Any


def load_model(models_dir: Path):
    obj = joblib.load(models_dir / 'quality_model.pkl')
    return obj['model'], obj['features']


def predict_label(model, features_vector: np.ndarray) -> str:
    return str(model.predict(features_vector)[0])
