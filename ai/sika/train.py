"""Melatih ulang intent classifier dari data CSV.

Pakai ini (bukan file .joblib dari perangkat lain) supaya versi scikit-learn
sama dengan lingkungan yang menjalankan API.

Contoh:
    python -m sika.train
    python -m sika.train --data data/intent_dataset.csv --out models/intent_classifier.joblib
    python -m sika.train --full     # latih pakai SEMUA data (untuk produksi)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from . import config
from .preprocess import preprocess

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """Parameter terbaik dari GridSearchCV di notebook: C=1, class_weight=balanced, char n-gram (2,4)."""
    features = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=1, class_weight='balanced', max_iter=3000, random_state=RANDOM_STATE)
    return Pipeline([('tfidf', features), ('clf', clf)])


def main() -> None:
    ap = argparse.ArgumentParser(description='Latih intent classifier SIKA')
    ap.add_argument('--data', default=str(config.DATA_PATH))
    ap.add_argument('--out', default=str(config.MODEL_PATH))
    ap.add_argument('--full', action='store_true',
                    help='latih dengan seluruh data (tanpa menyisihkan data uji)')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df['text'].apply(preprocess)
    y = df['intent']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = build_pipeline().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f'Data latih: {len(X_tr)} | data uji: {len(X_te)}')
    print(f'Akurasi data uji : {accuracy_score(y_te, pred):.4f}')
    print(f'Macro-F1 data uji: {f1_score(y_te, pred, average="macro"):.4f}\n')
    print(classification_report(y_te, pred, digits=3))

    if args.full:
        model = build_pipeline().fit(X, y)
        print('Model akhir dilatih ulang dengan SEMUA data.')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f'Model tersimpan: {out}')


if __name__ == '__main__':
    main()
