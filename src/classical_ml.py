from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from .utils import ensure_dir


def load_features(csv_path: str | Path):
    df = pd.read_csv(csv_path, sep=';')
    drop_cols = [c for c in ['label', 'image_name', 'image_path'] if c in df.columns]
    X = df.drop(drop_cols, axis=1)
    y_raw = df['label']
    return df, X, y_raw


def get_base_models(random_state: int = 42) -> Dict[str, object]:
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        'LogisticRegression': LogisticRegression(max_iter=5000, n_jobs=-1, random_state=random_state),
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
        'NaiveBayes': GaussianNB(),
        'SGD': SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=2000, random_state=random_state),
    }
    if XGBClassifier is not None:
        models['XGBoost'] = XGBClassifier(eval_metric='mlogloss', random_state=random_state, verbosity=0)
    if LGBMClassifier is not None:
        models['LightGBM'] = LGBMClassifier(random_state=random_state, verbose=-1)
    return models


def build_pipeline(model_name: str, model, k_features: int) -> Pipeline:
    tree_like = {'RandomForest', 'ExtraTrees', 'LightGBM', 'XGBoost'}
    steps = []
    if model_name not in tree_like:
        steps.append(('scaler', StandardScaler()))
    steps.append(('selector', SelectKBest(score_func=mutual_info_classif, k=k_features)))
    steps.append(('model', model))
    return Pipeline(steps)


def train_all(train_csv: str | Path, test_csv: str | Path, feature_sizes: Iterable[int], output_dir: str | Path,
              random_state: int = 42) -> pd.DataFrame:
    output_dir = Path(output_dir)
    model_dir = ensure_dir(output_dir / 'models')
    report_dir = ensure_dir(output_dir / 'reports')

    _, X_train, y_train_raw = load_features(train_csv)
    _, X_test, y_test_raw = load_features(test_csv)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    joblib.dump(le, model_dir / 'label_encoder.pkl')

    results = []
    for k in feature_sizes:
        base_models = get_base_models(random_state)
        fitted_for_ensembles = {}
        for name, model in base_models.items():
            start = time.time()
            pipe = build_pipeline(name, model, int(k))
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            elapsed = time.time() - start
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            joblib.dump(pipe, model_dir / f'{name}_{k}feat.pkl')
            fitted_for_ensembles[name] = pipe
            results.append({'FeatureCount': k, 'Model': name, 'Test_Accuracy': acc,
                            'Macro_Precision': prec, 'Macro_Recall': rec, 'Train_Time_s': elapsed})

        estimators = [(name.lower(), build_pipeline(name, model, int(k))) for name, model in base_models.items()]
        start = time.time()
        voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        voting.fit(X_train, y_train)
        y_pred = voting.predict(X_test)
        elapsed = time.time() - start
        joblib.dump(voting, model_dir / f'Voting_{k}feat.pkl')
        results.append({'FeatureCount': k, 'Model': 'Voting', 'Test_Accuracy': accuracy_score(y_test, y_pred),
                        'Macro_Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                        'Macro_Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                        'Train_Time_s': elapsed})

        start = time.time()
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=5000, random_state=random_state),
            n_jobs=-1
        )
        stack.fit(X_train, y_train)
        y_pred = stack.predict(X_test)
        elapsed = time.time() - start
        joblib.dump(stack, model_dir / f'Stacking_{k}feat.pkl')
        results.append({'FeatureCount': k, 'Model': 'Stacking', 'Test_Accuracy': accuracy_score(y_test, y_pred),
                        'Macro_Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                        'Macro_Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                        'Train_Time_s': elapsed})

    res_df = pd.DataFrame(results)
    res_df.to_csv(report_dir / 'model_comparison_different_features.csv', index=False)
    best = res_df.sort_values('Test_Accuracy', ascending=False).groupby('Model').head(1)
    best.to_csv(report_dir / 'best_models.csv', index=False)
    return res_df
