#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.classical_ml import build_pipeline, get_base_models, load_features
from src.utils import ensure_dir


def score_model(model, X, y):
    pred = model.predict(X)
    return {
        'Accuracy': accuracy_score(y, pred),
        'Macro_Precision': precision_score(y, pred, average='macro', zero_division=0),
        'Macro_Recall': recall_score(y, pred, average='macro', zero_division=0),
        'Macro_F1': f1_score(y, pred, average='macro', zero_division=0),
    }, pred


def main():
    parser = argparse.ArgumentParser(description='Train ML models with train/val/test feature CSV files.')
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--feature-sizes', nargs='+', type=int, default=[10, 20, 40, 60, 80, 100])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    model_dir = ensure_dir(output / 'models')
    report_dir = ensure_dir(output / 'reports')

    _, X_train, y_train_raw = load_features(args.train)
    _, X_val, y_val_raw = load_features(args.val)
    _, X_test, y_test_raw = load_features(args.test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    joblib.dump(le, model_dir / 'label_encoder.pkl')

    all_rows = []
    fitted = {}

    for k in args.feature_sizes:
        base_models = get_base_models(args.seed)
        estimators = [(name.lower(), build_pipeline(name, model, int(k))) for name, model in base_models.items()]

        for name, model in base_models.items():
            start = time.time()
            pipe = build_pipeline(name, model, int(k))
            pipe.fit(X_train, y_train)
            train_time = time.time() - start
            val_scores, _ = score_model(pipe, X_val, y_val)
            key = f'{name}_{k}feat'
            fitted[key] = pipe
            joblib.dump(pipe, model_dir / f'{key}.pkl')
            all_rows.append({'Model': name, 'FeatureCount': k, 'Train_Time_s': train_time, **{f'Val_{m}': v for m, v in val_scores.items()}})

        start = time.time()
        voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        voting.fit(X_train, y_train)
        train_time = time.time() - start
        val_scores, _ = score_model(voting, X_val, y_val)
        fitted[f'Voting_{k}feat'] = voting
        joblib.dump(voting, model_dir / f'Voting_{k}feat.pkl')
        all_rows.append({'Model': 'Voting', 'FeatureCount': k, 'Train_Time_s': train_time, **{f'Val_{m}': v for m, v in val_scores.items()}})

        start = time.time()
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=5000, random_state=args.seed),
            n_jobs=-1
        )
        stack.fit(X_train, y_train)
        train_time = time.time() - start
        val_scores, _ = score_model(stack, X_val, y_val)
        fitted[f'Stacking_{k}feat'] = stack
        joblib.dump(stack, model_dir / f'Stacking_{k}feat.pkl')
        all_rows.append({'Model': 'Stacking', 'FeatureCount': k, 'Train_Time_s': train_time, **{f'Val_{m}': v for m, v in val_scores.items()}})

    val_df = pd.DataFrame(all_rows)
    val_df.to_csv(report_dir / 'ml_validation_results.csv', index=False)

    best_rows = []
    for _, group in val_df.sort_values('Val_Accuracy', ascending=False).groupby('Model'):
        best_rows.append(group.iloc[0].to_dict())
    best_df = pd.DataFrame(best_rows).sort_values('Val_Accuracy', ascending=False)
    best_df.to_csv(report_dir / 'ml_best_by_validation.csv', index=False)

    test_rows = []
    for _, row in best_df.iterrows():
        key = f"{row['Model']}_{int(row['FeatureCount'])}feat"
        model = fitted[key]
        test_scores, y_pred = score_model(model, X_test, y_test)
        test_rows.append({'Model': row['Model'], 'FeatureCount': int(row['FeatureCount']), **{f'Test_{m}': v for m, v in test_scores.items()}})
        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(report_dir / f'cm_{key}.csv', encoding='utf-8-sig')

    test_df = pd.DataFrame(test_rows).sort_values('Test_Accuracy', ascending=False)
    test_df.to_csv(report_dir / 'ml_test_results_selected_by_val.csv', index=False)

    print('\nBest models selected by validation accuracy:')
    print(best_df[['Model', 'FeatureCount', 'Val_Accuracy', 'Val_Macro_F1']].to_string(index=False))
    print('\nIndependent test results:')
    print(test_df.to_string(index=False))


if __name__ == '__main__':
    main()
