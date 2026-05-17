from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .classical_ml import load_features
from .utils import ensure_dir


def plot_feature_dimensionality(results_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(results_csv)
    plt.figure(figsize=(10, 6))
    for model_name in df['Model'].unique():
        sub = df[df['Model'] == model_name].sort_values('FeatureCount')
        plt.plot(sub['FeatureCount'], sub['Test_Accuracy'], marker='o', label=model_name)
    plt.xlabel('Number of Selected Features (k)')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Feature Dimensionality on Classification Performance')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_confusion_matrices(test_csv: str | Path, model_dir: str | Path, output_dir: str | Path) -> None:
    _, X_test, y_raw = load_features(test_csv)
    le = joblib.load(Path(model_dir) / 'label_encoder.pkl')
    y_test = le.transform(y_raw)
    class_names = le.classes_
    output_dir = ensure_dir(output_dir)
    for model_path in sorted(Path(model_dir).glob('*.pkl')):
        if model_path.name == 'label_encoder.pkl':
            continue
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm)
        ax.set_title(model_path.stem)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(np.arange(len(class_names)), class_names, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(class_names)), class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f'{model_path.stem}_confusion_matrix.png', dpi=300)
        plt.close(fig)
