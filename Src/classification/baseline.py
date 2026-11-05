"""Milestone 1: untuned baseline classifiers, evaluated with 5-fold CV on TRAIN only.
The test set is NOT used here. Run: python -m src.classification.baseline
"""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from src.data.loader import SEED, load_raw
from src.classification.features import feature_columns
from src.classification.preprocessing import build_preprocessor
from src.classification.split import make_split

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main():
    df = load_raw()
    cols = feature_columns(df)                      # strict booking-time feature set
    X_train, X_test, y_train, y_test = make_split(df, cols)   # test set stays untouched
    print(f"Train {X_train.shape}, Test {X_test.shape} (test not used)")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, (name, model) in zip(axes, models.items()):
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])
        r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name, **{m: round(r[f"test_{m}"].mean(), 3) for m in SCORING}}
        row["fit_seconds"] = round(r["fit_time"].mean(), 1)
        rows.append(row)

        # Confusion matrix from out-of-fold predictions (train data only)
        y_pred = cross_val_predict(pipe, X_train, y_train, cv=cv)
        ConfusionMatrixDisplay.from_predictions(
            y_train, y_pred, display_labels=["Not canceled", "Canceled"], ax=ax, colorbar=False)
        ax.set_title(f"{name} (CV, train)")

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig("outputs/figures/baseline_confusion_matrices.png", dpi=120); plt.close(fig)

    out = pd.DataFrame(rows)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv("outputs/tables/classification_baseline_cv.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
