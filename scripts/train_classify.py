"""
train_classify.py

A deliberately simple classifier — logistic regression, not a neural net —
trained on the prosodic features from extract_features.py. The point of
this project isn't to chase accuracy, it's to see whether basic acoustic
features carry usable emotion signal, and which ones matter.

Output: results/confusion_matrix.png, printed accuracy + per-emotion report
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE, "results", "features.csv")
PLOT_PATH = os.path.join(BASE, "results", "confusion_matrix.png")

FEATURE_COLS = ["pitch_mean", "pitch_std", "energy_mean", "energy_std", "duration_sec"] + \
               [f"mfcc{i+1}_mean" for i in range(13)]


def main():
    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURE_COLS].values
    y = df["emotion"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")
    print(classification_report(y_test, preds))

    # Which features matter most? Look at coefficient magnitude per class.
    coef_df = pd.DataFrame(clf.coef_, columns=FEATURE_COLS, index=clf.classes_)
    print("\nTop 5 features by average |coefficient| across classes:")
    top_feats = coef_df.abs().mean().sort_values(ascending=False).head(5)
    print(top_feats)

    cm = confusion_matrix(y_test, preds, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap="Blues")
    plt.title("Emotion Classification — Confusion Matrix\n(prosodic features only, logistic regression)")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved confusion matrix to {PLOT_PATH}")


if __name__ == "__main__":
    main()
