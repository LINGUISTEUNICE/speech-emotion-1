"""
plot_pitch_by_emotion.py
One illustrative plot: pitch mean distribution across the three emotions.
Small, single-purpose — this is the plot the write-up actually references.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "results", "features.csv"))

plt.figure(figsize=(6, 4.5))
sns.boxplot(data=df, x="emotion", y="pitch_mean", order=["sad", "happy", "angry"], palette="Blues")
sns.stripplot(data=df, x="emotion", y="pitch_mean", order=["sad", "happy", "angry"],
              color="black", alpha=0.4, size=4)
plt.title("Mean Pitch (F0) by Emotion — RAVDESS speech clips")
plt.ylabel("Mean pitch (Hz)")
plt.xlabel("")
plt.tight_layout()
out_path = os.path.join(BASE, "results", "pitch_by_emotion.png")
plt.savefig(out_path, dpi=120)
print(f"Saved {out_path}")
