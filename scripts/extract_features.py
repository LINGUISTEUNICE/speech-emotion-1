"""
extract_features.py

Reads RAVDESS speech clips and pulls out a small set of prosodic features
per clip using librosa: pitch (F0), energy, and MFCCs. These are the
classic acoustic correlates of emotional speech in affective computing.

RAVDESS filenames encode the label directly, e.g. 03-01-03-01-01-01-01.wav
  modality-vocalchannel-EMOTION-intensity-statement-repetition-actor
Emotion codes used here: 03=happy, 04=sad, 05=angry

Output: results/features.csv
"""
import os
import glob
import numpy as np
import pandas as pd
import librosa

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "audio")
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "features.csv")

EMOTION_MAP = {"03": "happy", "04": "sad", "05": "angry"}


def parse_filename(path):
    name = os.path.basename(path).replace(".wav", "")
    parts = name.split("-")
    emotion_code = parts[2]
    actor = int(parts[6])
    gender = "female" if actor % 2 == 0 else "male"
    return EMOTION_MAP.get(emotion_code), actor, gender


def extract_features(path):
    y, sr = librosa.load(path, sr=None)

    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])

    rms = librosa.feature.rms(y=y)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    feats = {
        "pitch_mean": np.nanmean(f0_voiced) if len(f0_voiced) else np.nan,
        "pitch_std": np.nanstd(f0_voiced) if len(f0_voiced) else np.nan,
        "energy_mean": np.mean(rms),
        "energy_std": np.std(rms),
        "duration_sec": librosa.get_duration(y=y, sr=sr),
    }
    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])

    return feats


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    print(f"Found {len(files)} audio files")

    for i, path in enumerate(files):
        emotion, actor, gender = parse_filename(path)
        if emotion is None:
            continue
        feats = extract_features(path)
        feats.update({"file": os.path.basename(path), "emotion": emotion,
                       "actor": actor, "gender": gender})
        rows.append(feats)
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(files)}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH}")
    print(df["emotion"].value_counts())


if __name__ == "__main__":
    main()
