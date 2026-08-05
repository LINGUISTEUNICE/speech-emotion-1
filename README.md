# Speech Emotion — A First Look

A small, hands-on first project exploring this kind of multimodal
communication analysis. This round is scoped to a single
modality (speech audio)  — see [`docs/findings.md`](docs/findings.md)



## What's here

- 96 real audio clips from RAVDESS (4 actors, 3 emotions: happy/sad/angry)
- A feature extraction script pulling pitch, energy, and MFCCs out of raw audio
- A simple logistic regression classifier predicting emotion from those features
- Two plots and a short write-up discussing what the results actually show

## Structure

```
multicom-light/
├── data/audio/                    # 96 RAVDESS speech clips (CC BY-NC-SA 4.0)
├── scripts/
│   ├── extract_features.py        # audio -> results/features.csv
│   ├── train_classify.py          # trains classifier, saves confusion matrix
│   └── plot_pitch_by_emotion.py   # generates the pitch boxplot
├── results/
│   ├── features.csv
│   ├── confusion_matrix.png
│   └── pitch_by_emotion.png
├── docs/findings.md               # the actual write-up
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python3 scripts/extract_features.py       # ~1-2 min for 96 clips
python3 scripts/train_classify.py
python3 scripts/plot_pitch_by_emotion.py
```

## Dataset credit

Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of
Emotional Speech and Song (RAVDESS). Licensed CC BY-NC-SA 4.0. Only a small
subset (4 of 24 actors, 3 of 8 emotions) is included here.

## Next step 

Pull the matching video for these same actors/emotions and add facial
features (MediaPipe FaceMesh) to see whether visual cues resolve the
happy/sad confusion that shows up in the audio-only model
