# Findings: Can Pitch and Energy Alone Tell Happy, Sad, and Angry Apart?

This is a small first attempt at the kind of multimodal-communication analysis


## What I did

- 96 real clips from the RAVDESS dataset (Livingstone & Russo, 2018) — 4 actors
  (2 male, 2 female), 3 emotions (happy, sad, angry), both scripted sentences
- Extracted acoustic features with `librosa`: pitch (F0), energy (RMS), and
  MFCCs — the standard first-pass features in speech emotion research
- Trained a plain logistic regression classifier (no deep learning) to predict
  emotion from these features
- Looked at which features the model actually leaned on, not just the accuracy number

## What came out of it

**The model got 67% accuracy on held-out clips (chance level for 3 classes is 33%).**
Not a strong result, and I'm not claiming it should be — it's a tiny dataset
and a simple model. What's more interesting than the number is *where* it got
confused, and *why*.

![Confusion matrix](results/confusion_matrix.png)

Angry and happy were rarely confused with each other, but **sad got mixed up
with happy more than with angry** (3 of 8 sad clips predicted as happy, 0
predicted as angry). At first that seemed backwards — sad and happy feel like
opposite emotions. But looking at the pitch data explains it:

![Pitch by emotion](results/pitch_by_emotion.png)

Sad and happy clips actually have a very similar **median** pitch — the real
separation between them isn't pitch height, it's how *stable* the pitch stays
(sad clips have a wider spread — some very flat/monotone, some not — while
angry speech sits in a tighter, higher band). Energy told a cleaner story:
angry clips were reliably the loudest, sad the quietest.

This tracks with something I read about in affective computing while working
on this: emotion in the voice tends to separate along two axes — **arousal**
(energy/activation — how "worked up" the voice sounds) and **valence**
(positive/negative feeling). Pitch and energy are strong arousal cues, which
is why angry (high arousal, negative) is easy to tell apart acoustically —
but happy (high arousal, positive) and sad (low arousal, negative) don't
sit on the same axis as each other, so a model using mostly arousal-related
features has an easier time separating "energetic vs. calm" than "positive
vs. negative." That's a genuinely useful thing to have found by actually
running the numbers, not just something I read.

## What I'd want to check next 

I kept this round to audio only The  next step is to pull the matching
video clips for these same files, extract basic facial features (eyebrow/mouth
movement via MediaPipe), and see whether the same happy/sad confusion the
audio model shows also shows up visually, or whether the face actually
disambiguates it 

## Honest limitations

- Only 4 actors, 96 clips — nowhere near enough to generalize confidently
- Acted/read-aloud emotion, not spontaneous — RAVDESS actors are performing
  a scripted line, which is a real and known limitation of this kind of dataset
- Logistic regression on hand-picked features, not a model built to squeeze
  out best-possible accuracy — that wasn't the point of this pass
- No facial or gesture data yet, as noted above

This was a few evenings of work, not a research project — but it's the first
time I've actually pulled acoustic features out of raw audio and watched a
model's mistakes tell me something real about how emotion shows up in a
voice, instead of just reading that it does.
