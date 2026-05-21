---
title: Dialog Emotion Timeline
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
python_version: "3.10"
fullWidth: true
---

# Dialog Emotion Timeline

Gradio demo for tracking an emotional probability distribution across a
dialogue: `joy`, `warmth`, `sadness`, `anger`, `anxiety`, `neutral`.

The UI reads a CSV. Producing that CSV is a separate offline pipeline:
Telegram JSON export → parser → model → CSV → UI.

## Local run

```bash
uv run python app.py
```

## Pipeline

```
result.json ──► load_telegram_export ──► DataFrame[turn_index,timestamp,sender,text]
                                                       │
                                                       ▼
                                       EmotionModel.predict_proba(text)
                                                       │
                                                       ▼
                                  score_dialog (+ softmax if needed)
                                                       │
                                                       ▼
                                  write_scored_csv (validates schema)
                                                       │
                                                       ▼
                                       data/*.csv ──► Gradio UI
```

CLI entrypoint:

```bash
uv run python scripts/score_telegram_export.py \
  --input data/result.json \
  --output output/scored.csv \
  --parsed-output output/parsed.csv \   # optional: dump parsed dialog before scoring
  --model uniform
```

## Contracts

### 1. Telegram JSON → parsed frame

`dialog_emo_demo.telegram.load_telegram_export(path)` reads a Telegram
Desktop `result.json` and returns a `DataFrame` with four columns:

| column      | type | notes                                             |
| ----------- | ---- | ------------------------------------------------- |
| turn_index  | int  | Dense 0..N-1 after filtering                      |
| timestamp   | str  | ISO string from the message `date` field          |
| sender      | str  | `from`, fallback `from_id`                        |
| text        | str  | Flattened (entities + custom emojis preserved)    |

Dropped:
- `type == "service"`
- `media_type ∈ {sticker, voice_message, video_message}`
- messages with empty text after flattening

Kept (with their captions, if any): photo, video, animation, audio, plain
text messages, forwards, edited messages. The `text` field in TG JSON is
either a string or a list of segments (entities like mention, link,
custom_emoji); segments are concatenated by their `text` attribute.

### 2. Model interface

A model is anything that subclasses `dialog_emo_demo.models.EmotionModel`
and implements one method:

```python
from dialog_emo_demo.models import EmotionModel

class MyModel(EmotionModel):
    def predict_proba(self, text: str) -> dict[str, float]:
        ...
```

Contract:
- Input: one text string (one dialog turn).
- Output: a dict with **all six** keys: `joy, warmth, sadness, anger, anxiety, neutral`.
- Values may be probabilities (sum ≈ 1, all ≥ 0) **or** raw logits / scores —
  the pipeline normalises before writing the CSV (see below).
- Missing keys raise `ValueError` during scoring.

No batching, validation, or normalisation in the base class — keep your
subclass focused on inference. If you need batching, do it inside
`predict_proba` (cache, buffer, whatever) — the contract is per-text.

### 3. Scoring: logits → probabilities

`dialog_emo_demo.pipeline.score_dialog(frame, model)` adds the six emotion
columns. The rule:

- If the model output already looks like a probability distribution
  (all values ≥ 0 **and** sum within `1 ± 1e-3`), it is written as-is.
- Otherwise a numerically-stable softmax is applied over the six values.

So you can return whatever your model produces — probabilities, logits,
unnormalised scores — and the CSV always ends up with a valid distribution.

### 4. CSV schema (what the UI reads)

Columns, in order, validated by `validate_dialog_frame`:

```
turn_index, timestamp, sender, text, joy, warmth, sadness, anger, anxiety, neutral
```

Per row: each emotion in `[0, 1]`; six emotions sum to `1 ± 1e-3`. Bad
rows fail loudly with `DialogDataError`.

### 5. Model registry

`dialog_emo_demo.pipeline.MODEL_REGISTRY` is a `dict[str, () -> EmotionModel]`.
The CLI's `--model` flag picks from it. To register your model:

```python
# anywhere that runs before the CLI parses args, e.g. at the bottom of pipeline.py
from dialog_emo_demo.pipeline import MODEL_REGISTRY
from my_package import MyLogReg

MODEL_REGISTRY["logreg"] = lambda: MyLogReg.load("artifacts/logreg.joblib")
```

Lazy construction (lambda / factory) keeps heavyweight loads — HF
checkpoints, joblib pickles — out of import time so unrelated CLI commands
stay fast.

## Adding a new model — full example

```python
# my_models.py
from dialog_emo_demo.models import EmotionModel, EMOTION_GROUPS

class KeywordModel(EmotionModel):
    KEYWORDS = {
        "joy": ("рад", "класс", "ура"),
        "anxiety": ("боюсь", "тревог", "страш"),
        # ...
    }

    def predict_proba(self, text: str) -> dict[str, float]:
        text = text.lower()
        scores = {emotion: 0.0 for emotion in EMOTION_GROUPS}
        for emotion, stems in self.KEYWORDS.items():
            scores[emotion] = sum(text.count(stem) for stem in stems)
        if sum(scores.values()) == 0:
            scores["neutral"] = 1.0
        return scores  # softmax handled by the pipeline
```

Wire it in by appending to `MODEL_REGISTRY` and run:

```bash
uv run python scripts/score_telegram_export.py \
  --input data/result.json --output output/keyword.csv --model keyword
```

## Tests

```bash
uv run pytest
```

Covered:
- Parser: filters service/sticker/voice/round-video, flattens text-as-list,
  dense `turn_index`, sender/timestamp passthrough.
- Pipeline: end-to-end TG-JSON → CSV → `load_dialog_csv`; softmax for logit
  outputs; passthrough for valid probabilities.
- Schema and plotting tests existed already.
