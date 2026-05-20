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

Gradio demo for tracking an emotional probability distribution across a dialogue:
`joy`, `warmth`, `sadness`, `anger`, `anxiety`, and `neutral`.

The UI reads model-like output from CSV. The bundled sample is synthetic; a real
model can be connected later by writing the same table contract.

## Local run

```bash
uv run python app.py
```

## CSV columns

```text
turn_index,timestamp,sender,text,joy,warmth,sadness,anger,anxiety,neutral
```

Each emotion column is a probability in `[0, 1]`. For each row, the six emotion
probabilities must sum to `1`.
