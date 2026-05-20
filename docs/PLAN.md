# Dialog Emotion Timeline Demo Plan

## Goal

Build a readable Gradio demo for Hugging Face Spaces that tracks emotional color across a dialogue over time.

The interface reads model-like outputs from CSV instead of calling a model directly. The default demo uses synthetic Russian messenger-style data, and users can upload another CSV with the same contract.

## Decisions

- Chunk unit: one dialogue message per CSV row.
- Data mode: bundled synthetic CSV plus optional CSV upload.
- Refresh mode: static viewer, no live polling in v1.
- Probability semantics: six independent probabilities in `[0, 1]`; rows do not need to sum to one.
- Emotions: `joy`, `warmth`, `sadness`, `anger`, `anxiety`, `neutral`.
- UI language: Russian.
- Chunk details: shown inside each message block on hover/focus.

## CSV Contract

Required columns:

- `turn_index`
- `timestamp`
- `sender`
- `text`
- `joy`
- `warmth`
- `sadness`
- `anger`
- `anxiety`
- `neutral`

## Checklist

- [x] Add dependencies and HF Spaces metadata.
- [x] Generate deterministic synthetic dialogue CSV.
- [x] Implement CSV loading and validation.
- [x] Implement smoothing and Plotly figure generation.
- [x] Build Gradio Blocks UI.
- [x] Add focused tests.
- [x] Run local verification.
- [x] Commit changes in logical steps.

## Commit Log

- Done: `docs: add project plan`
- Done: `chore: add gradio dependencies and space config`
- Done: `feat: add csv contract and synthetic dialogue`
- Done: `feat: build emotion timeline ui`
- Done: `test: cover data loading and smoothing`
- Done: `docs: update implementation tracker`

## Verification

- `uv run pytest`
- `uv run python -m compileall app.py main.py dialog_emo_demo scripts tests`
- `uv run python app.py`
- Browser smoke test at `http://127.0.0.1:7860`

## Non-goals For v1

- Model inference inside the UI.
- Automatic CSV refresh or file watching.
- Plot-hover to message-block synchronization.
- Authentication, persistence, or multi-session state.
