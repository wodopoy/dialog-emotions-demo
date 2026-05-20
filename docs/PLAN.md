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

- [ ] Add dependencies and HF Spaces metadata.
- [ ] Generate deterministic synthetic dialogue CSV.
- [ ] Implement CSV loading and validation.
- [ ] Implement smoothing and Plotly figure generation.
- [ ] Build Gradio Blocks UI.
- [ ] Add focused tests.
- [ ] Run local verification.
- [ ] Commit changes in logical steps.

## Commit Log

- Planned: `docs: add project plan`
- Planned: `chore: add gradio dependencies and space config`
- Planned: `feat: add csv contract and synthetic dialogue`
- Planned: `feat: build emotion timeline ui`
- Planned: `test: cover data loading and smoothing`

## Non-goals For v1

- Model inference inside the UI.
- Automatic CSV refresh or file watching.
- Plot-hover to message-block synchronization.
- Authentication, persistence, or multi-session state.
