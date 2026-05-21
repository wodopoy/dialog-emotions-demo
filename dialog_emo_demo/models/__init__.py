from __future__ import annotations

from abc import ABC, abstractmethod

from dialog_emo_demo.schema import EMOTION_GROUPS


class EmotionModel(ABC):
    """Override `predict_proba` to return a dict with all six keys from
    `EMOTION_GROUPS`. Values are up to you (probabilities, logits, scores) —
    whatever your downstream code expects."""

    @abstractmethod
    def predict_proba(self, text: str) -> dict[str, float]:
        ...


__all__ = ["EmotionModel", "EMOTION_GROUPS"]
