"""
IntelligentTriageEngine
=======================
On-device Zero-Shot Classification triage engine using a lightweight
DistilBERT-based MNLI model via HuggingFace Transformers.

Responsibilities:
  - Load and cache the ZSC pipeline (singleton).
  - Classify free-text witness statements against emergency categories.
  - Compute a math-bounded Urgency Index (1.00 – 10.00).
  - Extract numeric entities (counts of vehicles, casualties, etc.).
"""

from __future__ import annotations

import re
import math
import logging
from functools import lru_cache
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANDIDATE_LABELS: list[str] = [
    "Fire/Explosion",
    "Critical Medical Trauma",
    "Minor Collision",
    "Hazardous Material Leak",
    "Vehicle Entrapment",
]

# Severity multipliers per label (used to bias the Urgency Index)
SEVERITY_WEIGHTS: dict[str, float] = {
    "Fire/Explosion":          1.00,
    "Critical Medical Trauma": 0.95,
    "Vehicle Entrapment":      0.85,
    "Hazardous Material Leak": 0.80,
    "Minor Collision":         0.30,
}

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class TriageResult(NamedTuple):
    top_label: str
    urgency_index: float          # 1.00 – 10.00
    label_scores: dict[str, float]
    entities: dict[str, list[str]]
    recommended_units: list[str]


# ---------------------------------------------------------------------------
# Pipeline loader (module-level singleton via lru_cache)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_pipeline():
    """Load the zero-shot classification pipeline exactly once."""
    # Model candidates in order of preference (lightweight first)
    model_candidates = [
        "typeform/distilbert-base-uncased-mnli",
        "cross-encoder/nli-MiniLM2-L6-H768",
        "facebook/bart-large-mnli",
    ]
    try:
        from transformers import pipeline  # type: ignore
        for model_name in model_candidates:
            try:
                logger.info("Loading ZSC pipeline — %s …", model_name)
                # Build kwargs without 'framework' arg (removed in transformers v5)
                kwargs: dict = {
                    "model": model_name,
                    "device": -1,  # CPU
                }
                try:
                    pipe = pipeline("zero-shot-classification", **kwargs)
                except TypeError:
                    # Older API: try with framework kwarg
                    pipe = pipeline(
                        "zero-shot-classification",
                        model=model_name,
                        device=-1,
                        framework="pt",
                    )
                logger.info("ZSC pipeline loaded: %s", model_name)
                return pipe
            except Exception as model_exc:
                logger.warning("Model %s failed: %s. Trying next …", model_name, model_exc)
                continue
        logger.warning("All model candidates failed. Using heuristic fallback.")
        return None
    except Exception as exc:
        logger.warning("Transformers pipeline unavailable (%s). Using heuristic fallback.", exc)
        return None


# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------

_ENTITY_PATTERNS: dict[str, str] = {
    "vehicle_count": r"""
        (?:
            (?P<digit>\d+)                        # "3 cars"
            |
            (?P<word>one|two|three|four|five|six|seven|eight|nine|ten)  # "two cars"
        )
        \s*
        (?:car|truck|vehicle|bus|van|lorry|SUV|motorcycle|bike)s?
    """,
    "casualty_count": r"""
        (?:
            (?P<digit>\d+)
            |
            (?P<word>one|two|three|four|five|six|seven|eight|nine|ten)
        )
        \s*
        (?:casualt(?:y|ies)|injur(?:ed|ies)|victim|person|people|passenger|survivor)s?
    """,
    "fire_mention":    r"\b(?:fire|flame|burning|ablaze|explosion|blast|smoke)\b",
    "hazmat_mention":  r"\b(?:chemical|gas|leak|spill|fuel|toxic|fumes|hazmat)\b",
    "speed_mention":   r"(?P<speed>\d+)\s*(?:km/h|mph|kph)",
}

_WORD_TO_NUM: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _extract_entities(text: str) -> dict[str, list[str]]:
    """Return a dict of entity type → list of extracted string values."""
    entities: dict[str, list[str]] = {}
    text_lower = text.lower()

    for entity_type, pattern in _ENTITY_PATTERNS.items():
        matches = re.findall(pattern, text_lower, re.VERBOSE | re.IGNORECASE)
        if matches:
            flat: list[str] = []
            for m in matches:
                if isinstance(m, tuple):
                    # named groups come as tuples; take the non-empty one
                    val = next((g for g in m if g), None)
                    if val:
                        flat.append(str(_WORD_TO_NUM.get(val, val)))
                else:
                    flat.append(str(m))
            if flat:
                entities[entity_type] = flat
    return entities


# ---------------------------------------------------------------------------
# Urgency Index computation
# ---------------------------------------------------------------------------

def _compute_urgency(label_scores: dict[str, float], entities: dict[str, list[str]]) -> float:
    """
    Compute an Urgency Index in [1.00, 10.00].

    Formula:
        base = Σ(score_i × severity_weight_i)        for all labels i
        bonus = 0.5 per extracted entity cluster (casualties, fire, hazmat)
        raw   = sigmoid(base × 6 - 3) * 9 + 1
        final = clamp(raw + bonus, 1.0, 10.0)
    """
    base = sum(
        score * SEVERITY_WEIGHTS.get(label, 0.5)
        for label, score in label_scores.items()
    )

    # Sigmoid stretch so mid-confidence → mid urgency
    sigmoid_val = 1.0 / (1.0 + math.exp(-(base * 6.0 - 3.0)))
    raw_urgency = sigmoid_val * 9.0 + 1.0

    # Entity-based bonuses
    bonus = 0.0
    if "casualty_count" in entities:
        try:
            n = int(entities["casualty_count"][0])
            bonus += min(n * 0.3, 1.5)
        except ValueError:
            bonus += 0.3
    if "fire_mention" in entities:
        bonus += 0.5
    if "hazmat_mention" in entities:
        bonus += 0.4

    return round(min(max(raw_urgency + bonus, 1.0), 10.0), 2)


# ---------------------------------------------------------------------------
# Unit recommendation
# ---------------------------------------------------------------------------

def _recommend_units(top_label: str, urgency: float, entities: dict) -> list[str]:
    units: list[str] = []
    if top_label == "Fire/Explosion":
        units = ["🚒 Fire Engine", "🚑 Paramedic Unit", "👮 Police Escort"]
    elif top_label == "Critical Medical Trauma":
        units = ["🚑 Advanced Life Support", "🏥 Trauma Helicopter"]
    elif top_label == "Vehicle Entrapment":
        units = ["🚒 Heavy Rescue Unit", "🚑 Paramedic Unit", "👮 Traffic Police"]
    elif top_label == "Hazardous Material Leak":
        units = ["☣️ HazMat Team", "🚒 Fire Engine", "👮 Road Closure Unit"]
    else:  # Minor Collision
        units = ["🚑 Standard Ambulance", "👮 Police Unit"]

    if urgency >= 8.5 and "🏥 Trauma Helicopter" not in units:
        units.append("🏥 Trauma Helicopter")
    return units


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class IntelligentTriageEngine:
    """
    Zero-Shot classification triage engine with entity extraction
    and urgency scoring.
    """

    def __init__(self) -> None:
        self._pipe = None  # Lazy loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, witness_text: str) -> TriageResult:
        """
        Full triage pipeline on witness_text.

        Returns a TriageResult with scores, urgency index, and entities.
        """
        if not witness_text or not witness_text.strip():
            raise ValueError("Witness text must not be empty.")

        self._ensure_pipeline()

        # Classification
        label_scores = self._classify(witness_text)

        # Top label by score
        top_label = max(label_scores, key=lambda k: label_scores[k])

        # Entity extraction
        entities = _extract_entities(witness_text)

        # Urgency index
        urgency = _compute_urgency(label_scores, entities)

        # Unit recommendation
        units = _recommend_units(top_label, urgency, entities)

        return TriageResult(
            top_label=top_label,
            urgency_index=urgency,
            label_scores=label_scores,
            entities=entities,
            recommended_units=units,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_pipeline(self) -> None:
        if self._pipe is None:
            self._pipe = _load_pipeline()

    def _classify(self, text: str) -> dict[str, float]:
        """Run ZSC pipeline or fall back to heuristics."""
        if self._pipe is not None:
            try:
                result = self._pipe(text, CANDIDATE_LABELS, multi_label=False)
                return dict(zip(result["labels"], result["scores"]))
            except Exception as exc:
                logger.warning("Pipeline inference failed (%s). Using heuristic.", exc)

        return self._heuristic_classify(text)

    @staticmethod
    def _heuristic_classify(text: str) -> dict[str, float]:
        """Keyword-weighted fallback classifier."""
        text_l = text.lower()
        keyword_map: dict[str, list[str]] = {
            "Fire/Explosion":          ["fire", "explosion", "flame", "burning", "ablaze", "blast", "smoke"],
            "Critical Medical Trauma": ["blood", "unconscious", "not breathing", "cardiac", "trauma", "critical", "severe injury"],
            "Minor Collision":         ["fender", "minor", "scratch", "small", "dent", "tap"],
            "Hazardous Material Leak": ["chemical", "gas", "leak", "spill", "hazmat", "toxic", "fuel"],
            "Vehicle Entrapment":      ["trapped", "pinned", "stuck", "entrap", "door", "crushed"],
        }
        raw: dict[str, float] = {}
        for label, keywords in keyword_map.items():
            hits = sum(1 for kw in keywords if kw in text_l)
            raw[label] = float(hits)

        total = sum(raw.values()) or 1.0
        # Softmax-like normalization with small uniform prior
        prior = 0.1
        scores = {label: (raw[label] + prior) / (total + prior * len(raw)) for label in raw}

        # Re-normalize to sum 1
        s_total = sum(scores.values())
        return {label: round(v / s_total, 4) for label, v in scores.items()}
