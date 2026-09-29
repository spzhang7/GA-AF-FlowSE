"""Deterministic, disjoint calibration-condition selection."""

from __future__ import annotations

import random
from typing import Mapping


def select_conditions(
    manifest: Mapping[str, str], *, count: int, seed: int, offset: int = 0
) -> list[str]:
    """Return one contiguous block of a seeded manifest permutation."""

    if count < 1 or offset < 0 or offset + count > len(manifest):
        raise ValueError("invalid calibration condition count or offset")
    utterances = sorted(str(utterance) for utterance in manifest)
    random.Random(seed).shuffle(utterances)
    return utterances[offset : offset + count]
