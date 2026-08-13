# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side helpers for DSpark confidence scheduling."""

from collections.abc import Sequence

import numpy as np


def confidence_prefix_lengths(
    confidence_rows: Sequence[Sequence[float]] | np.ndarray,
    threshold: float,
) -> list[int]:
    """Return the longest cumulative-confidence prefix for every request."""
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")

    confidence = np.asarray(confidence_rows, dtype=np.float32)
    if confidence.ndim != 2:
        raise ValueError(
            "confidence_rows must have shape [num_requests, num_draft_tokens]"
        )
    num_draft_tokens = confidence.shape[1]
    if threshold == 0.0:
        return [num_draft_tokens] * confidence.shape[0]

    valid = np.isfinite(confidence) & (confidence >= 0.0) & (confidence <= 1.0)
    survival = np.cumprod(confidence, axis=1)
    below = survival < threshold
    first_below = below.argmax(axis=1)
    lengths = np.where(below.any(axis=1), first_below, num_draft_tokens)
    # Confidence affects performance only. Preserve fixed-K behavior for a row
    # if the checkpoint unexpectedly produces an invalid probability.
    lengths = np.where(valid.all(axis=1), lengths, num_draft_tokens)
    return lengths.astype(np.int32).tolist()


def map_confidence_prefix_lengths(
    req_ids: Sequence[str],
    confidence_rows: Sequence[Sequence[float]] | np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Bind confidence-derived lengths to the proposal batch's request order."""
    lengths = confidence_prefix_lengths(confidence_rows, threshold)
    if len(req_ids) != len(lengths):
        raise ValueError(
            "req_ids and confidence_rows must contain the same number of requests"
        )
    return dict(zip(req_ids, lengths, strict=True))


def truncate_draft_token_ids(
    draft_token_ids: Sequence[Sequence[int]], lengths: Sequence[int]
) -> list[list[int]]:
    """Keep only each request's confidence-selected draft prefix."""
    if len(draft_token_ids) != len(lengths):
        raise ValueError(
            "draft_token_ids and lengths must contain the same number of requests"
        )
    return [
        list(token_ids[: max(0, min(len(token_ids), int(length)))])
        for token_ids, length in zip(draft_token_ids, lengths, strict=True)
    ]


def clamp_async_draft_length(
    confidence_length: int, current_length: int, max_length: int
) -> int:
    """Clamp an async prefix without re-expanding an existing constraint."""
    return max(0, min(int(confidence_length), current_length, max_length))
