# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.spec_decode.dspark_confidence import (
    clamp_async_draft_length,
    confidence_prefix_lengths,
    map_confidence_prefix_lengths,
    truncate_draft_token_ids,
)


def test_confidence_prefix_uses_cumulative_survival():
    confidence = np.array(
        [
            [0.9, 0.9, 0.9, 0.9, 0.9],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.05, 0.9, 0.9, 0.9, 0.9],
        ],
        dtype=np.float32,
    )

    assert confidence_prefix_lengths(confidence, 0.1) == [5, 3, 0]
    assert confidence_prefix_lengths(confidence, 0.0) == [5, 5, 5]


def test_draft_lists_are_truncated_to_per_request_prefixes():
    drafts = [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31]]

    assert truncate_draft_token_ids(drafts, [5, 2, 0]) == [
        [10, 11, 12, 13, 14],
        [20, 21],
        [],
    ]


def test_async_lengths_remain_bound_to_proposal_request_order():
    req_ids = ["request-c", "request-a", "request-b"]
    confidence = [
        [0.9, 0.9, 0.9, 0.9, 0.9],
        [0.05, 0.9, 0.9, 0.9, 0.9],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ]

    assert map_confidence_prefix_lengths(req_ids, confidence, 0.1) == {
        "request-c": 5,
        "request-a": 0,
        "request-b": 3,
    }


def test_async_confidence_cannot_reexpand_grammar_prefix():
    assert clamp_async_draft_length(4, current_length=2, max_length=5) == 2
    assert clamp_async_draft_length(1, current_length=2, max_length=5) == 1
