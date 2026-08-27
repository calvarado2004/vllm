# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Dynamic SD batch-size schedule helpers."""

import logging

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.structured_output import StructuredOutputManager

THINK_START_TOKEN_IDS = [1001, 1002]
THINK_END_TOKEN_IDS = [2001, 2002]


def _make_lookup(
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]],
    *,
    max_batch_size: int = 256,
    runtime_num_speculative_tokens: int = 3,
) -> list[int]:
    return build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=max_batch_size,
        vllm_num_speculative_tokens=runtime_num_speculative_tokens,
    )


def _make_scheduler_with_dynamic_sd(
    schedule: list[tuple[int, int, int]],
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    runtime_num_speculative_tokens: int = 3,
    num_speculative_tokens_during_reasoning: int | None = None,
) -> Scheduler:
    base_scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=runtime_num_speculative_tokens,
        num_speculative_tokens_during_reasoning=(
            num_speculative_tokens_during_reasoning
        ),
        reasoning_start_token_ids=THINK_START_TOKEN_IDS,
        reasoning_end_token_ids=THINK_END_TOKEN_IDS,
    )

    speculative_config = base_scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.num_speculative_tokens_per_batch_size = schedule

    return Scheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )


def _add_requests_and_schedule(
    scheduler: Scheduler,
    num_requests: int,
    *,
    num_tokens: int = 10,
    reasoning_ended: bool | None = None,
):
    requests = create_requests(num_requests=num_requests, num_tokens=num_tokens)
    for request in requests:
        request.reasoning_ended = reasoning_ended
        scheduler.add_request(request)
    return scheduler.schedule()


def test_dynamic_sd_uses_batch_size_schedule():
    dynamic_sd_lookup = _make_lookup(
        [
            (1, 16, 3),
            (32, 128, 2),
            (256, 2048, 0),
        ]
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[16] == 3
    assert dynamic_sd_lookup[17] == 3
    assert dynamic_sd_lookup[31] == 3
    assert dynamic_sd_lookup[32] == 2
    assert dynamic_sd_lookup[128] == 2
    assert dynamic_sd_lookup[129] == 2
    assert dynamic_sd_lookup[255] == 2
    assert dynamic_sd_lookup[256] == 0


def test_dynamic_sd_requires_schedule_starting_at_batch_size_one():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_lookup([(2, 16, 3)])


def test_dynamic_sd_clamps_k_to_runtime_max():
    dynamic_sd_lookup = _make_lookup(
        [(1, 256, 4)],
        runtime_num_speculative_tokens=3,
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[256] == 3


def test_dynamic_sd_rejects_invalid_schedule_entry():
    with pytest.raises(ValueError, match="3-item sequence"):
        _make_lookup([(1, 16, 3), (32, 64)])  # type: ignore[list-item]


def test_dynamic_sd_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="non-overlapping and sorted"):
        _make_lookup([(1, 16, 3), (16, 32, 2)])


def test_dynamic_sd_rejects_negative_k():
    with pytest.raises(ValueError, match="values must be >= 0"):
        _make_lookup([(1, 16, -1)])


def test_dynamic_sd_rejects_empty_schedule():
    with pytest.raises(ValueError, match="must not be empty"):
        _make_lookup([])


def test_dynamic_sd_requires_schedule_config():
    with pytest.raises(
        ValueError, match="num_speculative_tokens_per_batch_size is required"
    ):
        build_dynamic_sd_schedule_lookup(
            None,
            vllm_max_batch_size=256,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_lookup_rejects_invalid_batch_size_queries():
    dynamic_sd_lookup = _make_lookup([(1, 256, 3)])

    assert dynamic_sd_lookup[0] == 0
    with pytest.raises(IndexError):
        _ = dynamic_sd_lookup[257]


def test_scheduler_initializes_dynamic_sd_lookup_from_speculative_config():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )

    assert scheduler.dynamic_sd_lookup is not None
    assert scheduler.num_spec_tokens == 3


def test_scheduler_uses_dsd_k_based_on_number_of_scheduled_requests():
    test_cases = [
        (4, 3),
        (64, 2),
        (256, 0),
    ]

    for num_requests, expected_k in test_cases:
        scheduler = _make_scheduler_with_dynamic_sd(
            [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
            max_num_seqs=num_requests,
            max_num_batched_tokens=num_requests * 10,
            runtime_num_speculative_tokens=3,
        )
        output = _add_requests_and_schedule(scheduler, num_requests)

        assert len(output.num_scheduled_tokens) == num_requests
        assert output.num_spec_tokens_to_schedule == expected_k


def test_scheduler_clamps_dsd_k_to_runtime_num_speculative_tokens():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 256, 5)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_falls_back_to_static_k_when_dsd_not_configured():
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=40,
        num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 4)

    assert scheduler.dynamic_sd_lookup is None
    assert output.num_spec_tokens_to_schedule == 3


def test_reasoning_k_cannot_exceed_runtime_max():
    with pytest.raises(
        ValueError,
        match="num_speculative_tokens_during_reasoning must be less than",
    ):
        create_scheduler(
            num_speculative_tokens=3,
            num_speculative_tokens_during_reasoning=4,
            reasoning_start_token_ids=THINK_START_TOKEN_IDS,
            reasoning_end_token_ids=THINK_END_TOKEN_IDS,
        )


def test_scheduler_caps_k_during_reasoning_then_restores_output_k():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 5)],
        runtime_num_speculative_tokens=5,
        num_speculative_tokens_during_reasoning=3,
    )

    reasoning_output = _add_requests_and_schedule(scheduler, 1, reasoning_ended=False)
    assert reasoning_output.num_spec_tokens_to_schedule == 3

    output_scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 5)],
        runtime_num_speculative_tokens=5,
        num_speculative_tokens_during_reasoning=3,
    )
    output = _add_requests_and_schedule(output_scheduler, 1, reasoning_ended=True)
    assert output.num_spec_tokens_to_schedule == 5


def test_scheduler_infers_reasoning_phase_from_prompt_markers():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 5)],
        runtime_num_speculative_tokens=5,
        num_speculative_tokens_during_reasoning=3,
    )
    request = create_requests(num_requests=1, num_tokens=4)[0]
    request.prompt_token_ids = [7, 8, *THINK_START_TOKEN_IDS]
    request._all_token_ids = request.prompt_token_ids.copy()
    scheduler.add_request(request)

    assert request.reasoning_ended is False
    assert scheduler.schedule().num_spec_tokens_to_schedule == 3

    output_scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 5)],
        runtime_num_speculative_tokens=5,
        num_speculative_tokens_during_reasoning=3,
    )
    output_request = create_requests(num_requests=1, num_tokens=5)[0]
    output_request.prompt_token_ids = [
        *THINK_START_TOKEN_IDS,
        8,
        *THINK_END_TOKEN_IDS,
    ]
    output_request._all_token_ids = output_request.prompt_token_ids.copy()
    output_scheduler.add_request(output_request)

    assert output_request.reasoning_ended is True
    assert output_scheduler.schedule().num_spec_tokens_to_schedule == 5


def test_scheduler_uses_conservative_k_for_mixed_reasoning_batch():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 5)],
        runtime_num_speculative_tokens=5,
        num_speculative_tokens_during_reasoning=3,
    )
    requests = create_requests(num_requests=2, num_tokens=10)
    requests[0].reasoning_ended = True
    requests[1].reasoning_ended = False
    for request in requests:
        scheduler.add_request(request)

    assert scheduler.schedule().num_spec_tokens_to_schedule == 3


def test_reasoning_boundary_tokens_update_phase_across_steps():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 5)],
        runtime_num_speculative_tokens=5,
        num_speculative_tokens_during_reasoning=3,
    )
    request = create_requests(num_requests=1, num_tokens=10)[0]
    request.reasoning_ended = False

    scheduler._update_request_with_output(request, [7, *THINK_END_TOKEN_IDS])
    assert request.reasoning_ended is True

    scheduler._update_request_with_output(request, [8, *THINK_START_TOKEN_IDS])
    assert request.reasoning_ended is False


def test_reasoning_k_caps_batch_size_schedule():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 4)],
        runtime_num_speculative_tokens=5,
        num_speculative_tokens_during_reasoning=3,
    )

    output = _add_requests_and_schedule(scheduler, 1, reasoning_ended=False)
    assert output.num_spec_tokens_to_schedule == 3


def test_dynamic_sd_is_disabled_with_data_parallel(caplog_vllm):
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        scheduler = create_scheduler(
            max_num_seqs=256,
            max_num_batched_tokens=2560,
            num_speculative_tokens=3,
            num_speculative_tokens_per_batch_size=[
                (1, 16, 3),
                (64, 128, 2),
                (256, 4096, 0),
            ],
            num_speculative_tokens_during_reasoning=2,
            reasoning_start_token_ids=THINK_START_TOKEN_IDS,
            reasoning_end_token_ids=THINK_END_TOKEN_IDS,
            data_parallel_size=2,
        )

    speculative_config = scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    assert speculative_config.num_speculative_tokens_per_batch_size is None
    assert speculative_config.num_speculative_tokens_during_reasoning is None
    assert scheduler.dynamic_sd_lookup is None
    assert "Dynamic speculative decoding is not supported with data parallelism" in (
        caplog_vllm.text
    )

    output = _add_requests_and_schedule(scheduler, 256)
    assert len(output.num_scheduled_tokens) == 256
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_static_k_when_no_requests_are_scheduled():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )
    output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 0
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_rejects_bad_dsd_config_at_construction():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_scheduler_with_dynamic_sd([(2, 16, 3)])


def test_scheduler_passes_max_num_seqs_as_dsd_runtime_batch_limit():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert scheduler.dynamic_sd_lookup is not None
    assert len(scheduler.dynamic_sd_lookup) == 17
    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3
