# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.v1.worker.adaptive_verification_profile_cache import (
    PROFILE_CACHE_SCHEMA_VERSION,
    _cache_path,
    _device_total_memory_mib,
    _digest,
    build_profile_cache_factors,
    load_profile_cache,
    profile_cache_fingerprint,
    save_profile_cache,
    sentinel_batches,
    validate_profile_sentinel,
)


def test_device_total_memory_falls_back_for_unified_memory(monkeypatch) -> None:
    def unsupported() -> int:
        raise RuntimeError("Not Supported")

    monkeypatch.setattr(
        "vllm.v1.worker.adaptive_verification_profile_cache."
        "current_platform.get_device_total_memory",
        unsupported,
    )
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda _device: SimpleNamespace(total_memory=130663235584),
    )

    assert _device_total_memory_mib() == 124610


def test_profile_factors_flatten_attention_groups(monkeypatch) -> None:
    backend = SimpleNamespace(get_name=lambda: "TEST")
    group = SimpleNamespace(backend=backend, kv_cache_spec={"block_size": 16})
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            model="remote/model",
            hf_config=SimpleNamespace(
                _commit_hash="resolved", to_dict=lambda: {"model_type": "test"}
            ),
            revision=None,
            code_revision=None,
            tokenizer_revision=None,
            max_model_len=1024,
        ),
        vllm_config=SimpleNamespace(
            compute_hash=lambda: "vllm-hash",
            kernel_config=SimpleNamespace(moe_backend="auto", linear_backend="auto"),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(cudagraph_mode="FULL"),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        attn_groups=[[group]],
        kv_cache_config=None,
        kv_cache_dtype="auto",
        model=None,
        get_draft_model=lambda: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.adaptive_verification_profile_cache._device_total_memory_mib",
        lambda: 1024,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.adaptive_verification_profile_cache."
        "current_platform.get_device_capability",
        lambda: "12.1",
    )
    monkeypatch.setattr(
        "vllm.v1.worker.adaptive_verification_profile_cache."
        "current_platform.get_device_name",
        lambda: "TEST GPU",
    )

    factors = build_profile_cache_factors(runner, [1, 8])

    assert factors["backends"]["attention"] == ["TEST"]
    assert factors["backends"]["kv"] == [{"block_size": 16}]


def _factors() -> dict:
    return {
        "schema": PROFILE_CACHE_SCHEMA_VERSION,
        "model": {"revision": "abc"},
        "hardware": {"sm": "12.1"},
        "profile": {"k": 5, "capture_sizes": [1, 8]},
    }


def _curves() -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    return [(1, 0.5), (4, 0.8)], [(1, 1.25), (8, 2.5)]


def test_profile_cache_exact_round_trip(tmp_path) -> None:
    factors = _factors()
    curves = _curves()

    save_profile_cache(factors, curves, str(tmp_path))

    assert load_profile_cache(factors, str(tmp_path)) == curves
    changed = {**factors, "profile": {**factors["profile"], "k": 4}}
    assert load_profile_cache(changed, str(tmp_path)) is None


def test_profile_cache_rejects_corruption(tmp_path) -> None:
    factors = _factors()
    save_profile_cache(factors, _curves(), str(tmp_path))
    path = _cache_path(profile_cache_fingerprint(factors), str(tmp_path))
    with open(path) as file:
        payload = json.load(file)
    payload["draft_curve"][0][1] = 999.0
    with open(path, "w") as file:
        json.dump(payload, file)

    assert load_profile_cache(factors, str(tmp_path)) is None


def test_profile_cache_rejects_other_schema_version(tmp_path) -> None:
    factors = _factors()
    save_profile_cache(factors, _curves(), str(tmp_path))
    path = _cache_path(profile_cache_fingerprint(factors), str(tmp_path))
    with open(path) as file:
        payload = json.load(file)
    payload.pop("checksum")
    payload["schema"] = PROFILE_CACHE_SCHEMA_VERSION + 1
    payload["checksum"] = _digest(payload)
    with open(path, "w") as file:
        json.dump(payload, file)

    assert load_profile_cache(factors, str(tmp_path)) is None


def test_profile_sentinel_requires_matching_graph_points(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", 4096
    )
    assert sentinel_batches([1, 8]) == [
        {"num_tokens": 8, "context_len": 4096},
        {"num_tokens": 8, "context_len": 4096},
    ]
    samples = [
        SimpleNamespace(
            forward_ms=2.0,
            drafter_ms=0.75,
            num_target_tokens=8,
            num_reqs=4,
            full_cudagraph=True,
        ),
        SimpleNamespace(
            forward_ms=2.2,
            drafter_ms=0.70,
            num_target_tokens=8,
            num_reqs=4,
            full_cudagraph=True,
        ),
    ]
    assert validate_profile_sentinel(samples, _curves())
    assert not validate_profile_sentinel(
        [*samples[:1], SimpleNamespace(**{**vars(samples[1]), "forward_ms": 9.0})],
        _curves(),
    )
    assert not validate_profile_sentinel(
        [
            *samples[:1],
            SimpleNamespace(**{**vars(samples[1]), "full_cudagraph": False}),
        ],
        _curves(),
    )
