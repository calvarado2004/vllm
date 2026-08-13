# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 checkpoints that ship a DSpark drafter must not be routed to MTP.

``deepseek-ai/DeepSeek-V4-Flash-0731`` advertises ``num_nextn_predict_layers: 1``
like every other DeepSeek-V4 config, but the ``mtp.*`` tensors behind it are a
three-stage DSpark drafter with no ``enorm``/``hnorm``/``e_proj``/``h_proj``.
Routing it to ``DeepSeekV4MTPModel`` fails deep inside the weight loader with
``KeyError: model.layers.43.mtp_block.main_norm.weight`` (vllm-project/vllm#52111).
"""

import json
from pathlib import Path
from typing import Any, get_args

import pytest
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig
from vllm.config.speculative import (
    MTPModelTypes,
    SpeculativeConfig,
    ships_dspark_drafter,
)

# Config keys as published in the two checkpoints' config.json.
_DSPARK_KEYS = {
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
    "dspark_noise_token_id": 128799,
}

_CHECKPOINT_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "torch_dtype": "bfloat16",
    "vocab_size": 129280,
    "hidden_size": 4096,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 43,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 1048576,
    "num_nextn_predict_layers": 1,
    "compress_ratios": [0] * 46,
    **_DSPARK_KEYS,
}


def _deepseek_v4_config(**kwargs) -> PretrainedConfig:
    config = PretrainedConfig(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=43,
        num_nextn_predict_layers=1,
        **kwargs,
    )
    config.model_type = "deepseek_v4"
    return config


def _write_checkpoint(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_CHECKPOINT_CONFIG))


def _make_speculative_config(
    checkpoint: Path,
    *,
    method: str | None,
    num_speculative_tokens: int,
    explicit_model: bool = True,
) -> SpeculativeConfig:
    target_model_config = ModelConfig(
        model=str(checkpoint), tokenizer_mode="skip", max_model_len=4096
    )
    kwargs: dict[str, Any] = {
        "num_speculative_tokens": num_speculative_tokens,
        "target_model_config": target_model_config,
        "target_parallel_config": ParallelConfig(),
    }
    if method is not None:
        kwargs["method"] = method
    if explicit_model:
        kwargs["model"] = str(checkpoint)
    return SpeculativeConfig(**kwargs)


def test_dspark_drafter_detected_from_config_keys():
    """DeepSeek-V4-Flash-0731: dspark_* keys mark the mtp.* weights as DSpark."""
    assert ships_dspark_drafter(_deepseek_v4_config(**_DSPARK_KEYS))


def test_plain_mtp_checkpoint_is_not_a_dspark_drafter():
    """DeepSeek-V4-Flash: a real MTP head, no dspark_* keys, must stay on mtp."""
    assert not ships_dspark_drafter(_deepseek_v4_config())


def test_num_nextn_predict_layers_does_not_discriminate():
    """Both checkpoints declare one next-token layer, so it cannot be the signal."""
    dspark = _deepseek_v4_config(**_DSPARK_KEYS)
    plain = _deepseek_v4_config()

    assert dspark.num_nextn_predict_layers == plain.num_nextn_predict_layers
    assert ships_dspark_drafter(dspark) != ships_dspark_drafter(plain)


@pytest.mark.parametrize("model_type", ["deepseek_v3", "deepseek_v32", "qwen3_next"])
def test_other_model_types_are_untouched(model_type):
    """The guard is scoped to deepseek_v4; nothing else changes behaviour."""
    config = _deepseek_v4_config(**_DSPARK_KEYS)
    config.model_type = model_type

    assert not ships_dspark_drafter(config)


def test_detected_after_hf_config_override():
    """The draft path sees the config *after* ``hf_config_override`` has rewritten
    model_type to ``deepseek_mtp``; the dspark_* keys survive, so detection must
    survive with them."""
    config = _deepseek_v4_config(**_DSPARK_KEYS)

    overridden = SpeculativeConfig.hf_config_override(config)

    assert overridden.model_type == "deepseek_mtp"
    assert overridden.architectures == ["DeepSeekV4MTPModel"]
    assert overridden.model_type in get_args(MTPModelTypes)
    assert ships_dspark_drafter(overridden)


def test_plain_mtp_checkpoint_still_routed_to_mtp_after_override():
    """The same override on a real MTP head must not be claimed by DSpark."""
    overridden = SpeculativeConfig.hf_config_override(_deepseek_v4_config())

    assert overridden.architectures == ["DeepSeekV4MTPModel"]
    assert not ships_dspark_drafter(overridden)


def test_explicit_model_and_mtp_method_reject_dspark_checkpoint(tmp_path):
    checkpoint = tmp_path / "deepseek-v4-flash-0731"
    _write_checkpoint(checkpoint)

    with pytest.raises(
        ValueError,
        match=r"ships a DSpark drafter rather than an MTP head.*method='dspark'",
    ):
        _make_speculative_config(checkpoint, method="mtp", num_speculative_tokens=1)


def test_implicit_model_and_mtp_method_reject_dspark_checkpoint(tmp_path):
    checkpoint = tmp_path / "deepseek-v4-flash-0731"
    _write_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="ships a DSpark drafter"):
        _make_speculative_config(
            checkpoint,
            method="mtp",
            num_speculative_tokens=1,
            explicit_model=False,
        )


def test_implicit_method_routes_dspark_checkpoint_by_config(tmp_path):
    checkpoint = tmp_path / "deepseek-v4-flash-0731"
    _write_checkpoint(checkpoint)

    speculative_config = _make_speculative_config(
        checkpoint, method=None, num_speculative_tokens=5
    )

    assert speculative_config.method == "dspark"
    assert speculative_config.parallel_drafting
    assert speculative_config.draft_model_config.architectures == ["DSparkDraftModel"]


def test_explicit_dspark_method_is_preserved(tmp_path):
    checkpoint = tmp_path / "deepseek-v4-flash-0731"
    _write_checkpoint(checkpoint)

    speculative_config = _make_speculative_config(
        checkpoint, method="dspark", num_speculative_tokens=5
    )

    assert speculative_config.method == "dspark"
    assert speculative_config.parallel_drafting
    assert speculative_config.draft_model_config.architectures == ["DSparkDraftModel"]


def test_dspark_method_rejects_proposal_shorter_than_block(tmp_path):
    checkpoint = tmp_path / "deepseek-v4-flash-0731"
    _write_checkpoint(checkpoint)

    with pytest.raises(
        ValueError,
        match=r"num_speculative_tokens >= dspark_block_size \(5\); got 4",
    ):
        _make_speculative_config(
            checkpoint,
            method="dspark",
            num_speculative_tokens=4,
            explicit_model=False,
        )
