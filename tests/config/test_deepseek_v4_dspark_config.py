# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 checkpoints that ship a DSpark drafter must not be routed to MTP.

``deepseek-ai/DeepSeek-V4-Flash-0731`` advertises ``num_nextn_predict_layers: 1``
like every other DeepSeek-V4 config, but the ``mtp.*`` tensors behind it are a
three-stage DSpark drafter with no ``enorm``/``hnorm``/``e_proj``/``h_proj``.
Routing it to ``DeepSeekV4MTPModel`` fails deep inside the weight loader with
``KeyError: model.layers.43.mtp_block.main_norm.weight`` (vllm-project/vllm#52111).
"""

from typing import get_args

import pytest
from transformers import PretrainedConfig

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


def _deepseek_v4_config(**kwargs) -> PretrainedConfig:
    config = PretrainedConfig(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=43,
        num_nextn_predict_layers=1,
        **kwargs,
    )
    config.model_type = "deepseek_v4"
    return config


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
