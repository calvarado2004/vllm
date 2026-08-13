# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import modelopt
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config


def test_deepseek_v4_mixed_nvfp4_checkpoint_routes_experts_to_modelopt(
    monkeypatch,
):
    """Guard the mixed FP8/NVFP4 metadata used by 0731 checkpoints."""
    quant_config = DeepseekV4FP8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    )
    hf_config = SimpleNamespace(
        expert_dtype="fp4",
        quantization_config={"moe_quant_algo": "NVFP4"},
    )
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))

    routed_experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed_experts)
    routed_experts.moe_config = object()

    expected_method = object()
    nvfp4_factory = MagicMock(return_value=expected_method)
    monkeypatch.setattr(modelopt, "ModelOptNvFp4FusedMoE", nvfp4_factory)

    with set_current_vllm_config(vllm_config):
        method = quant_config.get_quant_method(
            routed_experts, "model.layers.0.ffn.experts"
        )

    assert method is expected_method
    nvfp4_factory.assert_called_once()
    factory_args = nvfp4_factory.call_args.kwargs
    assert factory_args["moe_config"] is routed_experts.moe_config
    assert factory_args["quant_config"].is_checkpoint_nvfp4_serialized
    assert factory_args["quant_config"].group_size == 16
