# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sm120_flatten_uses_device_query_lengths_when_cpu_rows_are_uniform():
    """The CPU scheduler preserves only the total draft budget.

    A uniform CPU allocation [4, 4] can describe a device allocation [5, 3].
    SM120 must take the device-driven expansion even though the CPU rows look
    eligible for the old uniform fast path.
    """
    device = torch.device("cuda")
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.supports_varlen = False
    builder.adaptive_device_query_lens = True
    builder.decode_seq_lens_buffer = torch.zeros(16, dtype=torch.int32, device=device)
    builder.expanded_block_table_buffer = torch.zeros(
        (16, 3), dtype=torch.int32, device=device
    )
    builder.decode_lens_buffer = torch.zeros(16, dtype=torch.int32, device=device)
    builder.arange_buffer = torch.arange(16, dtype=torch.int32, device=device)

    seq_lens = torch.tensor([105, 203], dtype=torch.int32, device=device)
    block_table = torch.tensor([[11, 12, 13], [21, 22, 23]], device=device)
    device_decode_lens = torch.tensor([5, 3], dtype=torch.int32, device=device)
    cpu_decode_lens = torch.tensor([4, 4], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 5], dtype=torch.int32, device=device)

    out_seq_lens, out_blocks, out_decode_lens, batch_size, requires_padding = (
        builder._prepare_decode_tensors(
            seq_lens=seq_lens,
            block_table=block_table,
            decode_lens=device_decode_lens,
            decode_lens_cpu=cpu_decode_lens,
            query_start_loc=query_start_loc,
            num_decodes=2,
            num_decode_tokens=8,
            use_native=False,
            next_n=6,
            max_decode_len=4,
        )
    )

    assert out_seq_lens.tolist() == [101, 102, 103, 104, 105, 201, 202, 203]
    assert out_blocks.tolist() == [[11, 12, 13]] * 5 + [[21, 22, 23]] * 3
    assert out_decode_lens.tolist() == [1] * 8
    assert batch_size == 8
    assert not requires_padding
