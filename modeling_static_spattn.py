"""
https://github.com/microsoft/DeepSpeed
deepspeed/ops/sparse_attention/sparsity_config.py
"""

import json
import random

import torch


MAX_SEQ_LENGTH = 512


def setup_layout(num_heads, max_position, block):
    if max_position % block != 0:
        raise ValueError(
            f"Sequence Length, {max_position}, needs to be dividable by Block size {block}!"
        )
    num_blocks = max_position // block
    layout = torch.zeros((num_heads, num_blocks, num_blocks), dtype=torch.int64)
    return layout


def build_dense_pattern(num_heads, max_position, **unused_kwargs):
    """Initialize the Dense Sparsity Pattern Config.
    In reality, this is not sparse and all blocks are used. We keep it for the sake of comparison and comprehension.

    Arguments:
            num_heads: required: an integer determining number of attention heads of the layer.
            seq_len: required: an integer determining number of attention heads of the layer.
            different_layout_per_head: optional: this is just for the sake of consistency with other sparsity formats; can ignore it for DenseSparsityConfig
    """
    return torch.ones(num_heads, max_position, max_position)


def build_fixed_pattern(
    num_heads,
    max_position,
    block=16,
    num_local_blocks=4,
    num_global_blocks=1,
    attention="bidirectional",
    horizontal_global_attention=False,
    num_different_global_patterns=1,
    **unused_kwargs,
):
    """Initialize `Fixed` Sparsity Pattern Config.

    For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

    Arguments:
            num_heads: required: an integer determining number of attention heads of the layer.
            block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
            different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability.
            num_local_blocks: optional: an integer determining the number of blocks in local attention window.
            num_global_blocks: optional: an integer determining how many consecutive blocks in a local window is used as the representative of the window for global attention.
            attention: optional: a string determining attention type. Attention can be `unidirectional`, such as autoregressive models, in which tokens attend only to tokens appear before them in the context. Considering that, the upper triangular of attention matrix is empty as above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to any other tokens before or after them. Then, the upper triangular part of the attention matrix is mirror of the lower triangular in the above figure.
            horizontal_global_attention: optional: a boolean determining if blocks that are global representative of a local window, also attend to all other blocks. This is valid only if attention type is `bidirectional`. Looking at the attention matrix, that means global attention not only includes the vertical blocks, but also horizontal blocks.
            num_different_global_patterns: optional: an integer determining number of different global attentions layouts. While global attention can be fixed by which block/s are representative of any local window, since there are multi-heads, each head can use a different global representative. For example, with 4 blocks local window and global attention size of 1 block, we can have 4 different versions in which the first, Second, third, or forth block of each local window can be global representative of that window. This parameter determines how many of such patterns we want. Of course, there is a limitation based on num_local_blocks and num_global_blocks.
    """

    if num_local_blocks % num_global_blocks != 0:
        raise ValueError(
            f"Number of blocks in a local window, {num_local_blocks}, must be dividable by number of global blocks, {num_global_blocks}!"
        )

    if attention != "unidirectional" and attention != "bidirectional":
        raise NotImplementedError(
            'only "uni/bi-directional" attentions are supported for now!'
        )

    if attention != "bidirectional" and horizontal_global_attention:
        raise ValueError(
            'only "bi-directional" attentions can support horizontal global attention!'
        )

    if num_different_global_patterns > (num_local_blocks // num_global_blocks):
        raise ValueError(
            f"Number of layout versions (num_different_global_patterns), {num_different_global_patterns}, cannot be larger than number of local window blocks divided by number of global blocks, {num_local_blocks} / {num_global_blocks} = {num_local_blocks//num_global_blocks}!"
        )

    def set_local_layout(h, layout):
        """Sets local attantion layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head; may not be completly set at this step

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head in which local layout is set
        """

        num_blocks = layout.shape[1]
        for i in range(0, num_blocks, num_local_blocks):
            end = min(i + num_local_blocks, num_blocks)
            for row in range(i, end):
                for col in range(
                    i, (row + 1 if attention == "unidirectional" else end)
                ):
                    layout[h, row, col] = 1
        return layout

    def set_global_layout(h, layout):
        """Sets global attantion layout used by the given head in the sparse attention.

        Currently we set global blocks starting from the last block of a local window to the first one. That means if a local window consists of 4 blocks and global attention size is one block, we use block #4 in each local window as global. If we have different layout per head, then other heads will get #3, #2, and #1. And if we have more heads (and different layout has set) than num of global attentions, multiple head may have same global attentions.
        Note) if horizontal_global_attention is set, global blocks will be set both horizontally and vertically.

        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head; may not be completly set at this step

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head in which global layout is set
        """

        num_blocks = layout.shape[1]
        first_global_block_idx = (
            num_local_blocks
            - (1 + h % num_different_global_patterns) * num_global_blocks
        )

        # set all global blocks except the last one if (in last local window)
        end = num_blocks - (num_blocks % num_local_blocks)
        for i in range(first_global_block_idx, end, num_local_blocks):

            # vertical global attention
            first_row = 0 if attention == "bidirectional" else i
            # (((i // self.num_local_blocks) + 1) * self.num_local_blocks)
            # if (first_row < num_blocks):
            layout[h, first_row:, i : i + num_global_blocks] = 1

            # horizontal global attention; only in bidirectional attention
            if horizontal_global_attention:
                layout[h, i : i + num_global_blocks, :] = 1

        # set last global blocks; handle possible short last local window
        if end < num_blocks:
            start = min(end + first_global_block_idx, num_blocks - num_global_blocks)
            end = start + num_global_blocks

            # vertical global attention
            first_row = 0 if attention == "bidirectional" else start
            # (((start // self.num_local_blocks) + 1) * self.num_local_blocks)
            # if (first_row < num_blocks):
            layout[h, first_row:, start:end] = 1

            # horizontal global attention
            if horizontal_global_attention:
                layout[h, start:end, :] = 1
        return layout

    layout = setup_layout(num_heads, max_position, block)
    for h in range(0, num_heads):
        layout = set_local_layout(h, layout)
        layout = set_global_layout(h, layout)

    num_blocks = layout.shape[1]
    full_layout = layout.new_zeros(num_heads, num_blocks, block, num_blocks, block)
    full_layout[:, :, :, :, :] = layout[:, :, None, :, None]
    full_layout = full_layout.reshape(num_heads, max_position, max_position)
    return full_layout


def build_longformer_pattern(
    num_heads,
    max_position,
    block=16,
    num_sliding_window_blocks=3,
    global_block_indices=[0],
    global_block_end_indices=None,
    **unused_kwargs,
):
    """Initialize the edited `Longformer` Sparsity Pattern Config.

    For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

    Arguments:
            num_heads: required: an integer determining number of attention heads of the layer.
            block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
            different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability.

            num_sliding_window_blocks: optional: an integer determining the number of blocks in sliding local attention window.
            global_block_indices: optional: a list of integers determining which blocks are considered as global attention. Given indices, determine the blocks that all other token blocks attend to and they attend to all other token blocks. Default value is only index 0. Notice that if global_block_end_indices parameter is set, this parameter is used as starting index of each global window.
            global_block_end_indices: optional: a list of integers determining end indices of global window blocks. By default this is not used. But if it is set, it must have the same size of global_block_indices parameter, and combining this two parameters, for each index i, blocks from global_block_indices[i] to global_block_end_indices[i] (exclusive) are considered as global attention.
    """

    if global_block_end_indices is not None:
        if len(global_block_indices) != len(global_block_end_indices):
            raise ValueError(
                f"Global block start indices length, {len(global_block_indices)}, must be same as global block end indices length, {len(global_block_end_indices)}!"
            )
        for _, (start_idx, end_idx) in enumerate(
            zip(global_block_indices, global_block_end_indices)
        ):
            if start_idx >= end_idx:
                raise ValueError(
                    f"Global block start index, {start_idx}, must be smaller than global block end index, {end_idx}!"
                )

    def set_sliding_window_layout(h, layout):
        num_blocks = layout.shape[1]
        if num_blocks < num_sliding_window_blocks:
            raise ValueError(
                f"Number of sliding window blocks, {num_sliding_window_blocks}, must be smaller than overal number of blocks in a row, {num_blocks}!"
            )

        w = num_sliding_window_blocks // 2
        for row in range(0, num_blocks):
            start = max(0, row - w)
            end = min(row + w + 1, num_blocks)
            layout[h, row, start:end] = 1
        return layout

    def set_global_layout(h, layout):
        num_blocks = layout.shape[1]
        if global_block_end_indices is None:
            for idx in global_block_indices:
                if idx < num_blocks:
                    layout[h, idx, :] = 1
                    layout[h, :, idx] = 1
        else:
            for _, (start_idx, end_idx) in enumerate(
                zip(global_block_indices, global_block_end_indices)
            ):
                if start_idx < num_blocks:
                    end_idx = min(end_idx, num_blocks)
                    layout[h, start_idx:end_idx, :] = 1
                    layout[h, :, start_idx:end_idx] = 1
        return layout

    layout = setup_layout(num_heads, max_position, block)

    for h in range(0, num_heads):
        layout = set_sliding_window_layout(h, layout)
        layout = set_global_layout(h, layout)

    num_blocks = layout.shape[1]
    full_layout = layout.new_zeros(num_heads, num_blocks, block, num_blocks, block)
    full_layout[:, :, :, :, :] = layout[:, :, None, :, None]
    full_layout = full_layout.reshape(num_heads, max_position, max_position)
    return full_layout


def build_bigbird_pattern(
    num_heads,
    max_position,
    block=16,
    num_random_blocks=1,
    num_sliding_window_blocks=3,
    num_global_blocks=1,
    **unused_kwargs,
):
    """Initialize the BigBird Sparsity Pattern Config.

    For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

    Arguments:
            num_heads: required: an integer determining number of attention heads of the layer.
            block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
            different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability.
            num_random_blocks: optional: an integer determining the number of random blocks in each block row.
            num_sliding_window_blocks: optional: an integer determining the number of blocks in sliding local attention window.
            num_global_blocks: optional: an integer determining how many consecutive blocks, starting from index 0, are considered as global attention. Global block tokens will be attended by all other block tokens and will attend to all other block tokens as well.
    """

    def set_random_layout(h, layout):
        """Sets random attantion layout used by the given head in the sparse attention.
        Note) By default, it assumes there will be a unique random block layout for all heads; unless `different_layout_per_head` parameter is set in which each head can have a different random layout.

        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head; may not be completly set at this step

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head in which random layout is set
        """

        num_blocks = layout.shape[1]
        if num_blocks < num_random_blocks:
            raise ValueError(
                f"Number of random blocks, {num_random_blocks}, must be smaller than overal number of blocks in a row, {num_blocks}!"
            )

        for row in range(0, num_blocks):
            rnd_cols = random.sample(range(0, num_blocks), num_random_blocks)
            layout[h, row, rnd_cols] = 1
        return layout

    def set_sliding_window_layout(h, layout):
        """Sets sliding local attantion layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head; may not be completly set at this step

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head in which local sliding window layout is set
        """

        num_blocks = layout.shape[1]
        if num_blocks < num_sliding_window_blocks:
            raise ValueError(
                f"Number of sliding window blocks, {num_sliding_window_blocks}, must be smaller than overal number of blocks in a row, {num_blocks}!"
            )

        w = num_sliding_window_blocks // 2
        for row in range(0, num_blocks):
            start = max(0, row - w)
            end = min(row + w + 1, num_blocks)
            layout[h, row, start:end] = 1
        return layout

    def set_global_layout_itc(h, layout):
        """Sets global attantion layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head; may not be completly set at this step

        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout of all head in which global layout is set
        """

        num_blocks = layout.shape[1]
        if num_blocks < num_global_blocks:
            raise ValueError(
                f"Number of global blocks, {num_global_blocks}, must be smaller than overal number of blocks in a row, {num_blocks}!"
            )

        layout[h, 0:num_global_blocks, :] = 1
        layout[h, :, 0:num_global_blocks] = 1
        return layout

    layout = setup_layout(num_heads, max_position, block)
    for h in range(0, num_heads):
        layout = set_random_layout(h, layout)
        layout = set_sliding_window_layout(h, layout)
        layout = set_global_layout_itc(h, layout)

    num_blocks = layout.shape[1]
    full_layout = layout.new_zeros(num_heads, num_blocks, block, num_blocks, block)
    full_layout[:, :, :, :, :] = layout[:, :, None, :, None]
    full_layout = full_layout.reshape(num_heads, max_position, max_position)
    return full_layout


def build_block_structure_random_pattern(
    num_heads,
    max_position,
    block_shape=(64, 128),
    pe_array_shape=(32, 32),
    different_layout_per_head=False,
    **unused_kwargs,
):
    if max_position % block_shape[0] != 0 or max_position % block_shape[1] != 0:
        raise ValueError(
            f"Sequence length, {max_position}, must be dividable by block size, {block_shape}!"
        )
    
    if pe_array_shape[0] > block_shape[0] or pe_array_shape[1] * 2 > block_shape[1]:
        raise ValueError(
            f"PE Array shape, {pe_array_shape}, must be smaller than half block, {(block_shape[0], block_shape[1] // 2)}!"
        )

    def set_block_layout(h, layout):
        # layout: int64 [num_heads, max_pos, max_pos]
        unstru_mask = torch.zeros(pe_array_shape[0], pe_array_shape[1] * 2, dtype=torch.int64)
        unstru_mask[:, :pe_array_shape[1]] = 1
        unstru_mask = unstru_mask[
            torch.arange(unstru_mask.shape[0]).unsqueeze(-1), 
            torch.argsort(torch.rand(*unstru_mask.shape), dim=-1)
        ]
        layout[h, :pe_array_shape[0], :pe_array_shape[1] * 2] = unstru_mask
        layout[h, :, :] = layout[
            h,
            torch.argsort(torch.rand(layout.shape[1])).unsqueeze(-1), 
            torch.argsort(torch.rand(layout.shape[2])).unsqueeze(0)
        ]
        return layout

    layout = setup_layout(num_heads, max_position, block=1)
    block_rows, block_cols = max_position//block_shape[0], max_position//block_shape[1]
    layout = layout.reshape(num_heads, block_rows, block_cols, *block_shape)

    if different_layout_per_head:
        for h in range(0, num_heads):
            for r in range(block_rows):
                for c in range(block_cols):
                    set_block_layout(h, layout[:, r, c])
    else:
        for r in range(block_rows):
            for c in range(block_cols):
                set_block_layout(0, layout[:, r, c])
        
        layout[1:, :, :] = layout[0, :, :]

    layout = layout.permute(0, 1, 3, 2, 4).reshape(num_heads, max_position, max_position)
    return layout


ATTN_MASK_BUILDERS = {
    "DenseSparsityConfig": build_dense_pattern,
    "FixedSparsityConfig": build_fixed_pattern,
    "BSLongformerSparsityConfig": build_longformer_pattern,
    "BigBirdSparsityConfig": build_bigbird_pattern,
    "VariableSparsityConfig": None,
    "BlockStructuredRandomSparsityConfig": build_block_structure_random_pattern,
}


def build_static_sparsity_mask(
    json_file, num_heads, max_position=MAX_SEQ_LENGTH
) -> torch.Tensor:
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    cfg_json: dict = json.loads(text)
    cfg_cls_name = cfg_json.pop("class")
    print(f"Loaded {cfg_cls_name} from {json_file}:\n{cfg_json}")

    sp_attn_builder = ATTN_MASK_BUILDERS.get(cfg_cls_name)
    assert (
        sp_attn_builder is not None
    ), f"Cannot find AttnMaskBuilder named {cfg_cls_name}"
    sparsity_mask = sp_attn_builder(num_heads, max_position, **cfg_json)
    return sparsity_mask


def apply_static_sparsity_mask(sparsity_mask, attention_scores):
    # sparsity_mask: [num_attention_heads, max_pos, max_pos] (max_pos >= tgt_len, src_len)
    # attention_scores: [batch_size, num_attention_heads, tgt_len, src_len]
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    sparsity_mask = sparsity_mask.type_as(attention_scores)
    tgt_len, src_len = attention_scores.shape[-2:]
    attention_scores += sparsity_mask[:, :tgt_len, :src_len]


if __name__ == "__main__":
    # attn_mask = build_static_sparsity_mask('big_bird_sparsity_config.json', max_position=8)
    # print(attn_mask.shape)
    # print(attn_mask[0])

    # attn_mask = build_static_sparsity_mask('longformer_sparsity_config.json', max_position=8)
    # print(attn_mask.shape)
    # print(attn_mask[0])

    # attn_mask = build_static_sparsity_mask('fixed_sparsity_config.json', max_position=8)
    # print(attn_mask.shape)
    # print(attn_mask[0])

    # # config: "block_shape": [4, 4], "pe_array_shape": [2, 2] 
    # attn_mask = build_static_sparsity_mask('block_structured_random_sparsity_config.json', max_position=16)
    # print(attn_mask.shape)
    # print(attn_mask[0])
    pass
