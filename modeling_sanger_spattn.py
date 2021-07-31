import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F


LOG_LOAD_BALANCE = os.getenv('LOG_LOAD_BALANCE', False)

if LOG_LOAD_BALANCE:
    csv_path = Path('load_balance.csv')
    assert not csv_path.exists(), f'{csv_path} already exists.'
    csv_file = csv_path.open('w')
    csv_file.write('50%-no-skip,50%-skip,25%-no-skip,25%-skip,overall-sparsity\n')


def _eval_load_balance(sparsity_mask, attn_mask, num_ports=64, num_pes=16, no_skip=False):
    # sparsity_mask: bool, [batch_size, num_heads, seq_len, seq_len]
    # attn_mask: bool, [batch_size, 1, seq_len, seq_len]
    batch_size, num_heads, seq_len, seq_len = sparsity_mask.shape
    assert seq_len % num_ports == 0

    # split sparsity mask into `num_ports`-dim vectors
    # sparsity_mask: [batch_size, num_heads, seq_len * seq_len / num_ports, num_ports]
    sparsity_mask = sparsity_mask.view(batch_size, num_heads, -1, num_ports)

    # count nonzeros in each vector
    # num_nonzero: [batch_size, num_heads, seq_len * seq_len / num_ports]
    num_nonzero = sparsity_mask.sum(dim=-1)
    
    # split attention mask into `num_ports`-dim vectors
    # attn_mask: [batch_size, 1, seq_len * seq_len / num_ports, num_ports]
    attn_mask = attn_mask.view(batch_size, 1, -1, num_ports)

    # vector-wise attention mask: mask out vectors that are completely covered by the original attention mask 
    # attn_mask: bool, [batch_size, 1, seq_len * seq_len / num_ports]
    attn_mask = attn_mask.sum(dim=-1).ne(0)
    
    # filter out masked vectors from num_nonzero
    # num_nonzero: 1-D vector
    num_nonzero = torch.masked_select(num_nonzero, attn_mask)
    
    # count and skip all-zero vectors
    skip_mask = num_nonzero.ne(0)
    num_skips = skip_mask.sum()

    # filter out skipped vectors from num_nonzero
    num_nonzero = torch.masked_select(num_nonzero, skip_mask)
    
    # split non-empty vectors into segments with nnz no greater than num_pes
    # assuming num_pes = 3, a vector of length 10 can be divided into four segments [3, 3, 3, 1]
    # in this case, there are three full segments (where all pes are occupied) and one unfull remnant
    num_splits = num_nonzero / num_pes
    num_full_splits = num_splits.floor().sum()
    num_all_splits = num_splits.ceil().sum()
    
    # a full segment leads to a pe utilization of 100%
    # while pe util of a remnant segment is calculated as num-occupied-pes / num-pes
    acc_full_split_utils = num_full_splits * 1.0
    acc_remn_split_utils = num_splits.frac().sum()
    # accumulated pe utilization of all segments
    acc_all_split_utils = acc_full_split_utils + acc_remn_split_utils

    if no_skip:
        pe_util = acc_all_split_utils / (num_all_splits + num_skips)
    else:
        pe_util = acc_all_split_utils / num_all_splits

    return pe_util.item()


def _eval_overall_sparsity(sparsity_mask, attn_mask):
    # sparsity_mask: bool, [batch_size, num_heads, seq_len, seq_len]
    # attn_mask: bool, [batch_size, 1, seq_len, seq_len]
    scaling_factor = attn_mask.mean(dim=(1, 2, 3))
    sparsity_per_seq = (sparsity_mask * attn_mask).mean(dim=(1, 2, 3))
    overall_sparsity = (sparsity_per_seq / scaling_factor).mean().item()
    return overall_sparsity


def gen_sparsity_mask(threshold, attention_scores, attn_mask):
    attention_scores = F.softmax(attention_scores+attn_mask, dim=-1)
    sparsity_mask = attention_scores > threshold
    sparsity_mask = sparsity_mask.type_as(attention_scores)

    if LOG_LOAD_BALANCE and random.random() < 3e-2:
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=False), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=False),
            _eval_overall_sparsity(sparsity_mask, attn_mask)
        ]
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')
    
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    return sparsity_mask.detach()


def quant_qk_matmul(query_layer, key_layer, config, quant_matmul=None):
    assert getattr(config, 'quant_qk', False)
    do_normalize = getattr(config, 'normalize_qk', False)
    if do_normalize:
        assert config.normalize_qk == 'inner_product'
        query_norm = query_layer.norm(dim=-1, keepdim=True)
        key_norm = key_layer.norm(dim=-2, keepdim=True)
        normed_query_layer = query_layer / query_norm
        normed_key_layer = key_layer / key_norm
        quant_attention_scores = quant_matmul(normed_query_layer, normed_key_layer)
        quant_attention_scores *= query_norm * key_norm
    else:
        quant_attention_scores = quant_matmul(query_layer, key_layer)
    return quant_attention_scores


def prune_attn_scores(attn_scores, attn_mask, config):
    assert getattr(config, 'prune_score', False)
    threshold = config.prune_score['threshold']
    sparsity_mask = gen_sparsity_mask(threshold, attn_scores, attn_mask)
    return sparsity_mask


# def sanger_sparse_attention(query_layer, key_layer, attention_mask, config, quant_matmul=None):
#     # query_layer:    [batch_size, num_attention_heads, seq_len, attention_head_size]
#     # key_layer:      [batch_size, num_attention_heads, attention_head_size, seq_len]
#     # attention_mask: [batch_size, num_attention_heads, seq_len, seq_len]

#     do_quant = getattr(config, 'quant_qk', False)
#     do_prune = getattr(config, 'prune_score', False)
    
#     attention_head_size = query_layer.shape[-1]
#     scale_factor = math.sqrt(attention_head_size)

#     attention_scores = torch.matmul(query_layer, key_layer)
#     attention_scores = attention_scores / scale_factor

#     if do_quant:
#         quant_attention_scores = quant_qk_matmul(query_layer, key_layer, config, quant_matmul)
#         quant_attention_scores = quant_attention_scores / scale_factor
#     else:
#         quant_attention_scores = None

#     if do_prune:
#         attn_scores = quant_attention_scores if do_quant else attention_scores 
#         sparsity_mask = prune_attn_scores(attn_scores, attention_mask, config)
#         attention_scores += sparsity_mask

#     attention_scores = attention_scores + attention_mask
#     attention_probs = F.softmax(attention_scores, dim=-1)

#     return attention_probs
