import time
import math
import argparse
from typing import Optional, Tuple
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertSelfAttention(nn.Module):
	def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0.1):
		super(BertSelfAttention, self).__init__()
		if hidden_size % num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (hidden_size, num_attention_heads))
		self.num_attention_heads = num_attention_heads
		self.attention_head_size = int(hidden_size / num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		
		self.query = nn.Linear(hidden_size, self.all_head_size)
		self.key = nn.Linear(hidden_size, self.all_head_size)
		self.value = nn.Linear(hidden_size, self.all_head_size)
		self.dense = nn.Linear(hidden_size, hidden_size)
		
		self.dropout = nn.Dropout(attention_probs_dropout_prob)
		
	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = torch.reshape(x, new_x_shape)
		return x.permute(0, 2, 1, 3)
	
	def transpose_key_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = torch.reshape(x, new_x_shape)
		return x.permute(0, 2, 3, 1)
	
	def forward(self, hidden_states, attention_mask):
		# assume attention_mask: [batch_size, um_attention_heads, seq_len, seq_len]
		
		# hidden_states: [batch_size, seq_len, config.hidden_size]
		# mixed_*_layer: [batch_size, seq_len, num_attention_heads * attention_head_size = config.hidden_size]
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)
		
		# {q,v}_layer: [batch_size, num_attention_heads, seq_len, attention_head_size]
		# key_layer:   [batch_size, num_attention_heads, attention_head_size, seq_len]
		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_key_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)
			
		# Take the dot product between "query" and "key" to get the raw attention scores.
		# attention_scores: [batch_size, num_attention_heads, seq_len, seq_len]
		attention_scores = torch.matmul(query_layer, key_layer)
			
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
			
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + attention_mask
		
		# Normalize the attention scores to probabilities.
		attention_probs = F.softmax(attention_scores, dim=-1)
			
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)
		
		# context_layer: [batch_size, num_attention_heads, seq_len, attention_head_size]
		context_layer = torch.matmul(attention_probs, value_layer)
		# context_layer: [batch_size, seq_len, num_attention_heads, attention_head_size]
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		# context_layer: [batch_size, seq_len, num_attention_heads * attention_head_size = config.hidden_size]
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

		context_layer = torch.reshape(context_layer, new_context_layer_shape)
		context_layer = self.dense(context_layer)
		return context_layer


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class GPT2SelfAttention(nn.Module):
    def __init__(self, nx, n_ctx, n_head, attn_pdrop, resid_pdrop, scale=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        # if only "normal" attention layer implements causal mask
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        outputs = torch.matmul(w, v)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        a = self._attn(query, key, value, attention_mask)

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        return a


class BartSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output


def build_bert_model_and_input(batch_size=1, seq_len=512, use_large=False, cuda=True, fp16=False):
    if not use_large:
        hidden_size, num_heads = 768, 12
    else:
        hidden_size, num_heads = 1024, 16

    model = BertSelfAttention(hidden_size, num_heads).eval()
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)

    if fp16:
        model = model.half()
        hidden_state = hidden_state.half()

    attn_mask = torch.zeros(batch_size, 1, 1, seq_len).long()
    
    if cuda:
        model = model.cuda()
        hidden_state = hidden_state.cuda()
        attn_mask = attn_mask.cuda()
    
    return model, (hidden_state, attn_mask)


def build_gpt2_model_and_input(batch_size=1, seq_len=512, use_large=False, cuda=True, fp16=False):
    attn_pdrop, resid_pdrop, scale = 0.1, 0.1, True
    if not use_large:
        n_embed, n_ctx, n_head = 768, 1024, 12
    else:
        n_embed, n_ctx, n_head = 1024, 1024, 16

    model = GPT2SelfAttention(n_embed, n_ctx, n_head, attn_pdrop, resid_pdrop, scale)
    hidden_state = torch.randn(batch_size, seq_len, n_embed)

    if fp16:
        model = model.half()
        hidden_state = hidden_state.half()

    attn_mask = torch.zeros(batch_size, 1, 1, seq_len).long()

    if cuda:
        model = model.cuda()
        hidden_state = hidden_state.cuda()
        attn_mask = attn_mask.cuda()

    return model, (hidden_state, attn_mask)


def build_bart_model_and_input(batch_size=1, seq_len=512, use_large=False, cuda=True, fp16=False):
    if not use_large:
        # d_model, encoder_attention_heads, attention_dropout
        embed_dim, num_heads, dropout = 768, 12, 0.1
    else:
        embed_dim, num_heads, dropout = 1024, 16, 0.1

    model = BartSelfAttention(embed_dim, num_heads, dropout)
    hidden_state = torch.randn(batch_size, seq_len, embed_dim)

    if fp16:
        model = model.half()
        hidden_state = hidden_state.half()

    if cuda:
        model = model.cuda()
        hidden_state = hidden_state.cuda()

    return model, (hidden_state,)


def bench_dense_attn_cpu(run_func, number=10, repeats=10):
    run_func()
    bench_res = []
    
    for i in range(repeats):
        time_record = []
        
        for j in range(number):
            tic = time.time()
            run_func()
            toc = time.time()
            time_record.append(1000 * (toc - tic))

        bench_res.append(np.mean(time_record))
    
    return bench_res


def bench_dense_attn_gpu(run_func, number=100, repeats=10):
    run_func()
    bench_res = []

    for i in range(repeats):
        time_record = []
        
        for j in range(number):
            torch.cuda.synchronize()
            
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            
            tic.record()
            
            run_func()

            toc.record()
            torch.cuda.synchronize()

            elapsed = tic.elapsed_time(toc)
            time_record.append(elapsed)
        
        avg_time = np.mean(time_record)
        bench_res.append(avg_time)

    return bench_res


def run_dense_attn(dense_attn, inputs):
    with torch.no_grad():
        output = dense_attn(*inputs)


def run_bert_benchmark(batch_size=1, seq_len=512, use_large=False, cuda=True, fp16=False):
    dense_attn, inputs = build_bert_model_and_input(batch_size=batch_size, seq_len=seq_len, use_large=use_large, cuda=cuda, fp16=fp16)
    run_func = partial(run_dense_attn, dense_attn=dense_attn, inputs=inputs)
    if cuda:
        bench_res = bench_dense_attn_gpu(run_func)
    else:
        bench_res = bench_dense_attn_cpu(run_func)
    print(f"Benchmark result ({'bert-large' if use_large else 'bert-base'}, {'GPU' if cuda else 'CPU'}, {'TC' if fp16 else 'NTC'}, {seq_len})")
    print(bench_res)
    print(f"mean: {np.mean(bench_res)}, std: {np.std(bench_res)}")
    return np.mean(bench_res)


def run_gpt2_benchmark(batch_size=1, seq_len=512, use_large=False, cuda=True, fp16=False):
    dense_attn, inputs = build_gpt2_model_and_input(batch_size=batch_size, seq_len=seq_len, use_large=use_large, cuda=cuda, fp16=fp16)
    run_func = partial(run_dense_attn, dense_attn=dense_attn, inputs=inputs)
    if cuda:
        bench_res = bench_dense_attn_gpu(run_func)
    else:
        bench_res = bench_dense_attn_cpu(run_func)
    print(f"Benchmark result ({'gpt2-medium' if use_large else 'gpt2-small'}, {'GPU' if cuda else 'CPU'}, {'TC' if fp16 else 'NTC'}, {seq_len})")
    print(bench_res)
    print(f"mean: {np.mean(bench_res)}, std: {np.std(bench_res)}")
    return np.mean(bench_res)


def run_bart_benchmark(batch_size=1, seq_len=512, use_large=False, cuda=True, fp16=False):
    dense_attn, inputs = build_bart_model_and_input(batch_size=batch_size, seq_len=seq_len, use_large=use_large, cuda=cuda, fp16=fp16)
    run_func = partial(run_dense_attn, dense_attn=dense_attn, inputs=inputs)
    if cuda:
        bench_res = bench_dense_attn_gpu(run_func)
    else:
        bench_res = bench_dense_attn_cpu(run_func)
    print(f"Benchmark result ({'bart-large' if use_large else 'bart-base'}, {'GPU' if cuda else 'CPU'}, {'TC' if fp16 else 'NTC'}, {seq_len})")
    print(bench_res)
    print(f"mean: {np.mean(bench_res)}, std: {np.std(bench_res)}")
    return np.mean(bench_res)


BENCH_FUNCS = {
    'bert': run_bert_benchmark,
    'gpt2': run_gpt2_benchmark,
    'bart': run_bart_benchmark,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, required=True, 
                        help="Model type selected in the list: bert-base, bert-large, "
                             "gpt2-small, gpt2-medium, bart-base, bart-large.")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--seq_len", default=128, type=int, help="The maximum total input sequence length")
    parser.add_argument("--cuda", default=False, action='store_true', help="Use GPU or not")
    parser.add_argument("--fp16", default=False, action='store_true', help="Enable half precision inference")
    parser.add_argument("--all", default=False, action='store_true', 
                        help="Evaluate all models ('bert-base', 'gpt2-small', 'bart-base') "
                             "and all sequence lengths (128, 384, 512)")
    args = parser.parse_args()

    if not args.all:
        model_name, variant = args.model_name.split('-')
        use_large = variant in ['large', 'medium']
        bench_func = BENCH_FUNCS[model_name]
        bench_func(args.batch_size, args.seq_len, use_large, args.cuda, args.fp16)
    else:
        bench_results = dict()
        for model_name in ['bert-base', 'gpt2-small', 'bart-base']:
            bench_results[model_name] = dict()
            model_name, variant = args.model_name.split('-')
            use_large = variant in ['large', 'medium']
            bench_func = BENCH_FUNCS[model_name]
            for seq_len in [128, 384, 512]:
                avg_lat = bench_func(args.batch_size, seq_len, use_large, args.cuda, args.fp16)
                bench_results[model_name][seq_len] = avg_lat
        
        df = pd.DataFrame(bench_results); print(df)
        df.to_csv('bench_results.csv')


if __name__ == '__main__':
    main()
