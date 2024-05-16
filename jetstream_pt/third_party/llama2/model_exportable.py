# pylint: disable-all
"""This version contains modification to make it easier to trace and support batch."""

from typing import Any, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch_xla2

from . import model_args 
import jax

from jetstream_pt.layers import Attention, RMSNorm, Int8Embedding, WeightOnlyInt8Linear

class ForwardParams:
  """Calculates the params used by Forward functions."""

  def __init__(self, start, input_pos, cache_len, block_size):
      def _pre_compute_ragged_block_indices(b, i, start, end, bk, batch_size, seq_len):
      # with jax.named_scope("compute_indices"):
        start = start.reshape((batch_size, 1))
        end = end.reshape((batch_size, 1))

        am_last_batch = b == batch_size - 1
        last_good_block = torch.where(start < end, torch.div(end - 1, bk), torch.div(seq_len -1, bk))

        next_b = torch.where(am_last_batch, b, b + 1)
        next_i = torch.where(am_last_batch, last_good_block, 0)

        # start < end
        def true_comp(b, i, bk, start, end, next_b, next_i):
          b_next = torch.where(i * bk >= end, next_b, b)
          i_next = torch.where(i * bk >= end, next_i, i)
          i_next = torch.where((i + 1) * bk <= start, torch.div(start, bk), i_next)
          return b_next, i_next

        # start > end
        def false_comp(b, i, bk, start, end):
          b_next = b
          i_next = torch.where(torch.logical_and(i * bk >= end, (i + 1) * bk <= start), torch.div(start, bk), i)
          return b_next, i_next

        true_comp_b, true_comp_i = true_comp(b_iota, i, bk, start, end, next_b, next_i)
        false_comp_b, false_comp_i = false_comp(b_iota, i, bk, start, end)

        b_next = torch.where(start < end, true_comp_b, torch.where(start == end, next_b, false_comp_b))
        i_next = torch.where(start < end, true_comp_i, torch.where(start == end, next_i, false_comp_i))
        return b_next, i_next

      start, input_pos = torch_xla2.tensor.wrap((start, input_pos))
      bsz = start.shape[0]
      b_iota = torch.arange(bsz).reshape((bsz, 1))
      num_bk = cache_len // block_size
      num_bk_iota = torch.arange(num_bk).reshape((1, num_bk))
      num_bk_iota = torch.broadcast_to(num_bk_iota, (bsz, num_bk))
      end = (start + input_pos) % cache_len
      self.pre_b, self.pre_i = _pre_compute_ragged_block_indices(b_iota, num_bk_iota, start, end, block_size, bsz, cache_len)
      self.pre_b, self.pre_i = self.pre_b.reshape(-1), self.pre_i.reshape(-1)

class FeedForward(nn.Module):
  """Feed-forward module."""

  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      multiple_of: int,
      ffn_dim_multiplier: Optional[float],
      device = 'meta',
      quantize = False,
      env = None,
  ):
    super().__init__()
    self.env = env
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    LinearLayer = WeightOnlyInt8Linear if quantize else nn.Linear

    self.w1 = LinearLayer(
        dim,
        hidden_dim,
        bias=False,
        device=device,
    )
    self.w2 = LinearLayer(
        hidden_dim,
        dim,
        bias=False,
        device=device,
    )
    self.w3 = LinearLayer(
        dim,
        hidden_dim,
        bias=False,
        device=device,
    )

  def forward(self, x):
    result = self.w2(F.silu(self.w1(x)) * self.w3(x))
    return result


class TransformerBlock(nn.Module):
  """Transformer block."""

  def __init__(
      self,
      layer_id: int,
      args: model_args.ModelArgs,
      env,
  ):
    super().__init__()
    self.env = env
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads

    self.attention = Attention(
        args,
        env,
    )
    self.feed_forward = FeedForward(
        dim=args.dim,
        hidden_dim=4 * args.dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        device=args.device,
        quantize=args.quantize,
        env=env
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, device=args.device)
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, device=args.device)

  def forward(
      self,
      x: torch.Tensor,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      cache,
      start: torch.Tensor | None,
      input_pos,
      forward_params : ForwardParams
  ):
    with jax.named_scope('Attention'):
      attn = self.attention.forward(
          self.attention_norm(x),
          freqs_cis,
          mask,
          cache,
          start,
          input_pos,
          forward_params,
      )
    with jax.named_scope('ffn_norm'):
        h = x + attn
        ffns = self.ffn_norm(h)

    with jax.named_scope('ffn'):
        out = h + self.feed_forward.forward(ffns)
        return out

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis

class Transformer(nn.Module):
  """Transformer module."""

  def __init__(
      self,
      params: model_args.ModelArgs,
      env,
  ):
    super().__init__()
    self.env = env
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers

    Embedding = Int8Embedding if params.quantize else nn.Embedding
    self.tok_embeddings = Embedding(
        params.vocab_size,
        params.dim,
        device=params.device,
    )

    self.layers = torch.nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(
          TransformerBlock(
              layer_id,
              params,
              env
          )
      )
    self.norm = RMSNorm(params.dim, eps=params.norm_eps, device=params.device)
    
    LinearLayer = WeightOnlyInt8Linear if params.quantize else nn.Linear

    self.output = LinearLayer(
        params.dim,
        params.vocab_size,
        bias=False,
        device=params.device,
    )
    # TODO what to do with this
    freqs_cis = precompute_freqs_cis(
        self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
    )

    self.register_buffer("freqs_cis", freqs_cis)


  @torch.no_grad()
  def forward(
      self,
      tokens: torch.Tensor,
      start: torch.Tensor | None,
      input_pos: torch.Tensor,
      caches: List[Any],
      mask,
  ):
    with jax.named_scope('transformer_tok'):
        seqlen = tokens.shape[-1]
        h = self.tok_embeddings(tokens)

    with jax.named_scope('transformer_freq'):
        bsz, seqlen = tokens.shape
        freqs_cis = self.freqs_cis[input_pos]
        freqs_cis = freqs_cis.reshape(bsz, seqlen, -1)

    forward_params = ForwardParams(start, input_pos, self.env.cache_len, self.env.block_size) if start is not None else None

    for layer, cache in zip(self.layers, caches):
      with jax.named_scope('TransformerBlock'):
        h = layer(
            h,
            freqs_cis,
            mask,
            cache,
            start,
            input_pos,
            forward_params,
        )

    with jax.named_scope('transformer_norm'):
        h = self.norm(h)
        output = self.output(h).float()
    return output
