# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable-all
"""This version contains modification to make it easier to trace and support batch."""

import math
from typing import Optional, Tuple
import functools

import torch
from torch import nn
import torch.nn.functional as F
import torch_xla2

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map

import numpy as np

class Int8Embedding(torch.nn.Module):

  def __init__(self, num_embeddings, embedding_dims, device="cpu"):
    super().__init__()
    table = torch.ones(
        (num_embeddings, embedding_dims), device=device, dtype=torch.int8
    )
    self.register_buffer("weight", table)
    embedding_scaler = torch.ones(
        (embedding_dims,), device=device, dtype=torch.bfloat16
    )
    self.register_buffer("weight_scaler", embedding_scaler)

  def forward(self, input):
    return F.embedding(input, self.weight) * self.weight_scaler


class WeightOnlyInt8Linear(torch.nn.Module):

  def __init__(self, in_features, out_features, bias=None, device=None):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    weight = torch.ones(
        (out_features, in_features), dtype=torch.int8, device=device
    )
    self.register_buffer("weight", weight)

    weight_scaler = torch.ones(
        (out_features,), dtype=torch.bfloat16, device=device
    )
    self.register_buffer("weight_scaler", weight_scaler)

    # if bias:
    #   self.bias = torch.nn.Parameter(
    #     torch.zeros((out_features, ),
    #     dtype=torch.bfloat16, device=device))
    # else:
    #   self.register_parameter('bias', None)

  def forward(self, inputs):
    return F.linear(inputs, self.weight) * self.weight_scaler


class RMSNorm(torch.nn.Module):
  """RMSNorm module."""

  def __init__(self, dim: int, eps: float = 1e-6, device="meta"):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, device=device))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    output = self._norm(x.float()).type_as(x)
    return output * self.weight


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
  ndim = x.ndim
  assert 1 < ndim
  assert freqs_cis.shape == (
      x.shape[0],
      x.shape[-3],
      x.shape[-1],
  ), f"freqs_cis: {freqs_cis.shape }, x: {x.shape}"
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  shape[0] = x.shape[0]  # batch size
  return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  # bs, seqlen, heads, dim
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
  """torch.repeat_interleave(x, dim=2, repeats=n_rep)."""

  bs, n_kv_heads, slen, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
      x[:, :, None, :, :]
      .expand(bs, n_kv_heads, n_rep, slen, head_dim)
      .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
  )


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)

def ragged_flash_attention_kernel(
    start_ref,
    end_ref,
    line_end_ref,
    pre_b_ref,
    pre_i_ref,
    q_ref,
    k_ref,
    v_ref,
    k_scaler_ref,
    v_scaler_ref,
    o_ref,
    m_ref,
    l_ref,
    bk: int,
    mask_value: float,
    normalize_var: bool,
    quantized: bool,
):
  """Pallas kernel for flash attention."""
  with jax.named_scope("attention_kernel"):
      b, i = pl.program_id(0), pl.program_id(1)

      @pl.when(i == 0)
      def init():
        with jax.named_scope("init"):
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            l_ref[...] = jnp.zeros_like(l_ref)
            o_ref[...] = jnp.zeros_like(o_ref)

      length = line_end_ref[b]
      start = start_ref[b]
      end = end_ref[b]

      @pl.when(jnp.logical_and(i * bk < length, start != end))
      def run():
        with jax.named_scope("run_qk"):
            q = q_ref[...].astype(jnp.float32)
            k = k_ref[...].astype(jnp.float32)
            v = v_ref[...].astype(jnp.float32)
            m_prev, l_prev = m_ref[...], l_ref[...]

            qk = jax.lax.dot_general(
                q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            if normalize_var:
              qk = qk / jnp.sqrt(k.shape[-1])
            if quantized:
              qk = qk * k_scaler_ref[...]
        with jax.named_scope("run_mask"):
            start = start_ref[b]
            end = end_ref[b]
            iota = jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1)
            mask_start_lt_end = jnp.logical_and(i * bk + iota >= start, i * bk + iota < end).astype(jnp.int32)
            mask_start_gt_end = jnp.logical_or(i * bk + iota >= start, i * bk + iota < end).astype(jnp.int32)
            #mask = jax.lax.cond(start <= end, lambda: mask_start_lt_end, lambda: mask_start_gt_end)
            mask = jnp.where(start <= end, mask_start_lt_end, mask_start_gt_end)

            qk = qk + jnp.where(mask, 0.0, mask_value)

        with jax.named_scope("run_softmax"):
            m_curr = qk.max(axis=-1)

            s_curr = jnp.exp(qk - m_curr[..., None])

            l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
            if quantized:
              s_curr = s_curr * v_scaler_ref[...]
            o_curr_times_l_curr = jnp.dot(s_curr, v)
            m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
            m_next = jnp.maximum(m_prev, m_curr)
            alpha = jnp.exp(m_prev - m_next)
            beta = jnp.exp(m_curr - m_next)
            l_next = alpha * l_prev + beta * l_curr
            l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

            m_ref[...], l_ref[...] = m_next, l_next_safe
            o_ref[...] = (
                (l_prev * alpha * o_ref[...] + beta * o_curr_times_l_curr) / l_next_safe
            ).astype(o_ref.dtype)

@functools.partial(jax.jit, static_argnames=["bk", "mask_value", "normalize_var"])
def ragged_mqa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    start: jax.Array,
    end: jax.Array,
    k_scaler: jax.Array | None = None,
    v_scaler: jax.Array | None = None,
    pre_batch = None,
    pre_block = None,
    bk: int = 512,
    mask_value: float = DEFAULT_MASK_VALUE,
    normalize_var: bool = True,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi query attention."""
  with jax.named_scope("ragged_mqa"):
      batch_size, num_heads, head_dim = q.shape 
      seq_len = k.shape[1]  

      def kv_index_map(b, i, start_ref, end_ref, line_end_ref, pre_batch_ref, pre_block_ref):
        index = b * (seq_len // bk) + i
        return pre_batch_ref[index], pre_block_ref[index], 0

      def q_index_map(b, i, start_ref, end_ref, line_end_ref, pre_batch_ref, pre_block_ref):
        index = b * (seq_len // bk) + i
        return pre_batch_ref[index], 0, 0

      def scaler_index_map(b, i, *_):
        return b, 0, i

      line_end = jnp.where(start < end, end, seq_len - 1)


      if k_scaler is not None:
          out, m, l = pl.pallas_call(
              functools.partial(
                  ragged_flash_attention_kernel,
                  bk=bk,
                  mask_value=mask_value,
                  normalize_var=normalize_var,
                  quantized=False,
              ),
              grid_spec=pltpu.PrefetchScalarGridSpec(
                  num_scalar_prefetch=5,
                  in_specs=[
                      pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                      pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
                      pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
                      pl.BlockSpec(scaler_index_map, (None, 1, bk)),
                      pl.BlockSpec(scaler_index_map, (None, 1, bk)),
                  ],
                  out_specs=[
                      pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                      pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                      pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                  ],
                  grid=(batch_size, seq_len // bk),
              ),
              compiler_params=dict(dimension_semantics=("parallel", "arbitrary")),
              out_shape=[
                  q,
                  jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
                  jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
              ],
          )(start, end, line_end, pre_batch, pre_block, q, k, v, k_scaler, v_scaler)
      else:
        out, m, l = pl.pallas_call(
          functools.partial(
              ragged_flash_attention_kernel,
              bk=bk,
              mask_value=mask_value,
              normalize_var=normalize_var,
              quantized=True,
          ),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=5,
              in_specs=[
                pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
                pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
              ],
              out_specs=[
                pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
              ],
              grid=(batch_size, seq_len // bk),
          ),
          compiler_params=dict(dimension_semantics=("parallel", "arbitrary")),
          out_shape=[
              q,
              jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
              jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
          ],
        )(start, end, line_end, pre_batch, pre_block, q, k, v)
  return out, (m[..., 0], l[..., 0])


@functools.partial(jax.jit, static_argnames=['bk', 'mask_value', 'normalize_var', 'shard_axis'])
def ragged_mha(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    start: jax.Array,
    end: jax.Array,
    pre_batch: jax.Array,
    pre_block: jax.Array,
    k_scaler: jax.Array | None = None,
    v_scaler: jax.Array | None = None,
    bk: int = 512,
    mask_value : float = DEFAULT_MASK_VALUE,
    normalize_var: bool = True,
    shard_axis: int = 1
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi head attention.
  Args:
    q: A [batch_size, compute_dim, num_heads, head_dim] jax.Array.
    k: A [batch_size, num_heads, seq_len, head_dim] jax.Array or
      PartitionQuantizedTensor.
    v: A [batch_size, num_heads, seq_len, head_dim] jax.Array or
      PartitionQuantizedTensor.
    start: A i32[batch_size] jax.Array
    end: A i32[batch_size] jax.Array
    bk: An integer that is the sequence block size.
    logit_cap: An optional float that caps logits via tanh. By default there is
      no logit capping.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    out_dtype: An optional dtype for the output. If not provided, the output
      dtype will be q's dtype.
  Returns:
    The output of attention([batch_size, num_heads, compute_dim, head_dim]),
    along with the max logit ([batch_size, num_heads, compute_dim, 1]) and
    softmax denominator ([batch_size, num_heads, compute_dim, 1]).
  """
  mask_value = DEFAULT_MASK_VALUE
  seqlen = q.shape[-2]
  if k_scaler is None:
    replicated_in_axes = 4
    replicated_inputs = (pre_batch, pre_block)
  else:
    replicated_in_axes = 6
    replicated_inputs = (k_scaler, v_scaler, pre_batch, pre_block)

  with jax.named_scope("ragged_mha_vmap"):
    out, (m, l) = jax.vmap(
      functools.partial(
          ragged_mqa,
          bk=bk,
          mask_value=mask_value,
          normalize_var=normalize_var,
          #out_dtype=out_dtype,
      ),
      in_axes=(shard_axis, shard_axis, shard_axis, *([None]*replicated_in_axes)),
      out_axes=shard_axis,
    )(q, k, v, start, end, *replicated_inputs)
  return out, (m, l)


def dense_attention(xq, keys, values, mask):
  head_dim = xq.shape[-1]
  with jax.named_scope("attn_mat1"):
      ## Attention start
      # scores = torch.einsum(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(self.head_dim)
      scores = torch.einsum("ikjl,ikml->ikjm", xq, keys) / math.sqrt(head_dim)
      if mask is not None:
        # if mask.shape != (1,1,16,16):
        #   breakpoint()
        scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
  with jax.named_scope("attn_soft"):
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)

  with jax.named_scope("attn_mat2"):
    # output = torch.einsum(
    #    "ikjm,ikml->ikjl", scores, values
    # )  # (bs, n_local_heads, seqlen, head_dim)
    output = torch.einsum("ikjm,ikml->ikjl", scores, values)


def dense_attention_quantized(
    xq: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    k_scaler = None,
    v_scaler = None,
    mask = None,
):
      bsz, _, _, head_dim = xq.shape
 
      with jax.named_scope("attn_mat1"):
        ## Attention start
        # scores = torch.einsum(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(self.head_dim)
        scores = (
            torch.einsum("ikjl,ikml->ikjm", xq, keys)
            / math.sqrt(head_dim)
            * (k_scaler.reshape(bsz, 1, 1, keys.shape[2]))
        )
        if mask is not None:
          scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
      with jax.named_scope("attn_soft"):
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = scores * v_scaler.reshape((bsz, 1, 1, keys.shape[2]))

      with jax.named_scope("attn_mat2"):
        # output = torch.einsum(
        #    "ikjm,ikml->ikjl", scores, values
        # )  # (bs, n_local_heads, seqlen, head_dim)
        output = torch.einsum("ikjm,ikml->ikjl", scores, values)

      return output


class AttentionKernel:

  def __init__(self, env):
    self.env = env
    self.shard_axis = 0 if self.env.shard_on_batch else 1
    qkv_pspec = self.env.partition_by_axis(self.shard_axis) # Number of heads
    others_pspec = self.env.partition_by_axis()
    self.binded_ragged_mha = functools.partial(ragged_mha, bk=self.env.block_size, shard_axis=self.shard_axis)
    self.binded_ragged_mha = shard_map(ragged_mha, env.mesh, in_specs=(*([qkv_pspec] * 3), *([others_pspec] * 4)), out_specs=(others_pspec, (others_pspec, others_pspec)), check_rep=False)
    self.binded_ragged_mha = jax.jit(self.binded_ragged_mha)

  def __call__(self, xq, xk, xv, mask, cache, start, end, pre_batch, pre_block):
    """
    Args:
      xq: torch.Tensor of (batch size, num_heads, seqlen, head_dim)
      xk: torch.Tensor of (batch size, num_kv_heads, seqlen, head_dim)
      xv: torch.Tensor of (batch size, num_kv_heads, seqlen, head_dim)
      mask: mask with 0 and -inf, or None
      cache: CacheManagerInterface object
    """
    bsz, num_heads, seqlen, head_dim = xq.shape
    _, num_kv_heads, _, kv_head_dim = xk.shape
    n_rep = num_heads // num_kv_heads
    if seqlen == 1:
      xq = torch.broadcast_to(xq, (xq.shape[0], xq.shape[1], 2, xq.shape[3]))

    with jax.named_scope("attn_insert_cache"):
      keys, values = cache.update(xk, xv)
      keys = repeat_kv(keys, n_rep)
      values = repeat_kv(values, n_rep)
  
    with jax.named_scope("attn_qkv"):
      if self.env.ragged_mha and seqlen == 1:
        output, _ = torch_xla2.extra.call_jax(self.binded_ragged_mha, xq, keys, values, start, end, pre_batch, pre_block)
      else:
        output = dense_attention(xq, keys, values, mask)

      if seqlen == 1:
        output = output[:, :, 0:1, :]
      # For XLA matmul performance boost
      # output = torch.matmul(scores, values)
      self.env.apply_sharding(output, axis=self.shard_axis)
      return output


class Int8KVAttentionKernel:

  def __init__(self, env):
    self.env = env
    self.shard_axis = 0 if self.env.shard_on_batch else 1
    qkv_pspec = self.env.partition_by_axis(self.shard_axis) # Number of heads
    others_pspec = self.env.partition_by_axis()
    self.binded_ragged_mha_quantized = functools.partial(ragged_mha, bk=self.env.block_size, shard_axis=self.shard_axis)
    self.binded_ragged_mha_quantized = shard_map(self.binded_ragged_mha_quantized, env.mesh, in_specs=(*([qkv_pspec] * 3), *([others_pspec]*6)), out_specs=(others_pspec, (others_pspec, others_pspec)), check_rep=False)
    self.binded_ragged_mha_quantized = jax.jit(self.binded_ragged_mha_quantized)

  def __call__(self, xq, xk, xv, mask, cache, start, end, pre_batch, pre_block):
    """
    Args:
      xq: torch.Tensor of (batch size, num_heads, seqlen, head_dim)
      xk: torch.Tensor of (batch size, num_kv_heads, seqlen, head_dim)
      xv: torch.Tensor of (batch size, num_kv_heads, seqlen, head_dim)
      mask: mask with 0 and -inf, or None
      cache: CacheManagerInterface object
    """
    bsz, num_heads, seqlen, head_dim = xq.shape
    _, num_kv_heads, _, kv_head_dim = xk.shape
    n_rep = num_heads // num_kv_heads

    if seqlen == 1:
      xq = torch.broadcast_to(xq, (xq.shape[0], xq.shape[1], 2, xq.shape[3]))

    with jax.named_scope("attn_insert_cache"):
      keys, values, k_scaler, v_scaler = cache.update(xk, xv)
      keys = repeat_kv(keys, n_rep)
      values = repeat_kv(values, n_rep)

    with jax.named_scope("attn_qkv"):
      if self.env.ragged_mha and seqlen == 1:
        output, _ = torch_xla2.extra.call_jax(self.binded_ragged_mha_quantized, xq, keys, values, start, end, pre_batch, pre_block, k_scaler, v_scaler)
      else:
        output= dense_attention_quantized(xq, keys, values, k_scaler, v_scaler, mask)

      if seqlen == 1:
        output = output[:, :, 0:1, :]

      self.env.apply_sharding(output, axis=self.shard_axis)
      return output


class Attention(nn.Module):
  """Attention module."""

  def __init__(self, n_heads, n_kv_heads, head_dim, hidden_size, device, env):
    super().__init__()
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.n_rep = self.n_heads // self.n_kv_heads
    self.env = env
    self.hidden_size = hidden_size

    LinearLayer = (
        WeightOnlyInt8Linear if env.enable_weight_quantization else nn.Linear
    )

    self.wo = LinearLayer(
        n_heads * self.head_dim,
        hidden_size,
        bias=False,
        device=device,
    )

    Kernel = (
        Int8KVAttentionKernel if env.enable_kv_quantization else AttentionKernel
    )
    self.attention_kernel = Kernel(env)

    self.q_size = n_heads * self.head_dim
    self.kv_size = self.n_kv_heads * self.head_dim
    if self.env.qkv_fusion:
      self._register_load_state_dict_pre_hook(self.load_hook)
      self.wqkv = LinearLayer(
          hidden_size,
          (n_heads + 2 * self.n_kv_heads) * self.head_dim,
          bias=False,
          device=device,
      )
    else:
      self.wq = LinearLayer(
          hidden_size,
          n_heads * self.head_dim,
          bias=False,
          device=device,
      )
      self.wk = LinearLayer(
          hidden_size,
          self.n_kv_heads * self.head_dim,
          bias=False,
          device=device,
      )
      self.wv = LinearLayer(
          hidden_size,
          self.n_kv_heads * self.head_dim,
          bias=False,
          device=device,
      )

  def load_hook(self, state_dict, prefix, *args):
    if prefix + "wq.weight" in state_dict:
      wq = state_dict.pop(prefix + "wq.weight")
      wk = state_dict.pop(prefix + "wk.weight")
      wv = state_dict.pop(prefix + "wv.weight")
      state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

  def forward(
      self,
      x: torch.Tensor,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      cache,
      start,
      end,
      pre_batch,
      pre_block,
  ):
    # bsz, seqlen, _ = x.shape
    with jax.named_scope("attn_linear_before_cache"):
      bsz, seqlen = x.shape[0], x.shape[-2]

      # qkv fuse
      if self.env.qkv_fusion:
        xq, xk, xv = self.wqkv(x).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
      else:
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
      xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
      xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
      xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

      shard_axis = 0 if self.env.shard_on_batch else 2
      self.env.apply_sharding(xq, axis=shard_axis)
      self.env.apply_sharding(xk, axis=shard_axis)
      self.env.apply_sharding(xv, axis=shard_axis)

    with jax.named_scope("attn_rope"):
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    xq = xq.transpose(1, 2)

    output = self.attention_kernel(xq, xk, xv, mask, cache, start, end, pre_batch, pre_block).type_as(xq)
    output = output.transpose(-3, -2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)
