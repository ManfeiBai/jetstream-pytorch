"""Kernels for ragged attention."""

import functools

import math
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import torch_xla2
from typing import Optional
import torch.nn.functional as F
import torch
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


#@functools.partial(jax.jit, static_argnames=["mask_value"])
def mqa_reference(
    q: jax.Array,       
    k: jax.Array,       
    v: jax.Array,       
    lengths: jax.Array, 
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Multi query attention reference.

  Args:
    q: A [batch_size, num_heads, head_dim] jax.Array.
    k: A [batch_size, seq_len, head_dim] jax.Array.
    v: A [batch_size, seq_len, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.

  Returns:
    The output of attention([batch_size, num_heads, head_dim]), along with the
    max logit ([batch_size, num_heads]) and softmax denominator ([batch_size,
    num_heads]).
  """
  logits = jnp.einsum(
      "bhd,btd->bht", q.astype(jnp.float32), k.astype(jnp.float32)
  )
  mask = jnp.arange(k.shape[1])[None] < lengths[:, None]        
  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None]   
  logits_max = logits.max(axis=-1)                              
  unnormalized = jnp.exp(logits - logits_max[..., None])        
  denominator = unnormalized.sum(axis=-1)                       
  o = (                                                         
      jnp.einsum("bht,btd->bhd", unnormalized.astype(v.dtype), v)
      / denominator[..., None]
  )
  return o, (logits_max, denominator)


def ragged_flash_attention_kernel(
    start_ref,
    end_ref,
    line_end_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    m_ref,
    l_ref,
    bk: int,
    mask_value: float,
    normalize_var: bool,
):
  """Pallas kernel for flash attention."""
  b, i = pl.program_id(0), pl.program_id(1)
  line_end = line_end_ref[b]
  start = start_ref[b]
  end = end_ref[b]

  @pl.when(i == 0)
  def init():
    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    o_ref[...] = jnp.zeros_like(o_ref)

  #length = lengths_ref[b]
  length = line_end_ref[...]

  @pl.when(i * bk < length)
  def run():
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)
    m_prev, l_prev = m_ref[...], l_ref[...]

    qk = lax.dot_general(
        q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    if normalize_var:
        qk = qk / jnp.sqrt(k.shape[-1])
    start = start_ref[b]
    end = end_ref[b] 
    #assert qk.shape == (1), f"qk shape is {qk.shape}"
    iota = jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1)
    mask_start_lt_end = jnp.logical_and(i * bk + iota >= start, i * bk + iota < end)
    mask_start_gt_end = jnp.logical_or(i * bk + iota >= start, i * bk + iota < end)
    mask = jax.lax.cond(start <= end, lambda: mask_start_lt_end, lambda: mask_start_gt_end)

    qk = qk + jnp.where(mask, 0.0, mask_value)
    m_curr = qk.max(axis=-1)

    s_curr = jnp.exp(qk - m_curr[..., None])
    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
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

def ragged_flash_attention_quantized_kernel(
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

      #cond1 = jnp.logical_or(start >= (i + 1) * bk, end <= i * bk)
      #cond2 = jnp.logical_and(start <= (i + 1) * bk, end >= i * bk)
      #cond3 = jnp.logical_and(start < end, jnp.logical_not(cond1))
      #cond4 = jnp.logical_and(start > end, jnp.logical_not(cond2))
      #cond = jnp.logical_or(cond2, cond3)
      #@pl.when(cond)
      @pl.when(jnp.logical_and(i * bk < length, start != end))
      def run():
        with jax.named_scope("run_qk"):
            q = q_ref[...].astype(jnp.float32)
            k = k_ref[...].astype(jnp.float32)
            v = v_ref[...].astype(jnp.float32)
            m_prev, l_prev = m_ref[...], l_ref[...]

            qk = lax.dot_general(
                q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            if normalize_var:
                qk = qk / jnp.sqrt(k.shape[-1])
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
    #lengths: jax.Array,
    start: jax.Array,
    end: jax.Array,
    k_scaler: jax.Array | None = None,
    v_scaler: jax.Array | None = None,
    pre_b = None,
    pre_i = None,
    bk: int = 512,
    mask_value: float = DEFAULT_MASK_VALUE,
    normalize_var: bool = False,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi query attention."""
  with jax.named_scope("ragged_mqa"):
      batch_size, num_heads, head_dim = q.shape 
      #assert lengths.shape == (batch_size,), lengths.shape
      #assert lengths.dtype == jnp.int32
      seq_len = k.shape[1]  

      def kv_index_map(b, i, start_ref, end_ref, line_end_ref, pre_b_ref, pre_i_ref):
        index = b * (seq_len // bk) + i
        return pre_b_ref[index], pre_i_ref[index], 0

      def q_index_map(b, i, start_ref, end_ref, line_end_ref, pre_b_ref, pre_i_ref):
        index = b * (seq_len // bk) + i
        return pre_b_ref[index], 0, 0

      def scaler_index_map(b, i, *_):
        return b, 0, i

      line_end = jnp.where(start < end, end, seq_len - 1)


      if k_scaler is not None:
          out, m, l = pl.pallas_call(
              functools.partial(
                  ragged_flash_attention_quantized_kernel,
                  bk=bk,
                  mask_value=mask_value,
                  normalize_var=normalize_var,
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
              #interpret=True,
          )(start, end, line_end, pre_b, pre_i, q, k, v, k_scaler, v_scaler)
      else:
        out, m, l = pl.pallas_call(
          functools.partial(
              ragged_flash_attention_kernel,
              bk=bk,
              mask_value=mask_value,
              normalize_var=normalize_var,
          ),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=3,
              in_specs=[
                  pl.BlockSpec(lambda b, i, _1, _2, _3: (b, 0, 0), (None, num_heads, head_dim)),
                  pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
                  pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
              ],
              out_specs=[
                  pl.BlockSpec(lambda b, i, _1, _2, _3: (b, 0, 0), (None, num_heads, head_dim)),
                  pl.BlockSpec(lambda b, i, _1, _2, _3: (b, 0, 0), (None, num_heads, head_dim)),
                  pl.BlockSpec(lambda b, i, _1, _2, _3: (b, 0, 0), (None, num_heads, head_dim)),
              ],
              grid=(batch_size, seq_len // bk),
          ),
          compiler_params=dict(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
          out_shape=[
              q,
              jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
              jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
          ],
        )(start, end, line_end, q, k, v)
  return out, (m[..., 0], l[..., 0])

def mha_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    layer: jax.Array | None,
    lengths: jax.Array,
    logit_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    out_dtype: jnp.dtype | None = None,
    use_base2: bool = False,
    sliding_window_size: int | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Multi head attention reference.

  Args:
    q: A [batch_size, num_heads, compute_dim, head_dim] jax.Array.
    k: A [num_layers?, batch_size, num_heads, seq_len, head_dim] jax.Array or
      QuantizedPartitionTensor.
    v: A [num_layers?, batch_size, num_heads, seq_len, head_dim] jax.Array or
      QuantizedPartitionTensor.
    layer: An optional i32[] jax.Array that indicates what layer of the kv to
      use. If it is None, there should be no num_layers dimension on kv.
    lengths: A i32[batch_size] jax.Array.
    logit_cap: An optional float that caps logits via tanh. By default there is
      no logit capping.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    out_dtype: An optional dtype for the output. If not provided, the output
      dtype will be q's dtype.
    use_base2: Whether or not attention should be computed using base2.
      Requires that either q or k be pre-scaled by log_2(e).
    sliding_window_size: Size of the sliding window for local attention.

  Returns:
    The output of attention([batch_size, num_heads, compute_dim, head_dim]),
    along with the max logit ([batch_size, num_heads, compute_dim, 1]) and
    softmax denominator ([batch_size, num_heads, compute_dim, 1]).
  """
  kv_head_axis = 1
  return jax.vmap(functools.partial(mqa_reference,
                                    mask_value=mask_value,
                                    #e2,
                                    #sliding_window_size=sliding_window_size
                                    ),
                  in_axes=(1, kv_head_axis, kv_head_axis, None),
                  out_axes=1)(q, k, v, lengths)

"""
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

devices = mesh_utils.create_device_mesh((8, 1))
mesh = Mesh(devices, axis_names=('x', 'y'))
x_sharding = P(None, 'x', None, None)
out_sharding = P(None, 'x', None)
replicate1 = P(None,)
replicate2 = P(None, None,)
replicate3 = P(None, None, None,)
input_sharding = (x_sharding, x_sharding, x_sharding, replicate1)
output_sharding = (x_sharding, (out_sharding, out_sharding))
#@functools.partial(custom_partitioning, static_argnums=(4, 5,))
#@custom_partitioning
@functools.partial(shard_map, mesh=mesh, in_specs=input_sharding, out_specs=output_sharding, check_rep=False)
"""
@functools.partial(jax.jit, static_argnames=['bk', 'normalize_var'])
def ragged_mha(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    #lengths: jax.Array,
    start: jax.Array,
    end: jax.Array,
    k_scaler: jax.Array | None = None,
    v_scaler: jax.Array | None = None,
    pre_b: jax.Array | None = None,
    pre_i: jax.Array | None = None,
    bk: int = 512,
    normalize_var: bool = True,
    #bk: int,
    #logit_cap: float | None = None,
    #mask_value: float = DEFAULT_MASK_VALUE,
    #out_dtype: jnp.dtype | None = None,
    #use_base2: bool = False,
    #interpret: bool = False,
    #sliding_window_size: int | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi head attention.

  Args:
    q: A [batch_size, num_heads, compute_dim, head_dim] jax.Array.
    k: A [num_layers?, batch_size, num_heads, seq_len, head_dim] jax.Array or
      PartitionQuantizedTensor.
    v: A [num_layers?, batch_size, num_heads, seq_len, head_dim] jax.Array or
      PartitionQuantizedTensor.
    layer: An optional i32[] jax.Array that indicates what layer of the kv to
      use. If it is None, there should be no num_layers dimension on kv.
    lengths: A i32[batch_size] jax.Array.
    bk: An integer that is the sequence block size.
    logit_cap: An optional float that caps logits via tanh. By default there is
      no logit capping.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    out_dtype: An optional dtype for the output. If not provided, the output
      dtype will be q's dtype.
    use_base2: Whether or not attention should be computed using base2.
      Requires that either q or k be pre-scaled by log_2(e).
    interpret: Whether or not the kernel should be run in interpret mode.
    sliding_window_size: Size of the sliding window for local attention.

  Returns:
    The output of attention([batch_size, num_heads, compute_dim, head_dim]),
    along with the max logit ([batch_size, num_heads, compute_dim, 1]) and
    softmax denominator ([batch_size, num_heads, compute_dim, 1]).
  """
  mask_value = DEFAULT_MASK_VALUE
  kv_head_axis = 1
  bsz, _, seqlen, _ = q.shape
  cache_len = k.shape[-2]

  #if lengths.ndim > 1:
  #    lengths.reshape((lengths.shape[0],))
  
  #num_blk_3 = jnp.where(lengths >= 0.75 * cache_len, 1, 0)
  #num_blk_2 = jnp.where(lengths >= 0.5 * cache_len, 1, 0)

  #bk = jax.lax.cond(jnp.sum(num_blk_2) >= bsz * 0.5, lambda:cache_len/2, lambda:cache_len/4)
  #bk = jax.lax.cond(jnp.sum(num_blk_3) >= bsz * 0.5, lambda:cache_len, lambda:bk)
  #bk = jax.lax.cond(bk < 512, lambda:512, lambda:bk)

  #bk = jax.lax.cond(jnp.sum(num_blk_2) >= bsz * 0.5, lambda:jnp.array(cache_len/2, dtype=jnp.int32), lambda:jnp.array(cache_len/4, dtype=jnp.int32))
  #bk = jax.lax.cond(jnp.sum(num_blk_3) >= bsz * 0.5, lambda:jnp.array(cache_len, dtype=jnp.int32), lambda:bk)
  #bk = jax.lax.cond(bk < 512, lambda:jnp.array(512, dtype=jnp.int32), lambda:bk)

  #bk = jnp.array([512])
  #bk = 512 
  #bk = 2048 
  #bk = 1024

  with jax.named_scope("ragged_mha_expand"):
      if seqlen == 1:
          q = jnp.broadcast_to(q, (q.shape[0], q.shape[1], 2, q.shape[3]))  
  with jax.named_scope("ragged_mha_vmap"):
      if k_scaler is not None:
        out, (m, l) = jax.vmap(
          functools.partial(
              ragged_mqa,
              bk=bk,
              #logit_cap=logit_cap,
              mask_value=mask_value,
              normalize_var=normalize_var,
              #out_dtype=out_dtype,
              #use_base2=use_base2,
              #sliding_window_size=sliding_window_size
          ),
          in_axes=(1, kv_head_axis, kv_head_axis, None, None, None, None, None, None),
          out_axes=1,
        )(q, k, v, start, end, k_scaler, v_scaler, pre_b, pre_i)
      else:
        out, (m, l) = jax.vmap(
          functools.partial(
              ragged_mqa,
              bk=bk,
              #logit_cap=logit_cap,
              mask_value=mask_value,
              normalize_var=normalize_var,
              #out_dtype=out_dtype,
              #use_base2=use_base2,
              #sliding_window_size=sliding_window_size
          ),
          in_axes=(1, kv_head_axis, kv_head_axis, None, None),
          out_axes=1,
        )(q, k, v, start, end)
  with jax.named_scope("ragged_mha_shrink"):
      if seqlen == 1:
          out = out[:, :, 0:1, :]
          m = m[:, :, 0:1]
          l = l[:, :, 0:1]
  return out, (m, l)

def dense_attention(
    xq: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    lengths: jax.Array  = None,
    env = None,
    mask = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    ) -> jax.Array: 
#-> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    if env is None:
        raise NotImplementedError('Env missing!')
    bsz, seqlen, num_heads, head_dim = xq.shape
    with jax.named_scope('attn_mat1'):
        ## Attention start
        if seqlen == 1:
          xq = torch.broadcast_to(xq, (xq.shape[0], 2, xq.shape[2], xq.shape[3]))
        #scores = torch.einsum(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(self.head_dim)
        scores = torch_xla2.extra.call_jax(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(head_dim)
        env.apply_sharding(scores, axis=1)
        if mask is not None:
          # if mask.shape != (1,1,16,16):
          #   breakpoint()
          scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
    with jax.named_scope('attn_soft'):
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

    with jax.named_scope('attn_mat2'):
        #output = torch.einsum(
        #    "ikjm,ikml->ikjl", scores, values
        #)  # (bs, n_local_heads, seqlen, head_dim)
        output = torch_xla2.extra.call_jax(jnp.einsum,"ikjm,ikml->ikjl", scores, values)
        if seqlen == 1:
          output = output[:, :, 0:1, :]
        # For XLA matmul performance boost
        #output = torch.matmul(scores, values)
        env.apply_sharding(output, axis=1)
    return output

def dense_attention_quantized(
    xq: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    lengths: Optional[jax.Array] = None,
    k_scaler = None,
    v_scaler = None,
    mask = None,
    env = None,
    mask_value: float = DEFAULT_MASK_VALUE,
):
      if env is None:
        raise NotImplementedError('Env missing!')
      if k_scaler is None or v_scaler is None:
        raise NotImplementedError('Scaler missing!')
      bsz, seqlen, num_heads, head_dim = xq.shape
      cachelen = keys.shape[2]
      with jax.named_scope('attn_mat1'):
        ## Attention start
        #scores = torch.einsum(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(self.head_dim)
        if seqlen == 1:
          xq = torch.broadcast_to(xq, (xq.shape[0], 2, xq.shape[2], xq.shape[3]))
        scores = torch_xla2.extra.call_jax(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(head_dim) * (k_scaler.reshape(bsz, 1, 1, cachelen))
        env.apply_sharding(scores, axis=1)
        if mask is not None:
          scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
      with jax.named_scope('attn_soft'):
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = scores * v_scaler.reshape((bsz, 1, 1, cachelen))
        env.apply_sharding(scores, axis=1)

      with jax.named_scope('attn_mat2'):
        #output = torch.einsum(
        #    "ikjm,ikml->ikjl", scores, values
        #)  # (bs, n_local_heads, seqlen, head_dim)
        output = torch_xla2.extra.call_jax(jnp.einsum,"ikjm,ikml->ikjl", scores, values)
        if seqlen == 1:
          output = output[:, :, 0:1, :]
        #output = torch.matmul(scores, values)
        env.apply_sharding(output, axis=1)
      return output

