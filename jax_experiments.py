import functools
import time

import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import PositionalSharding
import torch
import torch_xla2
import torch_xla2.extra

def test7():
  """insert cache test"""
  batch, seq, heads, dim = 96, 1024,  40, 128
  sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
  sharding = sharding.reshape((1, 8, 1, 1))
  val_sharding = sharding.reshape((1, 8, 1, 1))
  caches_k = jnp.zeros(
      (batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16
  )
  caches_k_4 = jnp.zeros(
      (batch, seq, heads, dim), device=sharding.reshape((1, 1, 8, 1)), dtype=jnp.bfloat16
  )
  caches_k_2 = jnp.zeros(
      (batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16
  )
  jnp.zeros((batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16)

  def insert_cache(cache, index, new_entry):
    res = cache.at[:, :, index, :].set(
        new_entry
    )
    res = jax.lax.with_sharding_constraint(res, sharding)
    return res

  def insert_cache2(cache, index, new_entry):
    res = cache.at[jnp.arange(batch), :, index, :].set(
        new_entry
    )
    res = jax.lax.with_sharding_constraint(res, sharding)
    return res

  # pylint: disable-next=all
  def insert_cache4(cache, index, new_entry):
    res = cache.at[jnp.arange(batch), index, :, :].set(
        new_entry
    )
    #res = jax.lax.with_sharding_constraint(res, sharding)
    return res

  def insert_cache3(cache, index, new_entry):
    print(f"cache shape: {cache.shape} update shape: {new_entry.shape}")
    res = cache.at[jnp.arange(batch), :, index, :].set(
        new_entry
    )
    #res = jax.lax.with_sharding_constraint(res, sharding)
    return res
  
  #batch_indices = jax.lax.iota(jnp.int32, cache.shape[0])
  #index = jax.lax.broadcasted_iota(dtype=jnp.int32, shape=(batch, 1, seq, 1), dimension=0)

  input_pos = jnp.arange(batch, dtype=jnp.int32)[:, None, None, None]
  input_pos = jnp.broadcast_to(input_pos, (batch, 1, seq, 1))

  #batch_pos = jnp.arange(batch, dtype=jnp.int32)
  
  def insert_cache5(cache, index, new_entry):
   index = index.reshape((batch, 1, 1, 1))
   index =  jnp.broadcast_to(index, (batch, 1, seq, 1))
   new_entry = new_entry.reshape(batch, heads, 1, dim)
   res = jnp.where(index == input_pos, new_entry, cache)
   res = jax.lax.with_sharding_constraint(res, sharding)
   return res


  #def insert_cache3(cache, index, new_entry):
  #  res = jnp.where()
  insert_cache = jax.jit(insert_cache, donate_argnums=(0, 1))
  insert_cache2 = jax.jit(insert_cache2, in_shardings=(sharding.reshape((1, 8, 1, 1)), None, None), out_shardings=sharding, donate_argnums=(0, 1))
  insert_cache3 = jax.pmap(insert_cache3, in_axes=(1, None, None), out_axes=1, donate_argnums=0)
  insert_cache5 = jax.jit(insert_cache5, in_shardings=(sharding.reshape((1, 8, 1, 1)), None, None), out_shardings=sharding, donate_argnums=(0, 1))
  insert_seqlen = 1024

  subkey = jax.random.PRNGKey(234)
  to_insert = jax.device_put(
      jax.random.normal(
          subkey, (1, heads, insert_seqlen, dim), dtype=jnp.bfloat16
      ),
      device=val_sharding,
  ).block_until_ready()
  # pylint: disable-next=all
  j = jnp.int32(7).block_until_ready()

  update_indexes = (jnp.arange(-insert_seqlen, 0) + 7) % 1024
  head_indexes = jnp.arange(heads).reshape(1, -1, 1)

  rng = jax.random.PRNGKey(0)

  #jax.profiler.start_trace('/cns/pi-d/home/lancewang/tensorboard/multislice')
  #jax.profiler.start_trace('/tmp/insert_trace')
  jax.profiler.start_trace('gs://lancewang-dev/tpu-pytorch/profiling')
  for func in (insert_cache, insert_cache2, insert_cache5): #, insert_cache3):
    for _ in range(2):
      all_times = 0
      for j in range(10):
        rng, subkey = jax.random.split(rng)
        val = jax.device_put(
            jax.random.normal(
                subkey, (batch, heads, dim), dtype=jnp.bfloat16
            ),
            device=sharding.reshape((1, 8, 1)),
        ).block_until_ready()
        # pylint: disable-next=all
        j = jnp.int32(j).block_until_ready()
        if func == insert_cache2 or func == insert_cache3 or func == insert_cache5:
          j = jnp.broadcast_to(j, (batch, )).block_until_ready()
        start = time.perf_counter()
        # Swap index for cache4
        if func == insert_cache4:
          caches_k = caches_k_4
        # pylint: disable-next=all
        if func == insert_cache3:
            caches_k = caches_k.reshape((batch, 8, heads//8, seq, dim)) 
            val = jax.random.normal(
                subkey, (batch, heads//8, dim), dtype=jnp.bfloat16
            )
        caches_k = func(caches_k, j,  val)
        print(f"caches_k shape: {caches_k.shape}")
        caches_k.block_until_ready()
        end = time.perf_counter()
        all_times += end - start
      print(func.__name__, "time is", all_times)
  jax.profiler.stop_trace()

import os
os.environ['JAX_TRACEBACK_FILTERING']="False"
os.environ['XLA_FLAGS'] = "--xla_dump_to=gs://lancewang-dev/tpu-pytorch/profiling"
test7()
