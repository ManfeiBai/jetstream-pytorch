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

import torch_xla2
import jax
import jax.numpy as jnp
import torch


class CacheInterface:
    # cache for ONE layer

    def update(self, key, value):
        """Update the cache for this key and value.
        
        The key, and val will have shape (Batch, Heads, Seqlen, Head dim)
        The cache is free to store them in a different format.
        Return the full cache after update.
        This cache instance need to know which position / layer is 
        the update for.
        """

class KVCachePrefill:

    def __init__(self, kv_quantize=False):
        self.kv_quantize = kv_quantize 
        self.cache_k = None
        self.cache_v = None

    def update(self, key, value):
        """This cache just remembers the stuff."""
        self.cache_k = key
        self.cache_v = value
        if self.kv_quantize:  # pretend to be quantized
            bsz, _, seq, _ = key.shape
            ones = torch_xla2.tensor.wrap(jnp.ones((bsz, 1, seq, 1), dtype=jnp.bfloat16))
            return key, value, ones, ones
        else:
            return key, value

    def state(self):
        return self.cache_k, self.cache_v


def KVCachePrefill_flatten(cache):
    return torch_xla2.tensor.unwrap((cache.cache_k, cache.cache_v)), cache.kv_quantize


def KVCachePrefill_unflatten(auxdata, data):
    cache = KVCachePrefill(auxdata)
    cache_k, cache_v = torch_xla2.tensor.wrap(data)
    cache.cache_k = cache_k
    cache.cache_v = cache_v


jax.tree_util.register_pytree_node(
    KVCachePrefill, 
    KVCachePrefill_flatten, 
    KVCachePrefill_unflatten)




# Refactor out cache management
# Easier to test for quantized kv cache
class KVCacheGenerate:

    def __init__(self, 
        cache_k: torch.Tensor,  # previous cache
        cache_v: torch.Tensor,  # previous cache
        position: torch.Tensor,  # position to store the cache
        sharding,
    ):
        super().__init__()
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.pos = position
        if len(position.shape) > 1:
            self.pos = torch.squeeze(self.pos, 1)
        assert len(position.shape) == 1
        self.pos = torch_xla2.tensor.unwrap(self.pos)
        self.batch = jnp.arange(self.pos.shape[0])
        self.sharding = sharding

    def update(self, key, value):
        keyj = torch.squeeze(key, 2)
        valuej = torch.squeeze(value, 2)
        keyj, valuej = torch_xla2.tensor.unwrap((keyj, valuej))
        self.cache_k._elem = self.cache_k._elem.at[self.batch, :, self.pos].set(keyj)
        self.cache_v._elem = self.cache_v._elem.at[self.batch, :, self.pos].set(valuej)
        return self.cache_k, self.cache_v 

    def state(self):
        return self.cache_k._elem, self.cache_v._elem

    @classmethod
    def empty(cls, shape, device, bf16_enable):
        default_dtype = jnp.bfloat16 if bf16_enable else jnp.float32
        k = jnp.zeros(shape, device=device, dtype=default_dtype)
        v = jnp.zeros(shape, device=device, dtype=default_dtype)
        pos = jnp.zeros((shape[0]))  # replicated
        k, v, pos = torch_xla2.tensor.wrap((k, v, pos))
        return cls(k, v, pos, device)

def KVCacheGenerate_flatten(cache):
    return torch_xla2.tensor.unwrap((cache.cache_k, cache.cache_v)), (cache.pos, cache.sharding)


def KVCacheGenerate_unflatten(auxdata, data):
    position, sharding = auxdata
    cache_k, cache_v = torch_xla2.tensor.wrap(data)
    cache = KVCacheGenerate(cache_k, cache_v, position, sharding)
    return cache


jax.tree_util.register_pytree_node(
    KVCacheGenerate, 
    KVCacheGenerate_flatten, 
    KVCacheGenerate_unflatten)
        

class Int8KVCacheGenerate:

    def __init__(self, 
        cache_k, 
        cache_v, 
        cache_k_scaler,
        cache_v_scaler, 
        input_pos,  # used to write cache
        sharding = None,
    ):
        super().__init__()
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.k_scaler = cache_k_scaler 
        self.v_scaler = cache_v_scaler 
        self.input_pos_scale = input_pos
        batch, _, seq, _ = self.cache_k.shape
        self.input_pos = input_pos.reshape((batch, 1, 1, 1))
        self.input_pos = jnp.broadcast_to(self.input_pos, (batch, 1, seq, 1))
        self.index = jnp.arange(self.cache_k.shape[2]).reshape(1, 1, seq, 1)
        self.index = jnp.broadcast_to(self.index, (batch, 1, seq, 1))
        self.batch = jnp.arange(batch)
        self.input_pos_scale, self.batch, self.input_pos, self.index =  torch_xla2.tensor.wrap((self.input_pos_scale, self.batch, self.input_pos, self.index))
        #self.batch = torch_xla2.tensor.wrap(jnp.arange(input_pos.shape[0]))
        #self.batch = jnp.arange(input_pos.shape[0])
        
    def state(self):
        return torch_xla2.tensor.unwrap((self.cache_k, self.cache_v))

    
    def scalers(self):
        return torch_xla2.tensor.unwrap((self.k_scaler, self.v_scaler))

    @classmethod
    def empty(cls, shape, device, bf16_enable):
        cache_k = jnp.zeros(shape, device=device, dtype=jnp.int8)
        cache_v = jnp.zeros(shape, device=device, dtype=jnp.int8)
        # bf16_enable is a placeholder parameter, it's not used in Int8KVCache 
        kscaler = jnp.ones((shape[0], 1, shape[2], 1), dtype=jnp.bfloat16)
        vscaler = jnp.ones((shape[0], 1, shape[2], 1), dtype=jnp.bfloat16)
        input_pos = jnp.zeros((shape[0]))
        cache_k, cache_v, kscaler, vscaler = torch_xla2.tensor.wrap((cache_k, cache_v, kscaler, vscaler))
        #input_pos = torch.zeros((shape[0]))
        
        return cls(cache_k, cache_v, kscaler, vscaler, input_pos, device)


    def quantize(self, val):
        # val is (batch, heads, seqlen, dim)
        scale = torch.amax(val.abs(), axis=(1, 3), keepdim=True)
        scale = scale / 127
        return (val / scale).to(torch.int8), scale

    def update(self, xk, xv):
        k_quant, kscale = self.quantize(xk)
        v_quant, vscale = self.quantize(xv)
        #k_quant, kscale, v_quant, vscale = torch_xla2.tensor.unwrap((k_quant, kscale, v_quant, vscale))
        #self.cache_k[self.batch, :, self.input_pos, :] = torch.squeeze(k_quant, 2)
        #self.cache_v[self.batch, :, self.input_pos, :] = torch.squeeze(v_quant, 2)
        #self.k_scaler[self.batch, :, self.input_pos, :] = torch.squeeze(kscale, 2)
        #self.v_scaler[self.batch, :, self.input_pos, :] = torch.squeeze(vscale, 2)

        # Follow FP16 way of converting
        #k_quant = torch.squeeze(k_quant, 2)
        #kscale = torch.squeeze(kscale, 2)
        #v_quant = torch.squeeze(v_quant, 2)
        #vscale = torch.squeeze(vscale, 2)
        #k_quant, kscale, v_quant, vscale = torch_xla2.tensor.unwrap((k_quant, kscale, v_quant, vscale))
        #self.cache_k._elem = self.cache_k._elem.at[self.batch, :, self.input_pos, :].set(k_quant)
        #self.cache_v._elem = self.cache_v._elem.at[self.batch, :, self.input_pos, :].set(v_quant)
        #self.k_scaler._elem = self.k_scaler._elem.at[self.batch, :, self.input_pos, :].set(kscale)
        #self.v_scaler._elem = self.v_scaler._elem.at[self.batch, :, self.input_pos, :].set(vscale)
        
        # Pure Pytorch
        #self.cache_k[self.batch, :, self.input_pos, :] = k_quant
        #self.cache_v[self.batch, :, self.input_pos, :] = v_quant
        #self.k_scaler[self.batch, :, self.input_pos, :] = kscale
        #self.v_scaler[self.batch, :, self.input_pos, :] = vscale
        
        # Iota
        self.cache_k = torch.where(self.index == self.input_pos, k_quant, self.cache_k)
        self.cache_v = torch.where(self.index == self.input_pos, v_quant, self.cache_v)
        
        #self.cache_k[self.batch, :, self.input_pos_scale, :] = torch.squeeze(k_quant, 2)
        #self.cache_v[self.batch, :, self.input_pos_scale, :] = torch.squeeze(v_quant, 2)
        self.k_scaler[self.batch, :, self.input_pos_scale, :] = torch.squeeze(kscale, 2) 
        self.v_scaler[self.batch, :, self.input_pos_scale, :] = torch.squeeze(vscale, 2)
        return self.cache_k, self.cache_v, self.k_scaler, self.v_scaler
