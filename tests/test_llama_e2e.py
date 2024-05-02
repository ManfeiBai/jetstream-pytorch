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

import torch
import os
from torch.utils import _pytree as pytree
import torch_xla2
import jax
import jax.numpy as jnp
import numpy as np
from jetstream_pt.engine import PyTorchEngine
from jetstream_pt.third_party.llama2 import model_exportable
from jetstream_pt.third_party.llama2.generation_original import LlamaOriginal
from jetstream_pt import environment


import unittest
#os.environ["JAX_PLATFORM_NAME"] = "cpu"

class LlamaE2ETest(unittest.TestCase):

    def setup(self):
        jax.config.update('jax_platform_name', 'cpu')
        torch.set_default_dtype(torch.bfloat16)

    def _to_jax(self, tree):
        return pytree.tree_map_only(
            torch.Tensor,
            torch_xla2.tensor.t2j, tree)    

    def _make_env(self, bf16_enable=True):
        torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
        torch.set_default_dtype(torch_dtype)
        jax.config.update('jax_dynamic_shapes', False)
        jax.config.update('jax_traceback_filtering', 'off')
        jax.config.update('jax_platform_name', 'cpu')
        env_data = environment.JetEngineEnvironmentData()
        env_data.max_input_sequence_length = 128
        env_data.cache_sequence_length = 128
        env_data.model_type = 'llama-2-tiny'
        env_data.batch_size = 2 
        env_data.bf16_enable = bf16_enable
        env = environment.JetEngineEnvironment(env_data)
        env.apply_sharding = lambda *args, **kwargs: None  # don't shard on cpu
        return env


    def test_original_llama2_seed(self):
        jax.config.update('jax_platform_name', 'cpu')
        x = jnp.square(2)
        print(f"---------> {jax.devices()}")
        torch.set_default_dtype(torch.bfloat16)
        env = self._make_env()
        model_arg = env._model_arg 
        tokens = np.arange(10, dtype=np.int32)
        file_dir = os.path.dirname(__file__)
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')
        output_tokens_multiple = []
        for i in [1, 999, 99999]:
            llama_original = LlamaOriginal.build(tokenizer_path, model_arg, i)
            prompt_tokens = [tokens]
            output_tokens = llama_original.generate(prompt_tokens, 10)
            output_tokens_multiple.append(output_tokens)

        for index, output_tokens in enumerate(output_tokens_multiple): 
            print(f"------------------- index: {index}, tokens:{output_tokens}")
            if index > 0:
                self.assertNotEqual(output_tokens_multiple[index], output_tokens_multiple[index - 1])
    def test_jetstream_llama2_seed(self):
        jax.config.update('jax_platform_name', 'cpu')
      #with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.square(2)
        print(f"---------> {jax.devices()}")

        torch.set_default_dtype(torch.bfloat16)
        env = self._make_env()
        model_arg = env._model_arg 
        tokens = np.arange(10, dtype=np.int32)
        true_length = tokens.shape[-1]
        padded_tokens = np.pad(tokens, (0, 6))
        padded_tokens = jnp.array(padded_tokens)

        seed = 1
        max_output_length = 10

        file_dir = os.path.dirname(__file__)
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')


        seed = 1
        # orginal
        llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
        model_orig = llama_original.model

        state_dict = dict(model_orig.state_dict())
        state_dict['freqs_cis'] = model_orig.freqs_cis
        params = self._to_jax(state_dict)

        output_tokens_multiple = []
        for i in [1, 2, 3]:
            torch.manual_seed(1)
            model_ours = model_exportable.Transformer(model_arg, env)
            engine = PyTorchEngine(
                pt_model=model_ours,
                env=env
            )
            
            decode_state = engine.init_decode_state()
            slot = 0 
            prefill_result = engine.prefill(
                params=params, padded_tokens=padded_tokens, true_length=true_length
            )

            decode_state = engine.insert(
                prefill_result, decode_state, slot=slot
            )

            out_tokens = []
            while True:
                decode_state, result_tokens = engine.generate(params, decode_state)
                slot_data = result_tokens.get_result_at_slot(slot)
                slot_tokens = slot_data.tokens
                slot_lengths = slot_data.lengths
                
                token_id = slot_tokens[slot, 0].item()
                out_tokens.append(token_id)
                if slot_lengths > max_output_length:
                    break

            output_tokens_multiple.append(out_tokens)

        for index, output_tokens in enumerate(output_tokens_multiple): 
            print(f"------------------- index: {index}, tokens:{output_tokens}")
            if index > 0:
                self.assertEqual(output_tokens_multiple[index], output_tokens_multiple[index - 1])

    def _llama_e2e(self, env):
        jax.config.update('jax_platform_name', 'cpu')
        token_true_len = 16
        env.seq_len = token_true_len
        model_arg = env._model_arg 
        #tokens = np.arange(10, dtype=np.int32)
        tokens = np.arange(token_true_len, dtype=np.int32)
        true_length = tokens.shape[-1]
        padded_tokens = tokens
        #padded_tokens = np.pad(tokens, (0, 6))
        padded_tokens = jnp.array(padded_tokens)

        seed = 1
        torch.manual_seed(1)
        max_output_length = 32

        file_dir = os.path.dirname(__file__)
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')

        # orginal
        llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
        prompt_tokens = [tokens]
        expected_output_tokens, logits_all = llama_original.generate(prompt_tokens, max_output_length)


        model_orig = llama_original.model

        state_dict = dict(model_orig.state_dict())
        state_dict['freqs_cis'] = model_orig.freqs_cis


        model_ours = model_exportable.Transformer(model_arg, env)
        
        engine = PyTorchEngine(
            pt_model=model_ours,
            env=env
        )

        params = self._to_jax(state_dict)
        decode_state = engine.init_decode_state()
        slot = 0 

        #prefill_result, prefill_logits
        prefill_result = engine.prefill(
            params=params, padded_tokens=padded_tokens, true_length=true_length
        )
        from jetstream_pt import engine as e
        decode_state = e.DecodeState(decode_state.tokens, decode_state.caches, decode_state.cache_scales, prefill_result.seq_len, decode_state.lens, decode_state.input_pos, decode_state.mask)
        decode_state = engine.insert(
                prefill_result, decode_state, slot=slot
            )
        #prefill_logits = torch_xla2.tensor.wrap(prefill_logits)
        #self.assertAlmostEqual(prefill_logits, logits_all[0], places=3) 
        #np.testing.assert_allclose(prefill_logits, torch.squeeze(logits_all[0], 0), atol=1e-3)
        out_tokens = []
        step = 1
        while True:
            #decode_state, result_tokens, decode_logits
            decode_state, result_tokens = engine.generate(params, decode_state)
            #np.testing.assert_allclose(jnp.squeeze(decode_logits), torch.squeeze(logits_all[step]), atol=1e-3, err_msg=f"step: {step}")
            #step = step + 1
            slot_data = result_tokens.get_result_at_slot(slot)
            slot_tokens = slot_data.tokens
            slot_lengths = slot_data.lengths
            
            token_id = slot_tokens[slot, 0].item()
            out_tokens.append(token_id)
            if slot_lengths > max_output_length:
                break
        return  out_tokens, expected_output_tokens[0]

    def test_llama_e2e_float32(self):
        jax.config.update('jax_platform_name', 'cpu')
        print(f"---------> {jax.devices()}")

        env = self._make_env(bf16_enable=False)
        out_tokens ,expected_output_tokens = self._llama_e2e(env)
        print(f"------------------- actual: {out_tokens}")
        print(f"------------------- expcted:{expected_output_tokens}")
        self.assertEqual(out_tokens ,expected_output_tokens)


    def test_llama_e2e_bfloat16(self):
        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
        print(f"---------> {jax.devices()}")

        env = self._make_env(bf16_enable=True)
        out_tokens ,expected_output_tokens = self._llama_e2e(env)
        self.assertNotEqual(out_tokens ,expected_output_tokens)


    def test_llama_e2e_two_addtional_tokens(self):
        jax.config.update('jax_platform_name', 'cpu')
        x = jnp.square(2)
        print(f"---------> {jax.devices()}")

        torch.set_default_dtype(torch.bfloat16)
        env = self._make_env()
        model_arg = env._model_arg 
        tokens = np.arange(10, dtype=np.int32)
        tokens = np.append(tokens, [15050, 3503], axis=-1)
        true_length = tokens.shape[-1]
        padded_tokens = np.pad(tokens, (0, 6))
        padded_tokens = jnp.array(padded_tokens)

        seed = 1
        torch.manual_seed(1)
        max_output_length = 10

        file_dir = os.path.dirname(__file__)
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')

        # orginal
        llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
        prompt_tokens = [tokens]
        expected_output_tokens = llama_original.generate(prompt_tokens, max_output_length)


        model_orig = llama_original.model

        state_dict = dict(model_orig.state_dict())
        state_dict['freqs_cis'] = model_orig.freqs_cis


        model_ours = model_exportable.Transformer(model_arg, env)
        
        engine = PyTorchEngine(
            pt_model=model_ours,
            env=env
        )

        params = self._to_jax(state_dict)
        decode_state = engine.init_decode_state()
        slot = 0 

        prefill_result = engine.prefill(
            params=params, padded_tokens=padded_tokens, true_length=true_length
        )

        decode_state = engine.insert(
            prefill_result, decode_state, slot=slot
        )

        out_tokens = []
        while True:
            decode_state, result_tokens = engine.generate(params, decode_state)
            slot_data = result_tokens.get_result_at_slot(slot)
            slot_tokens = slot_data.tokens
            slot_lengths = slot_data.lengths
            
            token_id = slot_tokens[slot, 0].item()
            out_tokens.append(token_id)
            if slot_lengths > max_output_length:
                break
        print(f"-------------------->out_tokens:{out_tokens}")
        print(f"-------------------->expected_output_tokens:{expected_output_tokens}")
        # self.assertEqual(out_tokens ,expected_output_tokens)


    def test_llama_e2e_four_addtional_tokens(self):
        jax.config.update('jax_platform_name', 'cpu')
        x = jnp.square(2)
        print(f"---------> {jax.devices()}")

        torch.set_default_dtype(torch.bfloat16)
        env = self._make_env()
        model_arg = env._model_arg 
        tokens = np.arange(10, dtype=np.int32)
        tokens = np.append(tokens, [15050, 3503, 11833, 28551], axis=-1)
        true_length = tokens.shape[-1]
        padded_tokens = np.pad(tokens, (0, 6))
        padded_tokens = jnp.array(padded_tokens)

        seed = 1
        torch.manual_seed(1)
        max_output_length = 10

        file_dir = os.path.dirname(__file__)
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')

        # orginal
        llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
        prompt_tokens = [tokens]
        expected_output_tokens = llama_original.generate(prompt_tokens, max_output_length)


        model_orig = llama_original.model

        state_dict = dict(model_orig.state_dict())
        state_dict['freqs_cis'] = model_orig.freqs_cis


        model_ours = model_exportable.Transformer(model_arg, env)
        
        engine = PyTorchEngine(
            pt_model=model_ours,
            env=env
        )

        params = self._to_jax(state_dict)
        decode_state = engine.init_decode_state()
        slot = 0 

        prefill_result = engine.prefill(
            params=params, padded_tokens=padded_tokens, true_length=true_length
        )

        decode_state = engine.insert(
            prefill_result, decode_state, slot=slot
        )

        out_tokens = []
        while True:
            decode_state, result_tokens = engine.generate(params, decode_state)
            slot_data = result_tokens.get_result_at_slot(slot)
            slot_tokens = slot_data.tokens
            slot_lengths = slot_data.lengths
            
            token_id = slot_tokens[slot, 0].item()
            out_tokens.append(token_id)
            if slot_lengths > max_output_length:
                break
        print(f"-------------------->out_tokens:{out_tokens}")
        print(f"-------------------->expected_output_tokens:{expected_output_tokens}")
        # self.assertEqual(out_tokens ,expected_output_tokens)        

if __name__ == '__main__':
    #os.environ["JAX_PLATFORM_NAME"] = "cpu"
    unittest.main()


