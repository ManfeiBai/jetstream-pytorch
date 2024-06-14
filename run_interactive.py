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

import os
import random
import time

from typing import List
from absl import app
from absl import flags
from colorama import Fore, Style

import jax

from jetstream.engine import token_utils
from colorama import Fore, Style
import numpy as np

import os

from jetstream_pt import engine as je

FLAGS = flags.FLAGS

_TOKENIZER_PATH = flags.DEFINE_string(
    "tokenizer_path",
    "tokenizer.model",
    "The tokenizer model path",
    required=False,
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", None, "model type", required=False
)
_CKPT_PATH = flags.DEFINE_string(
    "checkpoint_path", None, "Directory for .pth checkpoints", required=False
)
_BF16_ENABLE = flags.DEFINE_bool(
    "bf16_enable", False, "Whether to enable bf16", required=False
)
_CONTEXT_LENGTH = flags.DEFINE_integer(
    "context_length", 1024, "The context length", required=False
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 32, "The batch size", required=False
)
_PROFILING_OUTPUT = flags.DEFINE_string(
    "profiling_output",
    "",
    "The profiling output",
    required=False,
)

_SIZE = flags.DEFINE_string("size", "tiny", "size of model")

_QUANTIZE_WEIGHTS = flags.DEFINE_bool(
    "quantize_weights", False, "weight quantization"
)
_QUANTIZE_KV_CACHE = flags.DEFINE_bool(
    "quantize_kv_cache", False, "kv_cache_quantize"
)
_MAX_CACHE_LENGTH = flags.DEFINE_integer(
    "max_cache_length", 1024, "kv_cache_quantize"
)
_SHARDING_CONFIG = flags.DEFINE_string(
    "sharding_config", "", "config file for sharding"
)
_SHARD_ON_BATCH = flags.DEFINE_bool(
    "shard_on_batch",
    False,
    "whether to shard on batch dimension."
    "If set true, sharding_config will be ignored.",
)


def create_engine():
  """create a pytorch engine"""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  devices = jax.devices()
  start = time.perf_counter()
  engine = je.create_pytorch_engine(
      model_name=_MODEL_NAME.value,
      devices=devices,
      tokenizer_path=_TOKENIZER_PATH.value,
      ckpt_path=_CKPT_PATH.value,
      bf16_enable=True,
      param_size=_SIZE.value,
      context_length=_CONTEXT_LENGTH.value,
      batch_size=_BATCH_SIZE.value,
      quantize_weights=_QUANTIZE_WEIGHTS.value,
      quantize_kv=_QUANTIZE_KV_CACHE.value,
      max_cache_length=_MAX_CACHE_LENGTH.value,
      sharding_config=_SHARDING_CONFIG.value,
      shard_on_batch=_SHARD_ON_BATCH.value,
  )

  print("Initialize engine", time.perf_counter() - start)
  return engine


def do_simulation(prompts, replys, prefill_bucket_size_to_ms, system_time_per_decode_token_ms):
  # import pdb; pdb.set_trace()
  def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

  def tokens_in_input_str(s):
    # import pdb; pdb.set_trace()
    return_val = int(1.3 * len(s.split()))
    return return_val

  convo_numbers = []
  # Please update with your own data file path

  # with open(sharegpt_path, "r", encoding="utf-8") as f:
  #   loaded_share_gpt = json.load(f)
  # for example in prompts:
  for i in range(len(prompts)):
    # if len(example["conversations"]) < 2:
    #   continue
    input_tokens = tokens_in_input_str(prompts[i])
    # import pdb; pdb.set_trace()
    output_tokens = tokens_in_input_str(replys[i]) # .text)
    convo_numbers.append((input_tokens, output_tokens))

  num_convos = len(convo_numbers)
  kept_convos = [
      c for c in convo_numbers # if c[0] <= CUTOFF_INPUT and c[1] <= CUTOFF_OUTPUT # CUTOFF_INPUT = 1024 # CUTOFF_OUTPUT = 1024
  ]

  mean_input = sum(c[0] for c in kept_convos) / len(kept_convos)
  mean_output = sum(c[1] for c in kept_convos) / len(kept_convos)

  print(
      f"""Total {num_convos=} but only kept {kept_convos=}. 
    Out of kept, {mean_input=}, {mean_output=}"""
  )

  total_prefill_system_ms = 0
  total_generate_system_ms = 0

  # import pdb; pdb.set_trace()
  iteri = 0
  for convo in kept_convos:
    input_tok, output_tok = convo
    # bucket = max(128, next_power_of_2(input_tok))
    bucket = max(7, next_power_of_2(input_tok))
    generate_system_ms = output_tok * system_time_per_decode_token_ms
    prefill_system_ms = prefill_bucket_size_to_ms[iteri] # [bucket]
    iteri += 1

    print(
        f"{convo=} {bucket=}, {prefill_system_ms=:.2f}, {generate_system_ms=:.2f}"
    )

    total_prefill_system_ms += prefill_system_ms
    total_generate_system_ms += generate_system_ms

  total_time_ms = total_prefill_system_ms + total_generate_system_ms
  input_tokens = sum(c[0] for c in kept_convos)

  output_tokens = sum(c[1] for c in kept_convos)
  print(
      f"""Output tokens {output_tokens} in {total_time_ms/1000:.2f} seconds, 
      for {output_tokens/(total_time_ms/1000):.2f} out tok/s"""
  )

  total_prefill_sec = total_prefill_system_ms / 1000
  total_generate_sec = total_generate_system_ms / 1000

  print(
      f"""Total time {total_time_ms/1000:.2f} seconds, 
      split {total_prefill_sec=:.2f} seconds and {total_generate_sec=:.2f} seconds"""
  )

  idealized_prefill_sec = (
      1.1 * input_tokens / 1024 * 60.28 / 1000 # 1024: 60.28,
  )

  prefill_savings_sec = total_prefill_sec - idealized_prefill_sec

  idealized_generate_sec = (
      total_generate_sec / 2
  )  # (Roughly save 75% on KV cache high cost on the rest)
  generate_savings_sec = total_generate_sec - idealized_generate_sec

  print(
      f"""we think prefill will take {total_prefill_sec=:.2f}, 
    we could get it to {idealized_prefill_sec=:.2f} so we'd 
    save {prefill_savings_sec=:.2f} seconds """
  )
  print(
      f"""with sparsity we could go from  {total_generate_sec=:.2f}, 
    we could get it to {idealized_generate_sec=:.2f} so we'd save 
    {generate_savings_sec=:.2f} seconds """
  )

  idealized_overall_time = idealized_generate_sec + idealized_prefill_sec

  print(
      f"""Idealized out tokens {output_tokens} in {idealized_overall_time:.2f} seconds, 
    for {output_tokens/idealized_overall_time:.2f} out tok/s"""
  )
  print("prfill", prefill_bucket_size_to_ms)
  print("decode step", system_time_per_decode_token_ms)


# pylint: disable-next=all
def main(argv):

  # main_start_time = time.time()
  engine = create_engine()

  start = time.perf_counter()
  params = engine.load_params()
  print("Load params ", time.perf_counter() - start)

  metadata = engine.get_tokenizer()
  tokenizer = engine.build_tokenizer(metadata)
  max_output_length = 20

  if _PROFILING_OUTPUT.value:
    jax.profiler.start_trace(_PROFILING_OUTPUT.value)

  decode_state = engine.init_decode_state()

  main_start_time = time.time()
  prompts: List[str] = [
      "I believe the meaning of life is",
      "To add an element to an ArrayList of a specific class type in Java",
      "you can follow the following steps:.",
      "Create an instance of the class to be added.\n2.",
      "Get a reference to the ArrayList.\n3.", 
      "Call the `add()` method on the ArrayList,",
      "passing the instance of the class as the argument.",
      "Here's an example of how to add an object of type `Person` to an ArrayList of type `ArrayList<Person>`:",
      "```csharp\n// Create a new instance of the Person class\nPerson person = new Person(\"John\", 25);",
      "// Get a reference to the ArrayList\nArrayList<Person> peopleList = new ArrayList<>();",
      "Add the person object to the ArrayList peopleList",
      "In this example, the `Person` class is assumed to have a constructor that takes two arguments:",
      "a String for the person's name, and an int for their age. You can substitute your own class and constructor as necessary.",
      "You are an AI assistant. User will you give you a task.",
      "Your goal is to complete the task as faithfully as you can.",
      "While performing the task think step-by-step and justify your steps.\n<</SYS>>",
      "Question 1: What is commercial real estate finance?",
      "Question 2: What are Commercial Real Estate services?",
      "Options are:\n[a]. no.\n[b]. yes.\nWould the answer to these two questions be the same?",
      "You are an AI assistant that helps people find information.",
      "Provide a detailed answer so user don\u2019t need to search outside to understand the answer.",
      "Use reasoning to lead to the answer of the following question:\nWhere are you likely to find water underneath?",
      "Options:\n- toilet\n- sink\n- jar\n- bridge\n- house\n Reasoning process:",
      "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
      "Continue the following story.\n\nKay didn't have shoes that fit her feet properly.",
      "She only wore sneakers, because the \nChoose from: [I] shoes  fitted badly. [II] sneakers  fitted badly",
  ]

  prefill_times = {}
  dec_times = []
  replys = []
  for prompt in prompts:
    slot = random.randint(0, _BATCH_SIZE.value - 1)
    tokens, true_length = tokenizer.encode(prompt)

    # print(f"---- Input prompts are: {prompt}")
    # print(f"---- Encoded tokens are: {tokens}")

    # pylint: disable-next=all
    start_time_prefill = time.time()
    prefill_result = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    prefill_time = time.time() - start_time_prefill
    print("--- prefill time : %s seconds ---" % prefill_time, " for prompt: ", prompt)
    prefill_times[prompt] = prefill_time

    # pylint: disable-next=all
    start_time_decode = time.time()
    decode_state = engine.insert(prefill_result, decode_state, slot=slot)
    sampled_tokens_list = []
    # print(f"---- Streaming decode started on #slot{slot}.")
    complete = np.zeros((1,), dtype=np.bool_)
    while True:
      decode_state, result_tokens = engine.generate(params, decode_state)
      result_tokens = result_tokens.convert_to_numpy()
      res = result_tokens.get_result_at_slot(slot)
      stop_tokens = set(tokenizer.stop_tokens)
      stop_tokens.add(tokenizer.pad_id)
      token_id = res.tokens[0][0].item()
      sampled_tokens_list.append(token_id)
      if (
          token_id in stop_tokens
          or len(sampled_tokens_list) > max_output_length
      ):
        break
    decode_time = time.time() - start_time_decode
    dec_times.append(decode_time)
    replys.append(tokenizer.decode(sampled_tokens_list))
    print("--- finish all decode used : %s seconds ---" % decode_time)

    # print("--- finish all requests used : %s seconds ---" % (time.time() - start_time))
    # print("---- All output tokens.")
    # print(sampled_tokens_list)
    # print("---- All output text.")
    # print("---- All output text.", tokenizer.decode(sampled_tokens_list))

  print("--- finish all prompt requests used : %s seconds ---" % (time.time() - main_start_time))
  print("decode", sum(dec_times) / 10)

  # prefill_times_ms = {k: v * 1000 for k, v in prefill_times.items()}
  prefill_times_ms = [v for k, v in prefill_times.items()]
  decode_time_ms = sum(dec_times) * 1000 / 10 / 1 # FLAGS.batch_size

  # call fun
  do_simulation(prompts, replys, prefill_times_ms, decode_time_ms)

  if _PROFILING_OUTPUT.value:
    jax.profiler.stop_trace()


if __name__ == "__main__":
  start_time = time.time()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
  print("--- finish all requests used : %s seconds ---" % (time.time() - start_time))
