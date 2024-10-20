import sys
import argparse
from vllm import LLM, SamplingParams
from typing import List, Dict
import time

# argv
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--no-use-system-prompt", action='store_true')
parser.add_argument("--max-model-len", type=int, default=32768)
parser.add_argument("--tensor-parallel-size", type=int, default=1)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
parser.add_argument("--max-tokens", type=int, default=4096)

args = parser.parse_args(sys.argv[1:])

if args.model_path == None:
    exit()

model_id = args.model_path
use_system_prompt = not args.no_use_system_prompt
max_new_tokens = args.max_tokens
tensor_parallel_size = args.tensor_parallel_size
max_model_len = args.max_model_len
gpu_memory_utilization = args.gpu_memory_utilization

# トークナイザーとモデルの準備
model = LLM(
    model=model_id,
    dtype="auto",
    trust_remote_code=True,
    tensor_parallel_size=tensor_parallel_size,
    max_model_len=max_model_len,
    #quantization="awq",
    gpu_memory_utilization=gpu_memory_utilization
)
tokenizer = model.get_tokenizer()
is_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

def q(
    user_query: str,
    history: List[Dict[str, str]]=None
) -> List[Dict[str, str]]:
    # generation params
    generation_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        max_tokens=max_new_tokens,
        repetition_penalty=1.1
    )
    #
    start = time.time()
    # messages
    messages = ""
    if is_chat:
        messages = []
        if use_system_prompt:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            ]
        user_messages = [
            {"role": "user", "content": user_query}
        ]
    else:
        user_messages = user_query
    if history:
        user_messages = history + user_messages
    messages += user_messages
    # generation prompts
    if is_chat:
        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        prompt = messages
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=True,
    )
    print("--- prompt")
    print(prompt)
    print("--- output")
    # 推論
    outputs = model.generate(
        sampling_params=generation_params,
        prompt_token_ids=[input_ids],
    )
    print(outputs)
    output = outputs[0]
    print(output.outputs[0].text)
    if is_chat:
        user_messages.append(
            {"role": "assistant", "content": output.outputs[0].text}
        )
    else:
        user_messages += output.outputs[0].text
    end = time.time()
    ##
    input_tokens = len(output.prompt_token_ids)
    output_tokens = len(output.outputs[0].token_ids)
    total_time = end - start
    tps = output_tokens / total_time
    print(f"prompt tokens = {input_tokens:.7g}")
    print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
    print(f"   total time = {total_time:f} [s]")
    return user_messages

print('history = ""')
print('history = q("ドラえもんとはなにか")')
print('history = q("続きを教えてください", history)')

