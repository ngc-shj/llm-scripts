import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import List, Dict
import time

# argv
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--tokenizer-path", type=str, default=None)
parser.add_argument("--no-use-system-prompt", action='store_true')
parser.add_argument("--max-tokens", type=int, default=256)

args = parser.parse_args(sys.argv[1:])

if args.model_path == None:
    exit()

model_id = args.model_path
use_system_prompt = not args.no_use_system_prompt
max_new_tokens = args.max_tokens

tokenizer_id = model_id
if args.tokenizer_path:
    tokenizer_id = args.tokenizer_path

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_id,
    trust_remote_code=True
)
is_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    #torch_dtype=torch.bfloat16,
    device_map="auto",
    #device_map="cuda",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
#if torch.cuda.is_available():
#    model = model.to("cuda")

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"


def q(
    user_query: str,
    history: List[Dict[str, str]]=None
) -> List[Dict[str, str]]:
    # generation params
    generation_params = {
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.1,
    }
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
        return_tensors="pt"
    )
    print("--- prompt")
    print(prompt)
    print("--- output")
    # 推論
    output_ids = model.generate(
        input_ids.to(model.device),
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        **generation_params
    )
    output = tokenizer.decode(
        output_ids[0][input_ids.size(1) :],
        skip_special_tokens=True
    )
    if is_chat:
        user_messages.append(
            {"role": "assistant", "content": output}
        )
    else:
        user_messages += output
    end = time.time()
    ##
    input_tokens = len(input_ids[0])
    output_tokens = len(output_ids[0][input_ids.size(1) :])
    total_time = end - start
    tps = output_tokens / total_time
    print(f"prompt tokens = {input_tokens:.7g}")
    print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
    print(f"   total time = {total_time:f} [s]")
    return user_messages

print('history = ""')
print('history = q("ドラえもんとはなにか")')
print('history = q("続きを教えてください", history)')

