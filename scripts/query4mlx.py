import sys
import argparse
from mlx_lm import load, generate, stream_generate
from typing import List, Dict
import time

# argv
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--tokenizer-path", type=str, default=None)
parser.add_argument("--no-use-system-prompt", action='store_true')
parser.add_argument("--max-tokens", type=int, default=4096)

args = parser.parse_args(sys.argv[1:])

if args.model_path == None:
    exit()

model_id = args.model_path
use_system_prompt = not args.no_use_system_prompt
max_tokens = args.max_tokens

model, tokenizer = load(path_or_hf_repo=model_id)
is_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

def q(
    user_query: str,
    history: List[Dict[str, str]]=None
) -> List[Dict[str, str]]:
    # generation params
    generation_params = {
        "temp": 0.8,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.1,
        "repetition_context_size": 20,
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
    output=""
    for t in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        **generation_params
    ):
        print(t, end="", flush=True)
        output += t
    print()
    output_ids = tokenizer.encode(
        output,
        add_special_tokens=True,
        return_tensors="pt"
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
    output_tokens = len(output_ids[0])
    total_time = end - start
    tps = output_tokens / total_time
    print(f"prompt tokens = {input_tokens:.7g}")
    print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
    print(f"   total time = {total_time:f} [s]")
    return user_messages

print('history = ""')
print('history = q("ドラえもんとはなにか")')
print('history = q("続きを教えてください", history)')

