import sys
import os
import argparse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, llama_chat_format
from typing import List, Dict
import time

# argv
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--ggml-model-path", type=str, default=None)
parser.add_argument("--ggml-model-file", type=str, default=None)
parser.add_argument("--no-chat", action='store_true')
parser.add_argument("--no-use-system-prompt", action='store_true')
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--n-ctx", type=int, default=2048)
parser.add_argument("--n-threads", type=int, default=1)
parser.add_argument("--n-gpu-layers", type=int, default=-1)

args = parser.parse_args(sys.argv[1:])

## check and set args
if args.model_path == None:
    exit()
if args.ggml_model_path == None:
    exit()
if args.ggml_model_file == None:
    exit()

model_id = args.model_path
is_chat = not args.no_chat
use_system_prompt = not args.no_use_system_prompt
max_new_tokens = args.max_tokens
n_ctx = args.n_ctx
n_threads = args.n_threads
n_gpu_layers = args.n_gpu_layers

## Check if the GGUF model exists locally, if not download it
local_model_path = os.path.join(args.ggml_model_path, args.ggml_model_file)
if os.path.isfile(local_model_path):
    ggml_model_path = local_model_path
else:
    ## Download the GGUF model
    ggml_model_path = hf_hub_download(
        args.ggml_model_path,
        filename=args.ggml_model_file
    )

# Instantiate chat format and handler
chat_formatter = llama_chat_format.hf_autotokenizer_to_chat_formatter(model_id)
chat_handler = llama_chat_format.hf_autotokenizer_to_chat_completion_handler(model_id)

## Instantiate model from downloaded file
model = Llama(
    model_path=ggml_model_path,
    chat_handler=chat_handler,
    n_ctx=n_ctx,
    n_threads=n_threads,
    n_gpu_layers=n_gpu_layers
)

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

def q(
    user_query: str,
    history: List[Dict[str, str]]=None
) -> List[Dict[str, str]]:
    # generation params
    # https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py#L1268
    generation_params = {
        #"do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": max_new_tokens,
        "repeat_penalty": 1.1,
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
        prompt = chat_formatter(messages=messages)
    else:
        prompt = messages
    # debug
    print("--- messages")
    print(messages)
    print("--- prompt")
    print(prompt)
    print("--- output")
    # 推論
    if is_chat:
        outputs = model.create_chat_completion(
            messages=messages,
            #echo=True,
            #stream=True,
            **generation_params
        )
        output = outputs["choices"][0]["message"]["content"]
        user_messages.append(
            {"role": "assistant", "content": output}
        )
    else:
        outputs = model.create_completion(
            prompt=prompt,
            #echo=True,
            #stream=True,
            **generation_params
        )
        output = outputs["choices"][0]["text"]
        #for output in outputs:
        #    print(output["choices"][0]["text"], end='')
        user_messages += output
    print(output)
    end = time.time()
    ##
    input_tokens = outputs["usage"]["prompt_tokens"]
    output_tokens = outputs["usage"]["completion_tokens"]
    total_time = end - start
    tps = output_tokens / total_time
    print(f"prompt tokens = {input_tokens:.7g}")
    print(f"output tokens = {output_tokens:.7g} ({tps:f} [tps])")
    print(f"   total time = {total_time:f} [s]")
    return user_messages

print('history = ""')
print('history = q("ドラえもんとはなにか")')
print('history = q("続きを教えてください", history)')

